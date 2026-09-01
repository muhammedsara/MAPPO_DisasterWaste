"""
train_mappo.py — Domain-randomised MAPPO training.

Every training episode faces a fresh stochastic realization of the damage/waste
processes on the fixed training topology (env._randomize_on_reset), which prevents
the policy from over-fitting a single realization. Model selection ("best") uses a
held-out set of evaluation seeds that are disjoint from those used at test time.

Reusable for:
  * main robust models        (default weights, capacity_scale=1)
  * reward-weight sensitivity (--reward-weights)
  * fleet-capacity scaling    (--capacity-scale)

Saves <out_dir>/mappo_best.pt, mappo_final.pt, train_log.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.environment import DisasterWasteEnv, ScenarioGenerator, ScenarioTier
from src.agents import MAPPO, MAPPOConfig

TIER_MAP = {
    "S1_SMALL": ScenarioTier.S1_SMALL,
    "S2_MEDIUM": ScenarioTier.S2_MEDIUM,
    "S3_LARGE": ScenarioTier.S3_LARGE,
    "S4_SEVERE": ScenarioTier.S4_SEVERE,
}


def eval_reward(env, mappo, seeds) -> float:
    """Mean deterministic reward over held-out realization seeds."""
    tot = []
    for s in seeds:
        obs_dict, _ = env.reset(seed=s)
        agents = env.possible_agents
        r = 0.0
        done = False
        while not done:
            obs = np.array([obs_dict[a]["obs"] for a in agents], dtype=np.float32)
            msk = np.array([obs_dict[a]["action_mask"] for a in agents], dtype=np.float32)
            with torch.no_grad():
                act, _, _ = mappo.actor.get_action(
                    torch.tensor(obs, device=mappo.device),
                    torch.tensor(msk, device=mappo.device),
                    deterministic=True)
            obs_dict, rew, term, trunc, _ = env.step(
                {a: int(act[i].item()) for i, a in enumerate(agents)})
            r += sum(rew.values()) / len(rew)
            done = any(trunc.values()) or any(term.values())
        tot.append(r)
    return float(np.mean(tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(TIER_MAP))
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--reward-weights", default="0.25,0.25,0.25,0.25",
                    help="cost,time,emission,recycling")
    ap.add_argument("--capacity-scale", type=float, default=1.0)
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--mask-mode", default="hard", choices=["hard", "soft", "none"],
                    help="hard=exposed feasibility mask; soft=permissive+penalty; none=permissive")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-interval", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wc, wt, we, wr = [float(x) for x in args.reward_weights.split(",")]
    reward_weights = {"cost": wc, "time": wt, "emission": we, "recycling": wr}

    tier = TIER_MAP[args.scenario]
    scenario = ScenarioGenerator(seed=42).from_tier(tier)

    # Optional fleet-capacity scaling: scale every vehicle capacity.
    if abs(args.capacity_scale - 1.0) > 1e-9:
        for veh in scenario.vehicles:
            veh.config.capacity *= args.capacity_scale
        print(f"[cap] scaled vehicle capacities x{args.capacity_scale}")

    train_env = DisasterWasteEnv(scenario=scenario, seed=42, reward_weights=reward_weights)
    train_env._mask_mode = args.mask_mode
    if not args.no_randomize:
        train_env._randomize_on_reset = True
        train_env._reset_rng = np.random.default_rng(args.train_seed)
        print("[DR] domain randomization ENABLED (per-episode fresh realizations)")
    if args.mask_mode != "hard":
        print(f"[mask] mode={args.mask_mode}")

    # Separate eval env on the same topology (held-out realization seeds).
    eval_scenario = ScenarioGenerator(seed=42).from_tier(tier)
    if abs(args.capacity_scale - 1.0) > 1e-9:
        for veh in eval_scenario.vehicles:
            veh.config.capacity *= args.capacity_scale
    eval_env = DisasterWasteEnv(scenario=eval_scenario, seed=42, reward_weights=reward_weights)
    eval_env._mask_mode = args.mask_mode
    eval_seeds = [90_001 + i for i in range(5)]

    cfg = MAPPOConfig(
        lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
        n_epochs=4, mini_batch_size=64, rollout_length=128,
        total_timesteps=args.timesteps, use_linear_lr_decay=True,
        save_interval=0, log_interval=20, eval_interval=args.eval_interval,
    )
    mappo = MAPPO(n_agents=len(train_env.possible_agents),
                  obs_dim=train_env._local_obs_dim,
                  state_dim=train_env._global_state_dim,
                  action_dim=train_env._action_size,
                  config=cfg, device=device)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[{args.scenario}] out={out} weights={reward_weights} "
          f"cap_scale={args.capacity_scale} steps={args.timesteps}")

    best = {"reward": -1e18, "update": -1}
    hist = {"update": [], "steps": [], "train_reward": [], "eval_reward": []}
    t0 = time.time()

    def cb(agent, update_count, stats):
        hist["update"].append(update_count)
        hist["steps"].append(agent.total_steps)
        hist["train_reward"].append(None)
        if update_count % cfg.eval_interval == 0:
            er = eval_reward(eval_env, agent, eval_seeds)
            hist["eval_reward"].append(er)
            if er > best["reward"]:
                best["reward"] = er
                best["update"] = update_count
                agent.save(str(out / "mappo_best.pt"))
            print(f"    [eval] update={update_count} steps={agent.total_steps:,} "
                  f"eval_reward={er:.2f} best={best['reward']:.2f}", flush=True)
        else:
            hist["eval_reward"].append(None)

    mappo.train(train_env, total_timesteps=args.timesteps, callback=cb)
    mappo.save(str(out / "mappo_final.pt"))
    if best["update"] < 0:  # ensure a best exists
        mappo.save(str(out / "mappo_best.pt"))

    json.dump({"args": vars(args), "best": best,
               "elapsed_s": time.time() - t0, "history": hist},
              open(out / "train_log.json", "w"), indent=2)
    print(f"[done] {args.scenario} best_eval={best['reward']:.2f} "
          f"@update {best['update']} in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
