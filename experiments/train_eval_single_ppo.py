"""
train_eval_single_ppo.py — trained single-agent PPO ablation.

Trains a centralized single-agent PPO controller with the same hyperparameters
and domain-randomised regime as MAPPO, then evaluates it on the held-out seed
set. This provides a fair CTDE-vs-monolithic (trained-vs-trained) ablation and,
when budgets differ, a sample-efficiency comparison.

Outputs per-seed rows to experiments/results/single_ppo_<SC>_<TAG>.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.environment import DisasterWasteEnv, ScenarioGenerator, ScenarioTier
from src.baselines import SinglePPO, SinglePPOConfig

TIER_MAP = {k: getattr(ScenarioTier, k) for k in
            ["S1_SMALL", "S2_MEDIUM", "S3_LARGE", "S4_SEVERE"]}


def eval_episode(env, sppo, seed):
    obs_dict, _ = env.reset(seed=seed)
    agents = env.possible_agents
    r = 0.0
    done = False
    while not done:
        state_np = obs_dict[agents[0]]["global_state"]
        mask_np = np.array([obs_dict[a]["action_mask"] for a in agents], dtype=np.float32)
        with torch.no_grad():
            acts, _, _ = sppo.actor.get_actions(
                torch.tensor(state_np, dtype=torch.float32, device=sppo._device),
                torch.tensor(mask_np, dtype=torch.float32, device=sppo._device),
                True)
        obs_dict, rew, term, trunc, _ = env.step(
            {a: int(acts.cpu().numpy()[i]) for i, a in enumerate(agents)})
        r += sum(rew.values()) / len(rew)
        done = any(trunc.values()) or any(term.values())
    m = env.get_episode_metrics()
    m["total_reward"] = r
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2_MEDIUM", choices=list(TIER_MAP))
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--tag", default="500k")
    ap.add_argument("--train-seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.train_seed); np.random.seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tier = TIER_MAP[args.scenario]

    scenario = ScenarioGenerator(seed=42).from_tier(tier)
    train_env = DisasterWasteEnv(scenario=scenario, seed=42)
    train_env._randomize_on_reset = True
    train_env._reset_rng = np.random.default_rng(args.train_seed)

    cfg = SinglePPOConfig(total_timesteps=args.timesteps)
    sppo = SinglePPO(n_agents=len(train_env.possible_agents),
                     state_dim=train_env._global_state_dim,
                     action_dim=train_env._action_size,
                     config=cfg, device=device)
    print(f"[SinglePPO] actor params={sppo.actor.num_parameters:,} "
          f"critic params={sppo.critic.num_parameters:,}")
    t0 = time.time()
    sppo.train(train_env, total_timesteps=args.timesteps)
    print(f"[SinglePPO] trained {args.timesteps:,} steps in {time.time()-t0:.0f}s")

    eval_env = DisasterWasteEnv(scenario=ScenarioGenerator(seed=42).from_tier(tier), seed=42)
    rows = []
    for i in range(args.n_seeds):
        s = args.base_seed + i
        m = eval_episode(eval_env, sppo, s)
        rows.append({"algorithm": "SinglePPO", "scenario": args.scenario, "seed": s,
                     "total_cost": m["total_cost"], "total_emission": m["total_emission"],
                     "service_level": m["service_level"], "total_reward": m["total_reward"],
                     "total_time": m.get("total_time", np.nan),
                     "total_delivered": m.get("total_delivered", np.nan),
                     "actor_params": sppo.actor.num_parameters,
                     "train_steps": args.timesteps})
    out = f"experiments/results/single_ppo_{args.scenario}_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")
    print(pd.DataFrame(rows)[["total_cost", "total_emission", "total_reward"]].agg(["mean", "std"]).round(2))


if __name__ == "__main__":
    main()
