"""
eval_masking.py — Evaluate a trained MAPPO masking variant.

Loads a model trained under a given --mask-mode and evaluates it under the SAME
mode across held-out seeds, reporting cost/emission/service/reward AND the number
of infeasible (constraint-violating) actions per episode.

Also supports --corrupt-prob p: with probability p per step, the exposed
feasibility mask is corrupted (a random bit flipped) to emulate noisy/delayed
road-health and waste estimates, so we can measure robustness of the hard-mask
guarantee under imperfect state information.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, pandas as pd, torch
from src.environment import DisasterWasteEnv, ScenarioGenerator, ScenarioTier
from src.agents import MAPPO

TIER_MAP = {k: getattr(ScenarioTier, k) for k in ["S1_SMALL","S2_MEDIUM","S3_LARGE","S4_SEVERE"]}


def rollout(env, mappo, seed, corrupt_prob=0.0, rng=None):
    obs_dict, _ = env.reset(seed=seed)
    agents = env.possible_agents
    r = 0.0; done = False
    while not done:
        obs = np.array([obs_dict[a]["obs"] for a in agents], dtype=np.float32)
        msk = np.array([obs_dict[a]["action_mask"] for a in agents], dtype=np.float32)
        if corrupt_prob > 0.0 and rng is not None:
            for i in range(len(agents)):
                if rng.random() < corrupt_prob:
                    j = int(rng.integers(0, msk.shape[1]))
                    msk[i, j] = 1.0 - msk[i, j]  # flip a mask bit
            msk[:, -1] = 1.0  # keep WAIT valid so softmax is well-defined
        with torch.no_grad():
            act, _, _ = mappo.actor.get_action(
                torch.tensor(obs, device=mappo.device),
                torch.tensor(msk, device=mappo.device), deterministic=True)
        obs_dict, rew, term, trunc, _ = env.step(
            {a: int(act[i].item()) for i, a in enumerate(agents)})
        r += sum(rew.values()) / len(rew)
        done = any(trunc.values()) or any(term.values())
    m = env.get_episode_metrics(); m["total_reward"] = r
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2_MEDIUM", choices=list(TIER_MAP))
    ap.add_argument("--model", required=True)
    ap.add_argument("--mask-mode", default="hard", choices=["hard","soft","none"])
    ap.add_argument("--corrupt-prob", type=float, default=0.0)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tier = TIER_MAP[args.scenario]
    env = DisasterWasteEnv(scenario=ScenarioGenerator(seed=42).from_tier(tier), seed=42)
    env._mask_mode = args.mask_mode
    mappo = MAPPO(n_agents=len(env.possible_agents), obs_dim=env._local_obs_dim,
                  state_dim=env._global_state_dim, action_dim=env._action_size, device=device)
    mappo.load(args.model, load_optimizer=False)

    rng = np.random.default_rng(12345)
    rows = []
    for i in range(args.n_seeds):
        s = args.base_seed + i
        m = rollout(env, mappo, s, args.corrupt_prob, rng)
        rows.append({"variant": args.tag, "scenario": args.scenario, "mask_mode": args.mask_mode,
                     "corrupt_prob": args.corrupt_prob, "seed": s,
                     "total_cost": m["total_cost"], "total_emission": m["total_emission"],
                     "service_level": m["service_level"], "total_reward": m["total_reward"],
                     "invalid_action_count": m.get("invalid_action_count", 0)})
    out = f"experiments/results/masking_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    d = pd.DataFrame(rows)
    print(f"[{args.tag}] mode={args.mask_mode} corrupt={args.corrupt_prob} -> {out}")
    print(d[["total_cost","total_reward","service_level","invalid_action_count"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
