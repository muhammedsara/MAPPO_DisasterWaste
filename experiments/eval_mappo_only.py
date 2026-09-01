"""
eval_mappo_only.py — multi-seed evaluation of a single MAPPO model (physical KPIs).
Used for reward-weight sensitivity (E4) and fleet-capacity scaling (E6) variants,
where only the MAPPO policy's behaviour needs to be measured.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2_MEDIUM", choices=list(TIER_MAP))
    ap.add_argument("--model", required=True)
    ap.add_argument("--reward-weights", default="0.25,0.25,0.25,0.25")
    ap.add_argument("--capacity-scale", type=float, default=1.0)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tier = TIER_MAP[args.scenario]
    wc, wt, we, wr = [float(x) for x in args.reward_weights.split(",")]
    rw = {"cost": wc, "time": wt, "emission": we, "recycling": wr}
    sc = ScenarioGenerator(seed=42).from_tier(tier)
    if abs(args.capacity_scale - 1.0) > 1e-9:
        for veh in sc.vehicles:
            veh.config.capacity *= args.capacity_scale
    env = DisasterWasteEnv(scenario=sc, seed=42, reward_weights=rw)
    mappo = MAPPO(n_agents=len(env.possible_agents), obs_dim=env._local_obs_dim,
                  state_dim=env._global_state_dim, action_dim=env._action_size, device=device)
    mappo.load(args.model, load_optimizer=False)
    rows = []
    for i in range(args.n_seeds):
        s = args.base_seed + i
        obs_dict, _ = env.reset(seed=s)
        agents = env.possible_agents; r = 0.0; done = False
        while not done:
            obs = np.array([obs_dict[a]["obs"] for a in agents], dtype=np.float32)
            msk = np.array([obs_dict[a]["action_mask"] for a in agents], dtype=np.float32)
            with torch.no_grad():
                act, _, _ = mappo.actor.get_action(
                    torch.tensor(obs, device=device), torch.tensor(msk, device=device), deterministic=True)
            obs_dict, rew, term, trunc, _ = env.step({a: int(act[i2].item()) for i2, a in enumerate(agents)})
            r += sum(rew.values()) / len(rew); done = any(trunc.values()) or any(term.values())
        m = env.get_episode_metrics()
        rows.append({"tag": args.tag, "scenario": args.scenario, "seed": s,
                     "reward_weights": args.reward_weights, "capacity_scale": args.capacity_scale,
                     "total_cost": m["total_cost"], "total_emission": m["total_emission"],
                     "service_level": m["service_level"], "total_time": m["total_time"],
                     "total_delivered": m["total_delivered"], "total_reward": r})
    out = f"experiments/results/variant_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    d = pd.DataFrame(rows)
    print(f"[{args.tag}] -> {out}")
    print(d[["total_cost","total_emission","service_level","total_reward"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
