"""
eval_zeroshot.py — Zero-shot generalization to unseen topologies.

The policy is trained on the seed-42 topology of a tier. Here we evaluate it,
WITHOUT any retraining, on freshly generated topologies of the same tier
(different generator seeds => different node layouts, edges, demands). Instances
whose observation/action dimensions differ from the training graph are skipped
(the MLP backbone is dimension-specific); we log how many matched. Nearest
Neighbour and GA are run on each instance as re-optimizing references, so the
MAPPO-vs-baseline gap directly quantifies the generalization gap.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, pandas as pd, torch
from src.environment import DisasterWasteEnv, ScenarioGenerator, ScenarioTier
from src.agents import MAPPO
from src.baselines import NearestNeighborBaseline, GeneticAlgorithmBaseline, GAConfig

TIER_MAP = {k: getattr(ScenarioTier, k) for k in ["S1_SMALL","S2_MEDIUM","S3_LARGE","S4_SEVERE"]}


def rollout(env, mappo, seed):
    obs_dict, _ = env.reset(seed=seed)
    agents = env.possible_agents; r = 0.0; done = False
    while not done:
        obs = np.array([obs_dict[a]["obs"] for a in agents], dtype=np.float32)
        msk = np.array([obs_dict[a]["action_mask"] for a in agents], dtype=np.float32)
        with torch.no_grad():
            act, _, _ = mappo.actor.get_action(
                torch.tensor(obs, device=mappo.device),
                torch.tensor(msk, device=mappo.device), deterministic=True)
        obs_dict, rew, term, trunc, _ = env.step({a: int(act[i].item()) for i, a in enumerate(agents)})
        r += sum(rew.values()) / len(rew); done = any(trunc.values()) or any(term.values())
    m = env.get_episode_metrics(); m["total_reward"] = r; return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2_MEDIUM", choices=list(TIER_MAP))
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-instances", type=int, default=20)
    ap.add_argument("--gen-base", type=int, default=5000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tier = TIER_MAP[args.scenario]

    # Reference (training) dims
    ref = DisasterWasteEnv(scenario=ScenarioGenerator(seed=42).from_tier(tier), seed=42)
    ref_dims = (ref._local_obs_dim, ref._global_state_dim, ref._action_size, len(ref.possible_agents))
    print(f"[ref dims] obs={ref_dims[0]} state={ref_dims[1]} act={ref_dims[2]} agents={ref_dims[3]}")

    mappo = MAPPO(n_agents=ref_dims[3], obs_dim=ref_dims[0], state_dim=ref_dims[1],
                  action_dim=ref_dims[2], device=device)
    mappo.load(args.model, load_optimizer=False)
    nn_algo = NearestNeighborBaseline()
    ga_algo = GeneticAlgorithmBaseline(config=GAConfig(population_size=30, n_generations=50, seed=42))

    rows = []; matched = 0; tried = 0
    g = args.gen_base
    while matched < args.n_instances and tried < args.n_instances * 8:
        tried += 1
        try:
            sc = ScenarioGenerator(seed=g).from_tier(tier)
            env = DisasterWasteEnv(scenario=sc, seed=g)
            dims = (env._local_obs_dim, env._global_state_dim, env._action_size, len(env.possible_agents))
        except Exception:
            g += 1; continue
        if dims != ref_dims:
            g += 1; continue
        matched += 1
        m = rollout(env, mappo, g); rows.append({**base(g, "MAPPO_zeroshot"), **kpi(m)})
        m = nn_algo.solve(env, seed=g);      rows.append({**base(g, "NearestNeighbor"), **kpi(m)})
        m = ga_algo.solve(env, seed=g);      rows.append({**base(g, "GeneticAlgorithm"), **kpi(m)})
        print(f"  matched {matched}/{args.n_instances} (gen_seed={g})", flush=True)
        g += 1

    out = f"experiments/results/zeroshot_{args.scenario}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"matched {matched} instances (tried {tried}) -> {out}")
    d = pd.DataFrame(rows)
    print(d.groupby("algorithm")[["total_cost","total_reward","service_level"]].mean().round(3))


def base(seed, algo):
    return {"algorithm": algo, "gen_seed": seed}

def kpi(m):
    return {"total_cost": m["total_cost"], "total_emission": m["total_emission"],
            "service_level": m["service_level"], "total_reward": m["total_reward"]}


if __name__ == "__main__":
    main()
