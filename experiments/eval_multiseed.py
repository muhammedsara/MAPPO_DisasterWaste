"""
eval_multiseed.py — Paired multi-seed benchmark evaluation.

For each scenario we build ONE environment on the *training* topology
(ScenarioGenerator seed=42) and then evaluate every algorithm on the SAME set
of N independent stochastic realizations (episodes), obtained by re-seeding the
damage/waste RNGs via env.reset(seed=BASE+ep).  Because all algorithms face an
identical realization for each seed, the samples are *paired*, which lets us run
paired significance tests downstream.  Re-seeding on every episode ensures MAPPO,
the heuristics and MILP all face genuine cross-episode variance.

Outputs one tidy CSV: experiments/results/multiseed_<SCENARIO>.csv
with columns: algorithm, scenario, seed, total_cost, total_emission,
service_level, total_reward, total_time, total_delivered, runtime.
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
from src.agents import MAPPO
from src.baselines import (
    NearestNeighborBaseline,
    ClarkeWrightBaseline,
    GeneticAlgorithmBaseline,
    GAConfig,
    MILPSolver,
)

TIER_MAP = {
    "S1_SMALL": ScenarioTier.S1_SMALL,
    "S2_MEDIUM": ScenarioTier.S2_MEDIUM,
    "S3_LARGE": ScenarioTier.S3_LARGE,
    "S4_SEVERE": ScenarioTier.S4_SEVERE,
}
MILP_MAX_NODES = 30


def rollout_mappo(env, mappo, seed: int) -> dict:
    """One deterministic MAPPO episode on realization `seed`."""
    obs_dict, _ = env.reset(seed=seed)
    agent_list = env.possible_agents
    ep_reward = 0.0
    done = False
    while not done:
        obs_np = np.array([obs_dict[a]["obs"] for a in agent_list], dtype=np.float32)
        masks_np = np.array([obs_dict[a]["action_mask"] for a in agent_list], dtype=np.float32)
        obs_t = torch.tensor(obs_np, device=mappo.device)
        masks_t = torch.tensor(masks_np, device=mappo.device)
        with torch.no_grad():
            actions_t, _, _ = mappo.actor.get_action(obs_t, masks_t, deterministic=True)
        action_dict = {a: int(actions_t[i].item()) for i, a in enumerate(agent_list)}
        obs_dict, rewards, terms, truncs, _ = env.step(action_dict)
        ep_reward += sum(rewards.values()) / len(rewards)
        done = any(truncs.values()) or any(terms.values())
    m = env.get_episode_metrics()
    m["total_reward"] = ep_reward
    return m


def rollout_random_centralized(env, seed: int) -> dict:
    """Untrained centralized (single-agent) controller: uniform random over the
    joint valid action set. Kept only as a sanity reference; the *trained*
    SinglePPO ablation is produced by train_eval_single_ppo.py."""
    obs_dict, _ = env.reset(seed=seed)
    agent_list = env.possible_agents
    rng = np.random.default_rng(seed)
    ep_reward = 0.0
    done = False
    while not done:
        action_dict = {}
        for a in agent_list:
            mask = obs_dict[a]["action_mask"]
            valid = np.flatnonzero(mask)
            action_dict[a] = int(rng.choice(valid)) if valid.size else 0
        obs_dict, rewards, terms, truncs, _ = env.step(action_dict)
        ep_reward += sum(rewards.values()) / len(rewards)
        done = any(truncs.values()) or any(terms.values())
    m = env.get_episode_metrics()
    m["total_reward"] = ep_reward
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(TIER_MAP))
    ap.add_argument("--model", default=None,
                    help="MAPPO checkpoint; default experiments/models/<SC>/mappo_best.pt")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--ga-pop", type=int, default=30)
    ap.add_argument("--ga-gen", type=int, default=50)
    ap.add_argument("--milp-time", type=int, default=60)
    ap.add_argument("--skip-milp", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tier = TIER_MAP[args.scenario]

    # Training topology: seed 42 (the instance every model was trained on).
    scenario = ScenarioGenerator(seed=42).from_tier(tier)
    env = DisasterWasteEnv(scenario=scenario, seed=42)
    n_nodes = scenario.num_nodes
    print(f"[{args.scenario}] {n_nodes} nodes, {len(env.possible_agents)} vehicles, "
          f"obs={env._local_obs_dim} state={env._global_state_dim} act={env._action_size}")

    model_path = args.model or f"experiments/models/{args.scenario}/mappo_best.pt"
    mappo = MAPPO(n_agents=len(env.possible_agents), obs_dim=env._local_obs_dim,
                  state_dim=env._global_state_dim, action_dim=env._action_size,
                  device=device)
    if Path(model_path).exists():
        mappo.load(model_path, load_optimizer=False)
        print(f"[MAPPO] loaded {model_path}")
    else:
        print(f"[MAPPO] !! model not found {model_path} — using UNTRAINED")

    nn_algo = NearestNeighborBaseline()
    cw_algo = ClarkeWrightBaseline()
    ga_algo = GeneticAlgorithmBaseline(
        config=GAConfig(population_size=args.ga_pop, n_generations=args.ga_gen, seed=42))
    run_milp = (not args.skip_milp) and (n_nodes <= MILP_MAX_NODES)
    milp_algo = MILPSolver(time_limit_seconds=args.milp_time) if run_milp else None

    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    rows = []

    def record(algo_name, seed, m, rt):
        rows.append({
            "algorithm": algo_name, "scenario": args.scenario, "seed": seed,
            "total_cost": m.get("total_cost", np.nan),
            "total_emission": m.get("total_emission", np.nan),
            "service_level": m.get("service_level", np.nan),
            "total_reward": m.get("total_reward", np.nan),
            "total_time": m.get("total_time", np.nan),
            "total_delivered": m.get("total_delivered", np.nan),
            "runtime": rt,
        })

    for si, seed in enumerate(seeds):
        # MAPPO
        t0 = time.time(); m = rollout_mappo(env, mappo, seed); record("MAPPO", seed, m, time.time()-t0)
        # Heuristics (each resets env with the same seed internally)
        t0 = time.time(); m = nn_algo.solve(env, seed=seed); record("NearestNeighbor", seed, m, time.time()-t0)
        t0 = time.time(); m = cw_algo.solve(env, seed=seed); record("ClarkeWright", seed, m, time.time()-t0)
        t0 = time.time(); m = ga_algo.solve(env, seed=seed); record("GeneticAlgorithm", seed, m, time.time()-t0)
        if run_milp:
            t0 = time.time(); m = milp_algo.solve(env, seed=seed); record("MILP_ORTools", seed, m, time.time()-t0)
        print(f"  [{si+1}/{len(seeds)}] seed={seed} done", flush=True)

    out = args.out or f"experiments/results/multiseed_{args.scenario}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")
    print(df.groupby("algorithm")[["total_cost", "total_emission", "service_level", "total_reward"]]
          .agg(["mean", "std"]).round(2))


if __name__ == "__main__":
    main()
