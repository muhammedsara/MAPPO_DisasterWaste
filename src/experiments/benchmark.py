"""
benchmark.py — Comparative Benchmark & Publication-Quality Visualization
==========================================================================

Usage::

    cd /home/kurtar/KURTAR/WorkOut/MAPPO-DisasterWaste
    python src/experiments/benchmark.py --scenario S2_MEDIUM \\
           --model results/models/mappo_final.pt

This script:
    1. Loads the trained MAPPO model and all baseline algorithms.
    2. Evaluates each on the SINGLE specified scenario (N episodes).
    3. Saves results to results/benchmark_results_{scenario}.csv.
    4. Generates 3 publication-quality figures (300 DPI):
       a) Total Cost comparison (bar chart — x-axis: algorithms)
       b) Carbon Emission comparison (bar chart — x-axis: algorithms)
       c) Service Level comparison (bar chart — x-axis: algorithms)

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.environment import DisasterWasteEnv, ScenarioGenerator, ScenarioTier
from src.agents import MAPPO, MAPPOConfig
from src.baselines import (
    NearestNeighborBaseline,
    ClarkeWrightBaseline,
    GeneticAlgorithmBaseline,
    GAConfig,
    MILPSolver,
    SinglePPO,
    SinglePPOConfig,
)


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

TIER_MAP = {
    "S1_SMALL": ScenarioTier.S1_SMALL,
    "S2_MEDIUM": ScenarioTier.S2_MEDIUM,
    "S3_LARGE": ScenarioTier.S3_LARGE,
    "S4_SEVERE": ScenarioTier.S4_SEVERE,
}

# MILP is feasible only on small scenarios; larger ones will be skipped
MILP_MAX_NODES = 30  # Skip MILP when total nodes exceed this

ALGORITHM_NAMES = [
    "MAPPO",
    "SinglePPO",
    "MILP",
    "ClarkeWright",
    "NearestNeighbor",
    "GeneticAlgorithm",
]

# Publication-quality style
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Color palette for algorithms
ALGO_COLORS = {
    "MAPPO": "#2196F3",          # Blue
    "SinglePPO": "#9C27B0",      # Purple
    "MILP": "#4CAF50",           # Green
    "ClarkeWright": "#FF9800",   # Orange
    "NearestNeighbor": "#F44336",# Red
    "GeneticAlgorithm": "#795548",# Brown
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark comparisons on a SINGLE scenario"
    )
    parser.add_argument(
        "--scenario", type=str, default="S2_MEDIUM",
        choices=list(TIER_MAP.keys()),
        help="Scenario tier to benchmark (default: S2_MEDIUM)",
    )
    parser.add_argument(
        "--model", type=str, default="results/models/mappo_final.pt",
        help="Path to trained MAPPO model",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--milp-time", type=int, default=60,
        help="MILP time limit (seconds)",
    )
    parser.add_argument(
        "--ga-pop", type=int, default=30,
        help="GA population size",
    )
    parser.add_argument(
        "--ga-gen", type=int, default=50,
        help="GA generations",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-milp", action="store_true",
        help="Skip MILP solver entirely",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_mappo(
    env: DisasterWasteEnv,
    model_path: str,
    n_episodes: int,
    device,
) -> List[Dict]:
    """Evaluate trained MAPPO model."""
    n_agents = len(env.possible_agents)
    obs_dim = env._local_obs_dim
    state_dim = env._global_state_dim
    action_dim = env._action_size

    mappo = MAPPO(
        n_agents=n_agents, obs_dim=obs_dim,
        state_dim=state_dim, action_dim=action_dim,
        device=device,
    )

    if Path(model_path).exists():
        mappo.load(model_path, load_optimizer=False)
        print(f"    Loaded MAPPO from {model_path}")
    else:
        print(f"    [WARN] Model not found: {model_path}. Using untrained MAPPO.")

    results = []
    for ep in range(n_episodes):
        m = mappo.evaluate(env, n_episodes=1, deterministic=True)
        m["algorithm"] = "MAPPO"
        m["episode"] = ep
        results.append(m)
    return results


def evaluate_single_ppo(
    env: DisasterWasteEnv,
    n_episodes: int,
    device,
) -> List[Dict]:
    """Evaluate untrained SinglePPO (ablation)."""
    n_agents = len(env.possible_agents)
    state_dim = env._global_state_dim
    action_dim = env._action_size

    sppo = SinglePPO(
        n_agents=n_agents, state_dim=state_dim,
        action_dim=action_dim, device=device,
    )

    results = []
    for ep in range(n_episodes):
        m = sppo.evaluate(env, n_episodes=1, deterministic=False)
        m["episode"] = ep
        results.append(m)
    return results


def evaluate_heuristic(
    env: DisasterWasteEnv,
    algo,
    n_episodes: int,
    seed: int,
) -> List[Dict]:
    """Evaluate a heuristic baseline."""
    results = []
    for ep in range(n_episodes):
        m = algo.solve(env, seed=seed + ep)
        m["episode"] = ep
        results.append(m)
    return results


# ---------------------------------------------------------------------------
# Main benchmark — SINGLE SCENARIO
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    """Run all algorithms on the specified single scenario."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sc_name = args.scenario
    sc_tier = TIER_MAP[sc_name]

    gen = ScenarioGenerator(seed=args.seed)
    scenario = gen.from_tier(sc_tier)
    env = DisasterWasteEnv(scenario=scenario, seed=args.seed)

    n_nodes = scenario.num_nodes

    # Initialise baselines
    nn_algo = NearestNeighborBaseline()
    cw_algo = ClarkeWrightBaseline()
    ga_algo = GeneticAlgorithmBaseline(
        config=GAConfig(
            population_size=args.ga_pop,
            n_generations=args.ga_gen,
            seed=args.seed,
        )
    )
    milp_algo = MILPSolver(time_limit_seconds=args.milp_time)

    all_results: List[Dict] = []

    print(f"\n{'='*60}")
    print(f"  Scenario: {sc_name}  ({n_nodes} nodes, "
          f"{len(env.possible_agents)} vehicles)")
    print(f"{'='*60}")

    # --- MAPPO ---
    print(f"  → MAPPO...")
    t0 = time.time()
    mappo_results = evaluate_mappo(
        env, args.model, args.episodes, device
    )
    for r in mappo_results:
        r["scenario"] = sc_name
        r["runtime"] = (time.time() - t0) / max(len(mappo_results), 1)
    all_results.extend(mappo_results)

    # --- SinglePPO (ablation) ---
    print(f"  → SinglePPO...")
    t0 = time.time()
    sppo_results = evaluate_single_ppo(env, args.episodes, device)
    for r in sppo_results:
        r["scenario"] = sc_name
        r["runtime"] = (time.time() - t0) / max(len(sppo_results), 1)
    all_results.extend(sppo_results)

    # --- Clarke-Wright ---
    print(f"  → Clarke-Wright...")
    t0 = time.time()
    cw_results = evaluate_heuristic(env, cw_algo, args.episodes, args.seed)
    for r in cw_results:
        r["scenario"] = sc_name
        r["runtime"] = (time.time() - t0) / max(len(cw_results), 1)
    all_results.extend(cw_results)

    # --- Nearest Neighbor ---
    print(f"  → Nearest Neighbor...")
    t0 = time.time()
    nn_results = evaluate_heuristic(env, nn_algo, args.episodes, args.seed)
    for r in nn_results:
        r["scenario"] = sc_name
        r["runtime"] = (time.time() - t0) / max(len(nn_results), 1)
    all_results.extend(nn_results)

    # --- Genetic Algorithm ---
    print(f"  → Genetic Algorithm...")
    t0 = time.time()
    ga_results = evaluate_heuristic(env, ga_algo, args.episodes, args.seed)
    for r in ga_results:
        r["scenario"] = sc_name
        r["runtime"] = (time.time() - t0) / max(len(ga_results), 1)
    all_results.extend(ga_results)

    # --- MILP (with size guard) ---
    if args.skip_milp:
        print(f"  → MILP: Skipped (--skip-milp flag)")
    elif n_nodes > MILP_MAX_NODES:
        print(f"  → MILP: Skipped (too large: {n_nodes} nodes > "
              f"{MILP_MAX_NODES} max)")
    else:
        print(f"  → MILP (time_limit={args.milp_time}s, "
              f"nodes={n_nodes})...")
        t0 = time.time()
        milp_results = evaluate_heuristic(
            env, milp_algo, min(args.episodes, 3), args.seed
        )
        for r in milp_results:
            r["scenario"] = sc_name
            r["runtime"] = (time.time() - t0) / max(len(milp_results), 1)
        all_results.extend(milp_results)

    # Build DataFrame
    rows = []
    for r in all_results:
        rows.append({
            "algorithm": r.get("algorithm", "Unknown"),
            "scenario": r.get("scenario", sc_name),
            "episode": r.get("episode", 0),
            "total_cost": r.get("total_cost", r.get("mean_cost", np.nan)),
            "total_emission": r.get("total_emission",
                                    r.get("mean_emission", np.nan)),
            "service_level": r.get("service_level",
                                   r.get("mean_service_level", np.nan)),
            "total_reward": r.get("total_reward",
                                  r.get("mean_reward", np.nan)),
            "runtime": r.get("runtime", np.nan),
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Visualization — single-scenario bar plots
# ---------------------------------------------------------------------------

def generate_figures(
    df: pd.DataFrame,
    output_dir: Path,
    scenario_name: str,
) -> None:
    """Generate 3 publication-quality bar-chart figures for one scenario."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PAPER_STYLE)

    # Aggregate: mean ± std per algorithm (single scenario)
    agg = df.groupby("algorithm").agg(
        cost_mean=("total_cost", "mean"),
        cost_std=("total_cost", "std"),
        emission_mean=("total_emission", "mean"),
        emission_std=("total_emission", "std"),
        service_mean=("service_level", "mean"),
        service_std=("service_level", "std"),
        reward_mean=("total_reward", "mean"),
        runtime_mean=("runtime", "mean"),
    ).reset_index()

    # Order algorithms consistently
    avail_algos = [a for a in ALGORITHM_NAMES if a in agg["algorithm"].values]
    agg = agg.set_index("algorithm").reindex(avail_algos).reset_index()
    colors = [ALGO_COLORS[a] for a in avail_algos]

    x = np.arange(len(avail_algos))
    bar_width = 0.55

    # ================================================================
    # Figure 1: Total Cost (Bar Chart — x-axis: algorithms)
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(
        x, agg["cost_mean"].values, bar_width,
        color=colors,
        yerr=agg["cost_std"].fillna(0).values,
        capsize=4, edgecolor="white", linewidth=0.8,
    )
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Total Cost (×10³)")
    ax1.set_title(f"Total Cost Comparison — {scenario_name}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(avail_algos, rotation=25, ha="right")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    fig1.tight_layout()

    fname1 = f"fig1_cost_{scenario_name}"
    fig1.savefig(fig_dir / f"{fname1}.pdf")
    fig1.savefig(fig_dir / f"{fname1}.png", dpi=300)
    plt.close(fig1)
    print(f"  ✓ Figure 1: {fig_dir / f'{fname1}.pdf'}")

    # ================================================================
    # Figure 2: Carbon Emissions (Bar Chart — x-axis: algorithms)
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(
        x, agg["emission_mean"].values, bar_width,
        color=colors,
        yerr=agg["emission_std"].fillna(0).values,
        capsize=4, edgecolor="white", linewidth=0.8,
    )
    ax2.set_xlabel("Algorithm")
    ax2.set_ylabel("CO₂ Emissions (kg)")
    ax2.set_title(f"Carbon Emission Comparison — {scenario_name}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(avail_algos, rotation=25, ha="right")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    fig2.tight_layout()

    fname2 = f"fig2_emission_{scenario_name}"
    fig2.savefig(fig_dir / f"{fname2}.pdf")
    fig2.savefig(fig_dir / f"{fname2}.png", dpi=300)
    plt.close(fig2)
    print(f"  ✓ Figure 2: {fig_dir / f'{fname2}.pdf'}")

    # ================================================================
    # Figure 3: Service Level (Bar Chart — x-axis: algorithms)
    # ================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(
        x, agg["service_mean"].values, bar_width,
        color=colors,
        yerr=agg["service_std"].fillna(0).values,
        capsize=4, edgecolor="white", linewidth=0.8,
    )
    ax3.set_xlabel("Algorithm")
    ax3.set_ylabel("Service Level (Waste Collection Rate)")
    ax3.set_title(f"Service Level Comparison — {scenario_name}")
    ax3.set_xticks(x)
    ax3.set_xticklabels(avail_algos, rotation=25, ha="right")
    ax3.set_ylim(bottom=0, top=1.05)
    fig3.tight_layout()

    fname3 = f"fig3_service_{scenario_name}"
    fig3.savefig(fig_dir / f"{fname3}.pdf")
    fig3.savefig(fig_dir / f"{fname3}.png", dpi=300)
    plt.close(fig3)
    print(f"  ✓ Figure 3: {fig_dir / f'{fname3}.pdf'}")

    # === Summary table ===
    print(f"\n{'='*80}")
    print(f"  BENCHMARK SUMMARY — {scenario_name}")
    print(f"{'='*80}")
    summary_cols = [
        "algorithm", "cost_mean", "cost_std",
        "emission_mean", "emission_std",
        "service_mean", "service_std",
        "runtime_mean",
    ]
    print(agg[summary_cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sc_name = args.scenario

    print("=" * 60)
    print("  Benchmark: MAPPO vs Baselines (Single Scenario)")
    print("=" * 60)
    print(f"  Model    : {args.model}")
    print(f"  Scenario : {sc_name}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Output   : {output_dir}")

    # Run benchmark on the single specified scenario
    df = run_benchmark(args)

    # Save CSV — scenario name in filename
    csv_path = output_dir / f"benchmark_results_{sc_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Results saved: {csv_path}")
    print(f"    Shape: {df.shape}")
    print(f"\n  Preview:\n{df.head(10).to_string()}")

    # Generate figures — scenario name in filenames
    print("\n  Generating publication-quality figures...")
    generate_figures(df, output_dir, sc_name)

    print(f"\n{'='*60}")
    print(f"  Benchmark Complete for {sc_name}! ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
