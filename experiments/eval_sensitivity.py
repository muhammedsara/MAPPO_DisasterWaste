"""
eval_sensitivity.py — controlled sensitivity sweeps for the trained S2 policy.

Three parameters that do NOT change the observation/action dimensions are swept by
evaluating the fixed S2-Medium policy (no retraining):
  * road-passability threshold  (env._pass_threshold)
  * episode horizon T           (scenario.config.max_time_steps)
  * demand intensity            (waste_model._demand_scale)

Fleet size is handled separately (run_fleet_sensitivity.sh) because changing the
number of vehicles changes the network dimensions and requires retraining.

Outputs one CSV per sweep in experiments/results/sens_<param>.csv.
"""
from __future__ import annotations
import sys, dataclasses
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, pandas as pd, torch
from src.environment import DisasterWasteEnv, ScenarioGenerator, ScenarioTier
from src.environment.scenario_generator import _TIER_PRESETS
from src.agents import MAPPO

MODEL = "experiments/models/S2_MEDIUM/mappo_best.pt"
TIER = ScenarioTier.S2_MEDIUM
N_SEEDS, BASE = 30, 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(env):
    m = MAPPO(n_agents=len(env.possible_agents), obs_dim=env._local_obs_dim,
              state_dim=env._global_state_dim, action_dim=env._action_size, device=device)
    m.load(MODEL, load_optimizer=False)
    return m


def rollout(env, mappo, seed):
    obs, _ = env.reset(seed=seed); agents = env.possible_agents; r = 0.0; done = False
    while not done:
        o = np.array([obs[a]["obs"] for a in agents], dtype=np.float32)
        msk = np.array([obs[a]["action_mask"] for a in agents], dtype=np.float32)
        with torch.no_grad():
            act, _, _ = mappo.actor.get_action(torch.tensor(o, device=device),
                                               torch.tensor(msk, device=device), deterministic=True)
        obs, rew, term, trunc, _ = env.step({a: int(act[i].item()) for i, a in enumerate(agents)})
        r += sum(rew.values()) / len(rew); done = any(trunc.values()) or any(term.values())
    m = env.get_episode_metrics(); m["total_reward"] = r
    m["total_generated"] = env._waste_model.get_generation_summary()["total_generated"]
    return m


def sweep(name, values, apply_fn, base_tier_config=None):
    rows = []
    for v in values:
        cfg = base_tier_config(v) if base_tier_config else _TIER_PRESETS[TIER]
        scenario = ScenarioGenerator(seed=42).from_config(cfg) if base_tier_config \
            else ScenarioGenerator(seed=42).from_tier(TIER)
        env = DisasterWasteEnv(scenario=scenario, seed=42)
        apply_fn(env, v)
        mappo = load_model(env)
        for i in range(N_SEEDS):
            s = BASE + i
            m = rollout(env, mappo, s)
            rows.append({"param": name, "value": v, "seed": s,
                         "total_cost": m["total_cost"], "total_emission": m["total_emission"],
                         "service_level": m["service_level"], "total_reward": m["total_reward"],
                         "total_generated": m["total_generated"]})
        print(f"  {name}={v} done", flush=True)
    out = f"experiments/results/sens_{name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("value")[["total_cost", "total_reward", "service_level"]].mean().round(3).to_string())
    print(f"-> {out}\n")


if __name__ == "__main__":
    # 1) Passability threshold (dims unchanged) --------------------------------
    def set_pass(env, thr): env._pass_threshold = thr
    sweep("passability", [0.05, 0.10, 0.15, 0.20], set_pass)

    # 2) Horizon T (dims unchanged) --------------------------------------------
    def cfg_T(T): return dataclasses.replace(_TIER_PRESETS[TIER], max_time_steps=T)
    sweep("horizon", [100, 200, 300], lambda env, v: None, base_tier_config=cfg_T)

    # 3) Demand intensity (dims unchanged) -------------------------------------
    def set_demand(env, s): env._waste_model._demand_scale = s
    sweep("demand", [0.5, 1.0, 1.5, 2.0], set_demand)

    print("SENSITIVITY (passability/horizon/demand) DONE")
