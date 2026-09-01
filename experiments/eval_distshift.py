"""
eval_distshift.py — Zero-shot distribution-shift generalization.

The S2 policy is trained under lambda_damage=0.05. Here we evaluate it, WITHOUT
retraining, on the same S2 topology but under UNSEEN damage intensities
(lambda in {0.05, 0.10, 0.15, 0.20}). Because the topology/dimensions are fixed,
the trained MLP transfers directly; NN and GA re-optimize at every level and act
as adaptive references. This isolates robustness to a shift in the damage
process from the (architecture-limited) problem of transferring to a different
graph size, which we discuss as a GNN-based future direction.
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, pandas as pd, torch
from src.environment import DisasterWasteEnv, ScenarioGenerator, ScenarioTier
from src.agents import MAPPO
from src.baselines import NearestNeighborBaseline, GeneticAlgorithmBaseline, GAConfig

def rollout(env, mappo, seed):
    obs_dict, _ = env.reset(seed=seed)
    agents = env.possible_agents; r=0.0; done=False
    while not done:
        obs=np.array([obs_dict[a]["obs"] for a in agents],dtype=np.float32)
        msk=np.array([obs_dict[a]["action_mask"] for a in agents],dtype=np.float32)
        with torch.no_grad():
            act,_,_=mappo.actor.get_action(torch.tensor(obs,device=mappo.device),
                                           torch.tensor(msk,device=mappo.device),deterministic=True)
        obs_dict,rew,term,trunc,_=env.step({a:int(act[i].item()) for i,a in enumerate(agents)})
        r+=sum(rew.values())/len(rew); done=any(trunc.values()) or any(term.values())
    m=env.get_episode_metrics(); m["total_reward"]=r; return m

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tier=ScenarioTier.S2_MEDIUM
    model="experiments/models/S2_MEDIUM/mappo_best.pt"
    lambdas=[0.05,0.10,0.15,0.20]; seeds=[2000+i for i in range(15)]
    nn_algo=NearestNeighborBaseline()
    ga_algo=GeneticAlgorithmBaseline(config=GAConfig(population_size=30,n_generations=50,seed=42))
    rows=[]
    for lam in lambdas:
        env=DisasterWasteEnv(scenario=ScenarioGenerator(seed=42).from_tier(tier),seed=42)
        env._network.lambda_damage=lam
        mappo=MAPPO(n_agents=len(env.possible_agents),obs_dim=env._local_obs_dim,
                    state_dim=env._global_state_dim,action_dim=env._action_size,device=device)
        mappo.load(model,load_optimizer=False)
        for s in seeds:
            m=rollout(env,mappo,s); rows.append({"algorithm":"MAPPO_zeroshot","lambda":lam,"seed":s,
                "total_cost":m["total_cost"],"total_emission":m["total_emission"],
                "service_level":m["service_level"],"total_reward":m["total_reward"]})
            env._network.lambda_damage=lam
            m=nn_algo.solve(env,seed=s); env._network.lambda_damage=lam
            rows.append({"algorithm":"NearestNeighbor","lambda":lam,"seed":s,"total_cost":m["total_cost"],
                "total_emission":m["total_emission"],"service_level":m["service_level"],"total_reward":m.get("total_reward",np.nan)})
            m=ga_algo.solve(env,seed=s); env._network.lambda_damage=lam
            rows.append({"algorithm":"GeneticAlgorithm","lambda":lam,"seed":s,"total_cost":m["total_cost"],
                "total_emission":m["total_emission"],"service_level":m["service_level"],"total_reward":m.get("total_reward",np.nan)})
        print(f"lambda={lam} done",flush=True)
    out="experiments/results/distshift_S2.csv"
    pd.DataFrame(rows).to_csv(out,index=False)
    d=pd.DataFrame(rows)
    print(d.groupby(["lambda","algorithm"])[["total_cost","total_reward","service_level"]].mean().round(3).to_string())

if __name__=="__main__":
    main()
