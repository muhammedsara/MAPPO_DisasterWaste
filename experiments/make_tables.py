"""Emit LaTeX table rows from the 30-seed experiment data."""
from pathlib import Path
import numpy as np, pandas as pd
R = Path("experiments/results")
ONLINE = ["MAPPO","NearestNeighbor","ClarkeWright","GeneticAlgorithm","SinglePPO"]
LAB = {"MAPPO":"\\textbf{MAPPO (CTDE)}","NearestNeighbor":"NN","ClarkeWright":"CWSA",
       "GeneticAlgorithm":"GA","SinglePPO":"Single-PPO"}

def load(sc):
    df=pd.read_csv(R/f"multiseed_{sc}.csv")
    sp=R/f"single_ppo_{sc}_500k.csv"
    if sp.exists(): df=pd.concat([df,pd.read_csv(sp)],ignore_index=True)
    return df

def f(v,nd=0):
    return f"{v:,.{nd}f}"

print("="*70,"\nTABLE 6 body (cost, CO2, SL%, reward; mean+-std over 30 seeds)\n"+"="*70)
for sc,lab in [("S1_SMALL","S1-Small"),("S2_MEDIUM","S2-Medium"),("S3_LARGE","S3-Large"),("S4_SEVERE","S4-Severe")]:
    df=load(sc); algos=[a for a in ONLINE if a in df.algorithm.unique()]
    # best (lowest cost/emis, highest SL/reward) among online for bolding
    stat={a:{ "cost":df[df.algorithm==a].total_cost.mean(),"emis":df[df.algorithm==a].total_emission.mean(),
              "sl":df[df.algorithm==a].service_level.mean()*100,"rew":df[df.algorithm==a].total_reward.mean(),
              "cstd":df[df.algorithm==a].total_cost.std(),"estd":df[df.algorithm==a].total_emission.std(),
              "sstd":df[df.algorithm==a].service_level.std()*100,"rstd":df[df.algorithm==a].total_reward.std()} for a in algos}
    bcost=min(stat,key=lambda a:stat[a]["cost"]); bemis=min(stat,key=lambda a:stat[a]["emis"])
    bsl=max(stat,key=lambda a:stat[a]["sl"]); brew=max(stat,key=lambda a:stat[a]["rew"])
    print(f"\n%% {lab}")
    n=len(algos)
    for i,a in enumerate(algos):
        s=stat[a]
        def cell(val,std,best,nd=0,pct=False):
            body=f"{val:,.{nd}f}" + (f" $\\pm$ {std:,.{nd}f}" if std==std and std>0 else "")
            if pct: body=f"{val:.2f} $\\pm$ {std:.2f}" if (std==std and std>0) else f"{val:.2f}"
            return f"\\textbf{{{body}}}" if a==best else body
        first = f"\\multirow{{{n}}}{{*}}{{{lab}}} " if i==0 else ""
        print(f"    {first}& {LAB[a]:26s} & {cell(s['cost'],s['cstd'],bcost)} & {cell(s['emis'],s['estd'],bemis)} & "
              f"{cell(s['sl'],s['sstd'],bsl,pct=True)} & {cell(s['rew'],s['rstd'],brew,nd=1)} \\\\")
    print("    \\midrule")

print("\n"+"="*70,"\nTABLE 7 body (service level % across scenarios)\n"+"="*70)
scl={}
for sc in ["S1_SMALL","S2_MEDIUM","S3_LARGE","S4_SEVERE"]:
    df=load(sc)
    scl[sc]={a:(df[df.algorithm==a].service_level.mean()*100, df[df.algorithm==a].service_level.std()*100) for a in ONLINE if a in df.algorithm.unique()}
for a in ONLINE:
    row=LAB[a]
    for sc in ["S1_SMALL","S2_MEDIUM","S3_LARGE","S4_SEVERE"]:
        if a in scl[sc]:
            m,s=scl[sc][a]; best=max(scl[sc],key=lambda k:scl[sc][k][0])
            cell=f"{m:.2f} $\\pm$ {s:.2f}" if s>0 else f"{m:.2f}"
            row+=f" & "+(f"\\textbf{{{cell}}}" if a==best else cell)
        else: row+=" & ---"
    print("    "+row+" \\\\")

print("\n"+"="*70,"\nTABLE 8 (ablation MAPPO vs trained Single-PPO, S2)\n"+"="*70)
df=load("S2_MEDIUM")
for a in ["MAPPO","SinglePPO"]:
    d=df[df.algorithm==a]
    print(f"  {a}: cost={d.total_cost.mean():,.0f}+-{d.total_cost.std():,.0f}  emis={d.total_emission.mean():,.0f}  "
          f"SL={d.service_level.mean()*100:.2f}%  reward={d.total_reward.mean():.1f}+-{d.total_reward.std():.1f}")
sp=pd.read_csv(R/"single_ppo_S2_MEDIUM_500k.csv")
print(f"  Single-PPO actor params: {sp.actor_params.iloc[0]:,}")

print("\n"+"="*70,"\nMASKING TABLE\n"+"="*70)
for tag in ["hard","none","soft","hard_corrupt10","hard_corrupt20"]:
    fp=R/f"masking_{tag}.csv"
    if fp.exists():
        d=pd.read_csv(fp)
        print(f"  {tag:16s}: cost={d.total_cost.mean():,.0f}  reward={d.total_reward.mean():.1f}  "
              f"SL={d.service_level.mean()*100:.2f}%  invalid/ep={d.invalid_action_count.mean():.1f}")

print("\n"+"="*70,"\nSENSITIVITY TABLE\n"+"="*70)
for f2 in sorted((R).glob("variant_rw_*.csv")):
    d=pd.read_csv(f2)
    print(f"  {f2.stem.replace('variant_rw_',''):10s} w={d.reward_weights.iloc[0]:22s} cost={d.total_cost.mean():,.0f}  "
          f"emis={d.total_emission.mean():,.0f}  SL={d.service_level.mean()*100:.2f}%  time={d.total_time.mean():.0f}  rew={d.total_reward.mean():.1f}")

print("\n"+"="*70,"\nCAPACITY TABLE\n"+"="*70)
d0=load("S2_MEDIUM"); d0=d0[d0.algorithm=='MAPPO']
print(f"  x1 : cost={d0.total_cost.mean():,.0f}  SL={d0.service_level.mean()*100:.2f}%  delivered={d0.total_delivered.mean():.0f}")
for f2 in sorted((R).glob("variant_cap_*.csv")):
    d=pd.read_csv(f2)
    print(f"  x{int(d.capacity_scale.iloc[0]):<2d}: cost={d.total_cost.mean():,.0f}  SL={d.service_level.mean()*100:.2f}%  delivered={d.total_delivered.mean():.0f}")

print("\n"+"="*70,"\nDISTSHIFT ZERO-SHOT\n"+"="*70)
ds=pd.read_csv(R/"distshift_S2.csv")
print(ds.groupby(["lambda","algorithm"])[["total_cost","total_reward","service_level"]].mean().round(2).to_string())

print("\n"+"="*70,"\nHEADLINE IMPROVEMENTS (S2, vs best baseline)\n"+"="*70)
d=load("S2_MEDIUM"); mp=d[d.algorithm=="MAPPO"]; ga=d[d.algorithm=="GeneticAlgorithm"]; nn=d[d.algorithm=="NearestNeighbor"]
mc,gc,nc=mp.total_cost.mean(),ga.total_cost.mean(),nn.total_cost.mean()
me,ge,ne=mp.total_emission.mean(),ga.total_emission.mean(),nn.total_emission.mean()
print(f"  cost: vs GA {100*(gc-mc)/gc:.1f}%  vs NN {100*(nc-mc)/nc:.1f}%")
print(f"  emis: vs GA {100*(ge-me)/ge:.1f}%  vs NN {100*(ne-me)/ne:.1f}%")
print(f"  SL: MAPPO {mp.service_level.mean()*100:.2f}% vs NN {nn.service_level.mean()*100:.2f}% (x{mp.service_level.mean()/nn.service_level.mean():.2f})")
