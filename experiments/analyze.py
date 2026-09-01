"""Aggregate all experiment CSVs into final numbers + significance tests."""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
R = Path("experiments/results")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

def ci95(x):
    x = np.asarray(x, float); n=len(x)
    if n<2: return (np.nan,np.nan)
    se = x.std(ddof=1)/np.sqrt(n)
    return stats.t.interval(0.95, n-1, loc=x.mean(), scale=se)

print("#"*90); print("# 1. MAIN MULTI-SEED BENCHMARK (30 seeds, paired)"); print("#"*90)
ORDER=["MAPPO","NearestNeighbor","ClarkeWright","GeneticAlgorithm","SinglePPO"]
for sc in ["S1_SMALL","S2_MEDIUM","S3_LARGE","S4_SEVERE"]:
    f=R/f"multiseed_{sc}.csv"
    if not f.exists(): continue
    df=pd.read_csv(f)
    # attach trained SinglePPO for this scenario if present
    spf=R/f"single_ppo_{sc}_500k.csv"
    if spf.exists():
        df=pd.concat([df, pd.read_csv(spf)], ignore_index=True)
    print(f"\n===== {sc}  (n={df.groupby('algorithm').size().to_dict()}) =====")
    g=df.groupby("algorithm").agg(
        cost_m=("total_cost","mean"), cost_s=("total_cost","std"),
        emis_m=("total_emission","mean"), emis_s=("total_emission","std"),
        sl_m=("service_level","mean"), sl_s=("service_level","std"),
        rew_m=("total_reward","mean"), rew_s=("total_reward","std"))
    print(g.round(3).to_string())
    # paired significance: MAPPO vs each baseline on reward & cost
    mp=df[df.algorithm=="MAPPO"].sort_values("seed")
    for alg in ["NearestNeighbor","ClarkeWright","GeneticAlgorithm"]:
        bl=df[df.algorithm==alg].sort_values("seed")
        m=min(len(mp),len(bl))
        if m<3: continue
        a=mp["total_reward"].values[:m]; b=bl["total_reward"].values[:m]
        try:
            w=stats.wilcoxon(a,b);
        except Exception as e:
            w=None
        ac=mp["total_cost"].values[:m]; bc=bl["total_cost"].values[:m]
        try: wc=stats.wilcoxon(ac,bc)
        except Exception: wc=None
        print(f"   MAPPO vs {alg:16s} reward p={getattr(w,'pvalue',float('nan')):.2e} | cost p={getattr(wc,'pvalue',float('nan')):.2e}")

print("\n"+"#"*90); print("# 2. MASKING ABLATION (S2, 30 seeds)"); print("#"*90)
rows=[]
for tag in ["hard","none","soft","hard_corrupt10","hard_corrupt20"]:
    f=R/f"masking_{tag}.csv"
    if not f.exists(): continue
    d=pd.read_csv(f)
    rows.append({"variant":tag,"cost":d.total_cost.mean(),"reward":d.total_reward.mean(),
                 "service%":d.service_level.mean()*100,"invalid/ep":d.invalid_action_count.mean(),
                 "reward_std":d.total_reward.std()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n"+"#"*90); print("# 3. REWARD-WEIGHT SENSITIVITY (S2, 30 seeds)  weights=cost,time,emis,recy"); print("#"*90)
rows=[]
for f in sorted(glob.glob(str(R/"variant_rw_*.csv"))):
    d=pd.read_csv(f); tag=Path(f).stem.replace("variant_rw_","")
    rows.append({"variant":tag,"weights":d.reward_weights.iloc[0],"cost":d.total_cost.mean(),
                 "emission":d.total_emission.mean(),"service%":d.service_level.mean()*100,
                 "time":d.total_time.mean(),"reward":d.total_reward.mean()})
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n"+"#"*90); print("# 4. FLEET-CAPACITY SCALING (S2, 30 seeds)"); print("#"*90)
rows=[]
for f in sorted(glob.glob(str(R/"variant_cap_*.csv"))):
    d=pd.read_csv(f); tag=Path(f).stem.replace("variant_cap_","")
    rows.append({"variant":tag,"cap_scale":d.capacity_scale.iloc[0],"cost":d.total_cost.mean(),
                 "emission":d.total_emission.mean(),"service%":d.service_level.mean()*100,
                 "delivered":d.total_delivered.mean(),"reward":d.total_reward.mean()})
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n"+"#"*90); print("# 5. ZERO-SHOT (S2 model on unseen topologies)"); print("#"*90)
f=R/"zeroshot_S2_MEDIUM.csv"
if f.exists():
    d=pd.read_csv(f)
    print(d.groupby("algorithm").agg(n=("total_cost","size"),cost_m=("total_cost","mean"),
        cost_s=("total_cost","std"),rew_m=("total_reward","mean"),sl_m=("service_level","mean")).round(3).to_string())
else:
    print("  (zeroshot not run yet)")
