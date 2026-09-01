"""
compute_ci_sens.py — compute 95% CIs for headline KPIs and summarise sensitivity
sweeps (passability/horizon/demand/fleet) for the ASOC Rev2 manuscript.

Reads the existing 30-seed multiseed CSVs and the sensitivity CSVs; prints
ready-to-paste LaTeX-friendly mean +/- half-width (95% CI) figures.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parent / "results"


def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    sd = x.std(ddof=1)
    hw = 1.96 * sd / np.sqrt(n)        # normal approx (n=30 large enough)
    return m, hw, n


def fmt(m, hw, dec=1):
    return f"{m:.{dec}f} $\\pm$ {hw:.{dec}f}"


print("=" * 68)
print("HEADLINE 95% CIs (MAPPO, per tier) -- n=30 seeds")
print("=" * 68)
for tier in ["S1_SMALL", "S2_MEDIUM", "S3_LARGE", "S4_SEVERE"]:
    f = R / f"multiseed_{tier}.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    d = d[d["algorithm"] == "MAPPO"]
    print(f"\n[{tier}]  n={len(d)}")
    for col, dec in [("total_cost", 1), ("total_emission", 1),
                     ("service_level", 4), ("total_reward", 1)]:
        m, hw, n = ci95(d[col])
        if col == "service_level":
            m, hw = m * 100, hw * 100    # to %
            print(f"  {col:16s}: {m:.2f} +/- {hw:.2f} %")
        else:
            print(f"  {col:16s}: {m:.1f} +/- {hw:.1f}")

print("\n" + "=" * 68)
print("SENSITIVITY SUMMARIES (mean +/- 95% CI over 30 seeds)")
print("=" * 68)

for name, param_label, unit in [("passability", "Passability threshold $\\phi_{pass}$", ""),
                                 ("horizon", "Horizon $T$", ""),
                                 ("demand", "Demand scale", "")]:
    f = R / f"sens_{name}.csv"
    if not f.exists():
        print(f"  (missing {f.name})")
        continue
    d = pd.read_csv(f)
    print(f"\n--- {name} ---")
    for v, g in d.groupby("value"):
        mc, hc, _ = ci95(g["total_cost"])
        me, he, _ = ci95(g["total_emission"])
        ms, hs, _ = ci95(g["service_level"] * 100)
        print(f"  {v:>6}: cost {mc:7.1f}+/-{hc:4.1f} | "
              f"emis {me:7.1f}+/-{he:4.1f} | serv {ms:5.2f}+/-{hs:4.2f}%")

# Fleet: base K=10 comes from multiseed_S2, others from variant_fleet_K*.
print("\n--- fleet (n_vehicles) ---")
fleet_rows = []
d10 = pd.read_csv(R / "multiseed_S2_MEDIUM.csv")
d10 = d10[d10["algorithm"] == "MAPPO"]
mc, hc, _ = ci95(d10["total_cost"]); ms, hs, _ = ci95(d10["service_level"] * 100)
me, he, _ = ci95(d10["total_emission"])
fleet_rows.append((10, mc, hc, me, he, ms, hs))
for k in [5, 15, 20]:
    f = R / f"variant_fleet_K{k}.csv"
    if not f.exists():
        print(f"  K={k}: (pending)")
        continue
    d = pd.read_csv(f)
    mc, hc, _ = ci95(d["total_cost"]); ms, hs, _ = ci95(d["service_level"] * 100)
    me, he, _ = ci95(d["total_emission"])
    fleet_rows.append((k, mc, hc, me, he, ms, hs))
for k, mc, hc, me, he, ms, hs in sorted(fleet_rows):
    print(f"  K={k:>2}: cost {mc:7.1f}+/-{hc:4.1f} | "
          f"emis {me:7.1f}+/-{he:4.1f} | serv {ms:5.2f}+/-{hs:4.2f}%")

print("\nDONE")
