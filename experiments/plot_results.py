"""
plot_results.py — regenerate all paper figures from the 30-seed experiment data.

Serious/academic style: restrained palette (the proposed method MAPPO in a single
deep professional blue; all baselines in graded neutral greys), serif typography
to match the LaTeX body, thin marks, light y-only grid, 95% CI error bars.
Writes into figures/.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("experiments/results")
OUT = Path("figures"); OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5,
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "axes.grid": True, "axes.axisbelow": True,
    "grid.alpha": 0.35, "grid.linewidth": 0.5, "grid.linestyle": (0, (2, 3)),
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
})

# Restrained academic palette: MAPPO = deep professional blue; baselines = greys.
C_MAPPO = "#1f4e79"    # deep blue (proposed method)
C_ACCENT = "#6e2b2b"   # muted dark red (secondary series where needed)
COL = {
    "MAPPO":           C_MAPPO,
    "GeneticAlgorithm":"#595959",  # dark grey
    "NearestNeighbor": "#8c8c8c",  # medium grey
    "ClarkeWright":    "#bdbdbd",  # light grey
    "SinglePPO":       "#3d3d3d",  # near-black grey
}
LAB = {"MAPPO":"MAPPO","NearestNeighbor":"NN","ClarkeWright":"CWSA",
       "GeneticAlgorithm":"GA","SinglePPO":"Single-PPO"}
ORDER = ["MAPPO","NearestNeighbor","ClarkeWright","GeneticAlgorithm","SinglePPO"]
SCEN = {"S1_SMALL":"S1-Small","S2_MEDIUM":"S2-Medium","S3_LARGE":"S3-Large","S4_SEVERE":"S4-Severe"}


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2 or x.std() == 0: return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(n)

def thousands(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

def load_scenario(sc):
    df = pd.read_csv(R / f"multiseed_{sc}.csv")
    sp = R / f"single_ppo_{sc}_500k.csv"
    if sp.exists(): df = pd.concat([df, pd.read_csv(sp)], ignore_index=True)
    return df

def bar_panel(ax, df, metric, title, ylabel):
    algos = [a for a in ORDER if a in df.algorithm.unique()]
    means = [df[df.algorithm == a][metric].mean() for a in algos]
    errs  = [ci95(df[df.algorithm == a][metric].values) for a in algos]
    x = np.arange(len(algos))
    bars = ax.bar(x, means, width=0.66, yerr=errs, capsize=3,
                  color=[COL[a] for a in algos], edgecolor="#222222", linewidth=0.7,
                  error_kw=dict(ecolor="#222222", lw=0.9))
    for i, a in enumerate(algos):          # emphasise the proposed method
        if a == "MAPPO":
            bars[i].set_edgecolor("black"); bars[i].set_linewidth(1.3)
    ax.set_xticks(x); ax.set_xticklabels([LAB[a] for a in algos])
    ax.set_title(title, pad=6); ax.set_ylabel(ylabel)
    ax.margins(y=0.16); thousands(ax)

# ---- Fig 2 (cost) & Fig 3 (emission): per-scenario panels ----
for metric, fnpref, label in [("total_cost", "fig1_cost", "Total cost (\\$)"),
                              ("total_emission", "fig2_emission", "CO$_2$ emission (kg)")]:
    for sc in SCEN:
        df = load_scenario(sc)
        fig, ax = plt.subplots(figsize=(5.0, 3.7))
        bar_panel(ax, df, metric, SCEN[sc], label)
        fig.savefig(OUT / f"{fnpref}_{sc}.pdf"); fig.savefig(OUT / f"{fnpref}_{sc}.png"); plt.close(fig)
print("cost & emission panels done")

# ---- Fig 4: ablation MAPPO vs trained Single-PPO (cost AND emission) ----
df2 = load_scenario("S2_MEDIUM")
algos = ["MAPPO", "SinglePPO"]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
for ax, metric, ylab in zip(axes, ["total_cost", "total_emission"], ["Total cost (\\$)", "CO$_2$ emission (kg)"]):
    m = [df2[df2.algorithm == a][metric].mean() for a in algos]
    e = [ci95(df2[df2.algorithm == a][metric].values) for a in algos]
    bars = ax.bar([0, 1], m, width=0.55, yerr=e, capsize=4,
                  color=[COL["MAPPO"], COL["SinglePPO"]], edgecolor="#222222", linewidth=0.8)
    bars[0].set_edgecolor("black"); bars[0].set_linewidth(1.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["MAPPO\n(CTDE)", "Single-PPO\n(monolithic)"])
    ax.set_ylabel(ylab); ax.margins(y=0.18); thousands(ax)
    for xi, v in zip([0, 1], m): ax.text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_ablation_cost.pdf"); fig.savefig(OUT / "fig_ablation_cost.png"); plt.close(fig)
print("ablation fig done")

# ---- Fig 5: masking ablation (reward bars + invalid-action line) ----
mrows = []
for tag, lab in [("hard", "Hard\n(ours)"), ("none", "No\nmask"), ("soft", "Soft\npenalty"),
                 ("hard_corrupt10", "+10%\nnoise"), ("hard_corrupt20", "+20%\nnoise")]:
    f = R / f"masking_{tag}.csv"
    if f.exists():
        d = pd.read_csv(f)
        mrows.append((lab, d.total_reward.mean(), ci95(d.total_reward.values), d.invalid_action_count.mean()))
labs = [r[0] for r in mrows]; rew = [r[1] for r in mrows]; rerr = [r[2] for r in mrows]; inv = [r[3] for r in mrows]
x = np.arange(len(labs))
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.0))
# (a) invalid actions / episode -- the decisive metric (symlog so 0 and 2000 both readable)
cols = [C_MAPPO if i == 0 else "#595959" for i in range(len(labs))]
bA = axL.bar(x, inv, 0.62, color=cols, edgecolor="#222222", linewidth=0.7)
bA[0].set_edgecolor("black"); bA[0].set_linewidth(1.3)
axL.set_yscale("symlog", linthresh=1)
axL.set_ylabel("Invalid actions / episode"); axL.set_title("(a) Constraint violations", pad=6)
axL.set_xticks(x); axL.set_xticklabels(labs); axL.set_ylim(0, 3000)
for xi, v in zip(x, inv): axL.text(xi, max(v, 0.5), f"{v:.0f}", ha="center", va="bottom", fontsize=10)
# (b) mean reward -- points on a zoomed axis (avoids misleading from-zero bars)
order = np.argsort(rew)
axR.errorbar([rew[i] for i in order], range(len(order)), xerr=[rerr[i] for i in order],
             fmt="o", color=C_MAPPO, ecolor="#888888", ms=9, mec="black", mew=0.6, capsize=3, ls="none")
axR.set_yticks(range(len(order))); axR.set_yticklabels([labs[i].replace("\n", " ") for i in order])
axR.set_xlabel("Mean reward"); axR.set_title("(b) Mean reward", pad=6); axR.grid(axis="x")
fig.tight_layout()
fig.savefig(OUT / "fig_masking_ablation.pdf"); fig.savefig(OUT / "fig_masking_ablation.png"); plt.close(fig)
print("masking fig done")

# ---- Fig 6: reward-weight sensitivity (cost-emission frontier + service bars) ----
rows = []
for f in glob.glob(str(R / "variant_rw_*.csv")):
    d = pd.read_csv(f); tag = Path(f).stem.replace("variant_rw_", "")
    rows.append((tag, d.total_cost.mean(), d.total_emission.mean(), d.service_level.mean() * 100))
d0 = load_scenario("S2_MEDIUM"); d0 = d0[d0.algorithm == "MAPPO"]
rows.append(("balanced", d0.total_cost.mean(), d0.total_emission.mean(), d0.service_level.mean() * 100))
NICE = {"costHeavy": "cost-heavy", "timeHeavy": "time-heavy", "emisHeavy": "emis-heavy",
        "recyHeavy": "recy-heavy", "pe20": r"$\omega_e{=}0.2$", "pe40": r"$\omega_e{=}0.4$",
        "pe60": r"$\omega_e{=}0.6$", "balanced": "balanced"}
sd = pd.DataFrame(rows, columns=["tag", "cost", "emis", "sl"])
def hbar(ax, order_col, val_col, xlabel, title, fmt):
    s = sd.sort_values(order_col)
    y = list(range(len(s)))
    colors = [C_MAPPO if t == "balanced" else "#595959" for t in s.tag]
    ax.barh(y, s[val_col], color=colors, edgecolor="#222222", linewidth=0.7, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([NICE.get(t, t) for t in s.tag])
    ax.set_xlabel(xlabel); ax.set_title(title, pad=6); ax.grid(axis="x")
    ax.locator_params(axis="x", nbins=5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 3.9))
hbar(axa, "cost", "cost", "Total cost (\\$)", "(a) Total cost by weighting", lambda v, _: f"{v:,.0f}")
hbar(axb, "sl", "sl", "Service level (%)", "(b) Service level by weighting", lambda v, _: f"{v:.2f}")
fig.tight_layout()
fig.savefig(OUT / "fig_sensitivity.pdf"); fig.savefig(OUT / "fig_sensitivity.png"); plt.close(fig)
print("sensitivity fig done")

# ---- Fig 7: fleet-capacity scaling ----
rows = [("1", d0.total_cost.mean(), d0.service_level.mean() * 100)]
for f in sorted(glob.glob(str(R / "variant_cap_*.csv"))):
    d = pd.read_csv(f); rows.append((str(int(d.capacity_scale.iloc[0])), d.total_cost.mean(), d.service_level.mean() * 100))
cd = pd.DataFrame(rows, columns=["scale", "cost", "sl"]); cd["s"] = cd.scale.astype(int)
cd = cd.sort_values("s")
fig, ax = plt.subplots(figsize=(6.0, 3.9))
ax.plot(cd.s, cd.sl, "o-", color=C_MAPPO, lw=1.9, ms=8, mec="black", mew=0.5)
ax.set_ylabel("Service level (%)"); ax.set_xlabel("Fleet-capacity scale factor ($\\times$)")
ax.set_xticks(cd.s)
for _, r in cd.iterrows(): ax.annotate(f"{r.sl:.2f}%", (r.s, r.sl), xytext=(0, 7),
                                       textcoords="offset points", ha="center", fontsize=10, color="#333333")
ax.margins(y=0.18)
fig.tight_layout()
fig.savefig(OUT / "fig_capacity.pdf"); fig.savefig(OUT / "fig_capacity.png"); plt.close(fig)
print("capacity fig done")

# ---- Fig 8: training convergence (held-out reward vs steps) ----
import json
GREY = {"S1_SMALL": "#8c8c8c", "S3_LARGE": "#595959", "S4_SEVERE": "#3d3d3d"}
LABS = {"S1_SMALL": "S1-Small", "S2_MEDIUM": "S2-Medium", "S3_LARGE": "S3-Large", "S4_SEVERE": "S4-Severe"}
STY = {"S1_SMALL": "--", "S2_MEDIUM": "-", "S3_LARGE": "-.", "S4_SEVERE": ":"}
fig, ax = plt.subplots(figsize=(7.4, 4.2))
for sc in ["S1_SMALL", "S2_MEDIUM", "S3_LARGE", "S4_SEVERE"]:
    h = json.load(open(f"experiments/models/{sc}/train_log.json"))["history"]
    steps = np.array(h["steps"]); er = np.array([np.nan if v is None else v for v in h["eval_reward"]], float)
    m = ~np.isnan(er)
    col = C_MAPPO if sc == "S2_MEDIUM" else GREY[sc]
    lw = 2.2 if sc == "S2_MEDIUM" else 1.5
    ax.plot(steps[m] / 1000, er[m], STY[sc], color=col, lw=lw, label=LABS[sc])
ax.set_xlabel("Environment steps (thousands)"); ax.set_ylabel("Held-out evaluation reward")
ax.legend(loc="lower right", frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "fig_training_convergence.pdf"); fig.savefig(OUT / "fig_training_convergence.png"); plt.close(fig)
print("training convergence fig done")
print("ALL FIGURES DONE ->", OUT)
