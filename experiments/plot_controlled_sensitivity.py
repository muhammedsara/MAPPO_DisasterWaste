"""
plot_controlled_sensitivity.py — 4-panel controlled sensitivity figure (Rev2).
Fleet size, horizon T, demand intensity, road-passability threshold.
Matches the restrained academic style of plot_revision.py; 95% CI error bars.
Writes figures/fig_controlled_sensitivity.{pdf,png}.
"""
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
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "axes.grid": True, "axes.axisbelow": True,
    "grid.alpha": 0.35, "grid.linewidth": 0.5, "grid.linestyle": (0, (2, 3)),
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
})

C_COST = "#1f4e79"   # deep blue -> cost
C_SERV = "#6e2b2b"   # muted dark red -> service level


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2 or x.std() == 0: return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(n)


def agg(df, xcol, ycol, scale=1.0):
    g = df.groupby(xcol)[ycol]
    xs = np.array(sorted(df[xcol].unique()), float)
    means = np.array([g.get_group(x).mean() * scale for x in xs])
    cis = np.array([ci95(g.get_group(x)) * scale for x in xs])
    return xs, means, cis


def twin_panel(ax, xs, cost_m, cost_ci, serv_m, serv_ci, xlabel, title, xticks=None):
    ax.errorbar(xs, cost_m, yerr=cost_ci, color=C_COST, marker="o", ms=5,
                lw=1.6, capsize=3, label="Total cost")
    ax.set_xlabel(xlabel); ax.set_ylabel("Total cost", color=C_COST)
    ax.tick_params(axis="y", labelcolor=C_COST)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.errorbar(xs, serv_m, yerr=serv_ci, color=C_SERV, marker="s", ms=4.5,
                 lw=1.6, ls="--", capsize=3, label="Service level")
    ax2.set_ylabel("Service level (%)", color=C_SERV)
    ax2.tick_params(axis="y", labelcolor=C_SERV)
    ax2.grid(False)
    return ax2


fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) Fleet size -----------------------------------------------------------
rows = []
for k, f in [(5, "variant_fleet_K5.csv"), (15, "variant_fleet_K15.csv"),
             (20, "variant_fleet_K20.csv")]:
    d = pd.read_csv(R / f); d["K"] = k; rows.append(d)
d10 = pd.read_csv(R / "multiseed_S2_MEDIUM.csv")
d10 = d10[d10["algorithm"] == "MAPPO"].copy(); d10["K"] = 10; rows.append(d10)
fleet = pd.concat(rows, ignore_index=True)
xs, cm, cc = agg(fleet, "K", "total_cost")
_, sm, sc = agg(fleet, "K", "service_level", scale=100)
twin_panel(axes[0, 0], xs, cm, cc, sm, sc, "Fleet size $K$ (vehicles)",
           "(a) Fleet size", xticks=[5, 10, 15, 20])

# (b) Horizon T ------------------------------------------------------------
hor = pd.read_csv(R / "sens_horizon.csv")
xs, cm, cc = agg(hor, "value", "total_cost")
_, sm, sc = agg(hor, "value", "service_level", scale=100)
twin_panel(axes[0, 1], xs, cm, cc, sm, sc, "Horizon $T$ (steps)",
           "(b) Horizon", xticks=[100, 200, 300])

# (c) Demand intensity -----------------------------------------------------
dem = pd.read_csv(R / "sens_demand.csv")
xs, cm, cc = agg(dem, "value", "total_cost")
_, sm, sc = agg(dem, "value", "service_level", scale=100)
twin_panel(axes[1, 0], xs, cm, cc, sm, sc, "Demand scale",
           "(c) Demand intensity", xticks=[0.5, 1.0, 1.5, 2.0])

# (d) Passability threshold ------------------------------------------------
pas = pd.read_csv(R / "sens_passability.csv")
xs, cm, cc = agg(pas, "value", "total_cost")
_, sm, sc = agg(pas, "value", "service_level", scale=100)
twin_panel(axes[1, 1], xs, cm, cc, sm, sc,
           r"Passability threshold $\varphi_{\mathrm{pass}}$",
           "(d) Passability threshold", xticks=[0.05, 0.1, 0.15, 0.2, 0.5])

# shared legend
h1 = plt.Line2D([], [], color=C_COST, marker="o", lw=1.6, label="Total cost")
h2 = plt.Line2D([], [], color=C_SERV, marker="s", ls="--", lw=1.6,
                label="Service level")
fig.legend(handles=[h1, h2], loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 1.02), frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.98])
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_controlled_sensitivity.{ext}")
print("wrote", OUT / "fig_controlled_sensitivity.pdf")
