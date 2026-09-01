"""Professional MAPPO-CTDE architecture diagram (box-and-arrow, muted palette)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("figures")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "dejavuserif"})

C_BLUE = "#1f4e79"; C_GREY = "#595959"; C_MASK = "#6e2b2b"
P_TRAIN = "#e7edf3"; P_POL = "#ededed"; P_EXEC = "#f3efe7"; ENV = "#dde3e8"

fig, ax = plt.subplots(figsize=(13, 6.6))
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")

def panel(x0, y0, x1, y1, color, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.02,rounding_size=0.15",
                 facecolor=color, edgecolor="#b8bfc6", linewidth=1.0, zorder=1))
    ax.text((x0 + x1) / 2, y1 - 0.32, title, ha="center", va="center",
            fontsize=14, fontweight="bold", color="#222222", zorder=5)

def box(cx, cy, w, h, text, edge, fc="white", lw=1.0, fs=11.5, tc="black"):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=fc, edgecolor=edge, linewidth=lw, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)
    return (cx, cy, w, h)

def arrow(p_from, p_to, color="#333333", ls="-", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=14,
                 color=color, lw=lw, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=6, shrinkA=2, shrinkB=2))

# ----- panels -----
panel(0.3, 2.7, 5.0, 9.5, P_TRAIN, "Centralised Training")
panel(5.5, 2.7, 10.5, 9.5, P_POL, "Shared Policy (CTDE)")
panel(11.0, 2.7, 15.7, 9.5, P_EXEC, "Decentralised Execution")

BW, BH = 4.1, 0.9
# Training column
xt = 2.65
b_state  = box(xt, 8.5, BW, BH, r"Global state $\mathbf{s}(t)$", C_GREY)
b_critic = box(xt, 7.2, BW, BH, r"Centralised critic $V_\phi(\mathbf{s})$", C_GREY)
b_gae    = box(xt, 5.9, BW, BH, r"GAE  ($\gamma{=}0.99,\ \lambda{=}0.95$)", C_GREY)
b_adv    = box(xt, 4.6, BW, BH, r"Advantages $\hat{A}_k$", C_GREY)
for a, b in [(b_state, b_critic), (b_critic, b_gae), (b_gae, b_adv)]:
    arrow((a[0], a[1]-BH/2), (b[0], b[1]+BH/2))

# Policy column
xp = 8.0
b_obs   = box(xp, 8.5, BW, BH, r"Local obs. $\mathbf{o}_k$", C_BLUE, lw=1.8)
b_actor = box(xp, 7.2, BW, BH, r"Shared actor $\pi_\theta$ (MLP)", C_BLUE, lw=1.8)
b_logit = box(xp, 5.9, BW, BH, r"Raw logits $\mathbf{z}_k$", C_GREY)
b_mask  = box(xp, 4.6, BW, BH, r"Hard action mask ($-10^{9}$)", C_MASK, lw=1.8, fc="#f6eeee")
b_pol   = box(xp, 3.3, BW, BH, r"Policy $\pi_\theta(a_k\,|\,\mathbf{o}_k)$", C_BLUE, lw=1.8)
for a, b in [(b_obs, b_actor), (b_actor, b_logit), (b_logit, b_mask), (b_mask, b_pol)]:
    arrow((a[0], a[1]-BH/2), (b[0], b[1]+BH/2))

# Execution column
xe = 13.35
b_a1 = box(xe, 8.4, BW, 0.8, r"Agent $1$: $\pi_\theta$ + mask", C_GREY)
b_a2 = box(xe, 7.4, BW, 0.8, r"Agent $2$: $\pi_\theta$ + mask", C_GREY)
ax.text(xe, 6.75, r"$\vdots$", ha="center", va="center", fontsize=15, zorder=4)
b_aK = box(xe, 6.0, BW, 0.8, r"Agent $K$: $\pi_\theta$ + mask", C_GREY)
ax.text(xe, 5.0, "no inter-agent\ncommunication", ha="center", va="center",
        fontsize=11, style="italic", color=C_GREY, zorder=4)

# ----- cross-panel flows -----
arrow((b_adv[0]+BW/2, b_adv[1]), (b_actor[0]-BW/2, b_actor[1]), color=C_BLUE, ls=(0,(4,3)), lw=1.6, rad=-0.15)
ax.text(5.25, 6.35, r"$\nabla_\theta$", ha="center", fontsize=12, color=C_BLUE, zorder=7)
arrow((b_pol[0]+BW/2, b_pol[1]), (11.0, b_a1[1]-0.4), color="#333333", lw=1.6, rad=0.12)

# ----- environment bar -----
ax.add_patch(FancyBboxPatch((0.3, 0.5), 15.4, 1.4,
             boxstyle="round,pad=0.02,rounding_size=0.12",
             facecolor=ENV, edgecolor="#9aa4ad", linewidth=1.2, zorder=2))
ax.text(8.0, 1.2, r"Dynamic disaster-network environment  $\mathcal{G}=(\mathcal{V},\mathcal{E})$:"
                  "\n"
        r"compound-Poisson road damage $+$ concurrent repair  $\cdot$  Log-Normal waste generation",
        ha="center", va="center", fontsize=12, zorder=5)

# env -> panels (observations/state up); execution -> env (actions down)
arrow((2.65, 1.9), (2.65, 4.15), color="#555555", lw=1.2)                   # env -> training (global state)
ax.text(3.15, 3.0, "state", fontsize=10, color="#555555", rotation=90, va="center")
arrow((8.0, 1.9), (8.0, 2.85), color="#555555", lw=1.2)                     # env -> policy (obs)
ax.text(8.5, 2.35, r"obs. $\mathbf{o}_k$", fontsize=10, color="#555555", va="center")
arrow((13.35, 5.6), (13.35, 1.9), color="#333333", lw=1.4)                  # actions -> env
ax.text(13.85, 3.6, r"actions $a_k$", fontsize=10, color="#333333", rotation=90, va="center")

fig.savefig(OUT / "fig_system_architecture.pdf", bbox_inches="tight", dpi=400)
fig.savefig(OUT / "fig_system_architecture.png", bbox_inches="tight", dpi=200)
print("architecture diagram done ->", OUT)
