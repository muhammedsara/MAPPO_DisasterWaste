"""
dashboard.py — Professional Streamlit Dashboard for MAPPO Disaster Waste Management
=====================================================================================

A publication-grade interactive dashboard with three views:

    1. **Scenario Viewer**  — Interactive PyDeck network map with colour-coded
       nodes and health-gradient edges.
    2. **Model Test**       — Run MAPPO / baselines with progress tracking;
       KPI cards shown only after successful completion.
    3. **Benchmark Metrics** — Bar charts and tables from benchmark CSV.

Usage::

    cd /home/kurtar/KURTAR/WorkOut/MAPPO-DisasterWaste
    streamlit run src/ui/dashboard.py

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

from src.environment import DisasterWasteEnv, ScenarioGenerator, ScenarioTier
from src.environment.network import NodeType


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & STYLE
# ═══════════════════════════════════════════════════════════════════════════

# --- Page configuration ---
st.set_page_config(
    page_title="MAPPO — Disaster Waste Management",
    page_icon=":recycle:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Inject FontAwesome CDN + custom CSS ---
st.markdown(
    """
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
          crossorigin="anonymous" referrerpolicy="no-referrer" />
    <style>
    /* ── Global ──────────────────────────────────────────── */
    .block-container { padding-top: 1.2rem; }

    /* ── Metric cards ────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 18px;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.5rem;
    }

    /* ── Tab labels ──────────────────────────────────────── */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
    }

    /* ── Sidebar styling ─────────────────────────────────── */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] h2 {
        color: #e2e8f0;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    /* ── Info / Warning boxes ────────────────────────────── */
    .stAlert { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Theme palette ---
PALETTE = {
    "primary":   "#3b82f6",  # Blue-500
    "success":   "#22c55e",  # Green-500
    "warning":   "#f59e0b",  # Amber-500
    "danger":    "#ef4444",  # Red-500
    "secondary": "#8b5cf6",  # Violet-500
    "neutral":   "#94a3b8",  # Slate-400
}

# --- Node type definitions with FontAwesome icon classes ---
NODE_STYLE = {
    NodeType.DEPOT: {
        "label": "Depo",
        "color": [59, 130, 246, 220],      # Blue
        "hex":   "#2196F3",
        "radius": 600,
        "fa":    "fa-solid fa-warehouse",
    },
    NodeType.WASTE_GENERATION: {
        "label": "Yikım Alanı",
        "color": [239, 68, 68, 220],       # Red
        "hex":   "#F44336",
        "radius": 400,
        "fa":    "fa-solid fa-person-digging",
    },
    NodeType.TCP: {
        "label": "Gecici Toplama",
        "color": [245, 158, 11, 220],      # Amber
        "hex":   "#FF9800",
        "radius": 450,
        "fa":    "fa-solid fa-box-archive",
    },
    NodeType.SORTING_FACILITY: {
        "label": "Ayrıstırma Tesisi",
        "color": [34, 197, 94, 220],       # Green
        "hex":   "#4CAF50",
        "radius": 500,
        "fa":    "fa-solid fa-recycle",
    },
    NodeType.LANDFILL: {
        "label": "Duzenli Depolama",
        "color": [156, 39, 176, 220],      # Purple
        "hex":   "#9C27B0",
        "radius": 500,
        "fa":    "fa-solid fa-dumpster",
    },
}

TIER_MAP = {
    "S1 — Small (Kucuk)":     ScenarioTier.S1_SMALL,
    "S2 — Medium (Orta)":     ScenarioTier.S2_MEDIUM,
    "S3 — Large (Buyuk)":     ScenarioTier.S3_LARGE,
    "S4 — Severe (Siddetli)": ScenarioTier.S4_SEVERE,
}

TIER_KEYS = {
    ScenarioTier.S1_SMALL:  "S1_SMALL",
    ScenarioTier.S2_MEDIUM: "S2_MEDIUM",
    ScenarioTier.S3_LARGE:  "S3_LARGE",
    ScenarioTier.S4_SEVERE: "S4_SEVERE",
}

ALGO_COLORS_HEX = {
    "MAPPO":            "#3b82f6",
    "SinglePPO":        "#8b5cf6",
    "MILP_ORTools":     "#22c55e",
    "ClarkeWright":     "#f59e0b",
    "NearestNeighbor":  "#ef4444",
    "GeneticAlgorithm": "#78350f",
}


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple:
    """Render the sidebar with scenario and model controls.

    Returns
    -------
    tuple
        (scenario_display_name, seed, model_path_or_None)
    """
    with st.sidebar:
        st.markdown(
            "<h1 style='text-align:center; font-size:1.3rem; "
            "color:#e2e8f0; margin-bottom:0.2rem;'>"
            "MAPPO Dashboard</h1>"
            "<p style='text-align:center; color:#64748b; font-size:0.80rem; "
            "margin-top:0;'>"
            "Disaster Waste Logistics</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Scenario ──
        st.markdown("## Senaryo")
        scenario_name = st.selectbox(
            "Afet Senaryosu",
            list(TIER_MAP.keys()),
            index=1,
            label_visibility="collapsed",
        )
        seed = st.number_input(
            "Seed", min_value=0, max_value=9999, value=42,
        )
        st.divider()

        # ── Model ──
        st.markdown("## Model")
        model_dir = PROJECT_ROOT / "results" / "models"
        model_files = sorted(model_dir.glob("*.pt")) if model_dir.exists() else []

        if model_files:
            model_path = st.selectbox(
                "Model dosyası",
                model_files,
                format_func=lambda p: p.name,
                label_visibility="collapsed",
            )
        else:
            model_path = None
            st.warning(
                "Model dosyası bulunamadı.\n\n"
                "`python src/experiments/train.py`\n"
                "komutu ile egitim gerceklestirin.",
            )

        st.divider()
        st.caption("Muhammed Şara  ·  2026")

    return scenario_name, int(seed), model_path


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — SCENARIO VIEWER   (PyDeck interactive map)
# ═══════════════════════════════════════════════════════════════════════════

def _health_to_rgb(h: float) -> List[int]:
    """Map road-health coefficient [0, 1] to [red ... yellow ... green].

    Parameters
    ----------
    h : float
        Health coefficient in [0, 1].

    Returns
    -------
    List[int]
        [R, G, B, A] suitable for PyDeck.
    """
    h = max(0.0, min(1.0, h))
    if h < 0.5:
        r = 239
        g = int(68 + (245 - 68) * (h / 0.5))
        b = int(68 + (11 - 68) * (h / 0.5))
    else:
        t = (h - 0.5) / 0.5
        r = int(245 - (245 - 34) * t)
        g = int(158 + (197 - 158) * t)
        b = int(11 + (94 - 11) * t)
    return [r, g, b, 200]


def tab_scenario_viewer(scenario_name: str, seed: int) -> None:
    """Render interactive PyDeck network map and scenario statistics.

    Nodes are colour-coded by type; edges are coloured from
    green (healthy) to red (damaged) based on ``phi_ij``.
    """
    st.markdown("### Senaryo Goruntuleyici")

    tier = TIER_MAP[scenario_name]
    gen = ScenarioGenerator(seed=seed)
    scenario = gen.from_tier(tier)
    env = DisasterWasteEnv(scenario=scenario, seed=seed)

    # ── Time slider: advance environment to reveal Poisson damage ──
    sim_t = st.slider(
        "Simülasyon Zamanı (Adım İleri Sar)",
        min_value=0, max_value=100, value=0, step=1,
        help="Ortamı T adım ilerletir. Araçlar WAIT eylemini kullanır, "
             "yalnızca Poisson hasar süreci ve atık üretimi ilerler.",
    )

    if sim_t > 0:
        obs, _ = env.reset(seed=seed)
        wait_actions = {
            agent: env.action_space(agent).n - 1
            for agent in env.possible_agents
        }
        progress = st.progress(0, text=f"Ortam ilerletiliyor (0/{sim_t})...")
        for t in range(sim_t):
            obs, _, terms, truncs, _ = env.step(wait_actions)
            if t % max(1, sim_t // 20) == 0 or t == sim_t - 1:
                progress.progress(
                    (t + 1) / sim_t,
                    text=f"Ortam ilerletiliyor ({t + 1}/{sim_t})...",
                )
            if all(terms.values()) or all(truncs.values()):
                st.info(f"Episode t={t+1}'de sona erdi.")
                break
        progress.empty()

    network = env._network
    G = network.graph

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_vehicles = len(env.possible_agents)
    n_waste = len(network.get_nodes_by_type(NodeType.WASTE_GENERATION))

    # ── KPI cards ──
    cols = st.columns(4)
    cols[0].metric("Toplam Dugum", n_nodes)
    cols[1].metric("Kenar Sayısı", n_edges)
    cols[2].metric("Arac Sayısı", n_vehicles)
    cols[3].metric("Yikım Alanı", n_waste)

    # ── Build PyDeck data ──
    node_records: List[Dict[str, Any]] = []
    for nid in G.nodes():
        pos = network.get_node_position(nid)
        nt = network.get_node_type(nid)
        style = NODE_STYLE.get(nt, NODE_STYLE[NodeType.WASTE_GENERATION])
        node_records.append({
            "lon": float(pos[0]),
            "lat": float(pos[1]),
            "color": style["color"],
            "radius": style["radius"],
            "label": f"{style['label']} #{nid}",
            "type": style["label"],
        })

    edge_records: List[Dict[str, Any]] = []
    for u, v in G.edges():
        h = network.get_edge_health(u, v)
        pu = network.get_node_position(u)
        pv = network.get_node_position(v)
        edge_records.append({
            "src_lon": float(pu[0]),
            "src_lat": float(pu[1]),
            "dst_lon": float(pv[0]),
            "dst_lat": float(pv[1]),
            "color": _health_to_rgb(h),
            "health": round(float(h), 3),
        })

    node_df = pd.DataFrame(node_records)
    edge_df = pd.DataFrame(edge_records)

    # ── PyDeck layers ──

    # 1) Edges — health-gradient coloured lines
    edge_layer = pdk.Layer(
        "LineLayer",
        data=edge_df,
        get_source_position=["src_lon", "src_lat"],
        get_target_position=["dst_lon", "dst_lat"],
        get_color="color",
        get_width=3,
        width_min_pixels=1,
        pickable=True,
    )

    # 2) Nodes — bright, stroked circles
    node_layer = pdk.Layer(
        "ScatterplotLayer",
        data=node_df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=6,
        radius_max_pixels=22,
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
        stroked=True,
        pickable=True,
        auto_highlight=True,
    )

    # 3) Labels — node type text next to each dot
    text_layer = pdk.Layer(
        "TextLayer",
        data=node_df,
        get_position=["lon", "lat"],
        get_text="type",
        get_color=[220, 220, 220, 230],
        get_size=13,
        get_alignment_baseline="'top'",
        get_pixel_offset=[0, 14],
        font_family="'Inter', 'Segoe UI', sans-serif",
        pickable=False,
    )

    # Compute map centre
    mid_lon = float(node_df["lon"].mean())
    mid_lat = float(node_df["lat"].mean())
    span = max(
        node_df["lon"].max() - node_df["lon"].min(),
        node_df["lat"].max() - node_df["lat"].min(),
    )
    # Approximate zoom from span
    zoom = max(1, 11 - np.log2(max(span, 1)))

    view = pdk.ViewState(
        longitude=mid_lon,
        latitude=mid_lat,
        zoom=zoom,
        pitch=30,
    )

    tooltip = {
        "html": "<b>{label}</b><br/>Tip: {type}",
        "style": {
            "backgroundColor": "#1e293b",
            "color": "#f1f5f9",
            "fontSize": "12px",
            "borderRadius": "6px",
            "padding": "8px 12px",
        },
    }

    # CartoDB dark_matter — free, no API key required
    deck = pdk.Deck(
        layers=[edge_layer, node_layer, text_layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_provider="carto",
        map_style="dark",
    )
    st.pydeck_chart(deck, use_container_width=True)

    # ── Glassmorphism Legend with FontAwesome icons ──
    legend_html = (
        "<div style='"
        "background: rgba(20, 24, 36, 0.85); "
        "backdrop-filter: blur(12px); "
        "-webkit-backdrop-filter: blur(12px); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 12px; "
        "box-shadow: 0 4px 6px rgba(0,0,0,0.3), 0 0 40px rgba(59,130,246,0.06); "
        "padding: 16px 20px; "
        "margin-top: 12px; "
        "max-width: 720px;'"
        ">"
        "<div style='color:#94a3b8; font-size:0.72rem; text-transform:uppercase; "
        "letter-spacing:0.1em; margin-bottom:10px; font-weight:700;'>Lejant</div>"
    )

    # Node-type items with FontAwesome
    legend_html += "<div style='display:flex; flex-wrap:wrap; gap:14px; margin-bottom:12px;'>"
    for nt, style in NODE_STYLE.items():
        fa_cls = style.get("fa", "fa-solid fa-circle")
        hex_c = style.get("hex", "#888")
        legend_html += (
            f"<span style='display:inline-flex; align-items:center; gap:7px;'>"
            f"<i class='{fa_cls}' style='color:{hex_c}; font-size:0.9rem;'></i>"
            f"<span style='color:#e2e8f0; font-size:0.82rem;'>{style['label']}</span>"
            f"</span>"
        )
    legend_html += "</div>"

    # Edge health items with line icons
    legend_html += (
        "<div style='border-top:1px solid rgba(255,255,255,0.06); "
        "padding-top:10px; display:flex; flex-wrap:wrap; gap:14px;'>"
        "<span style='color:#94a3b8; font-size:0.72rem; text-transform:uppercase; "
        "letter-spacing:0.1em; font-weight:700; width:100%; margin-bottom:4px;'>"
        "Yol Sagligi</span>"
    )
    for label, h_val, fa_color in [
        ("Saglik >70%", 0.85, "#4CAF50"),
        ("Saglik 30-70%", 0.50, "#FF9800"),
        ("Saglik <30%", 0.15, "#F44336"),
    ]:
        legend_html += (
            f"<span style='display:inline-flex; align-items:center; gap:7px;'>"
            f"<i class='fa-solid fa-minus' style='color:{fa_color}; font-size:0.9rem;'></i>"
            f"<span style='color:#e2e8f0; font-size:0.82rem;'>{label}</span>"
            f"</span>"
        )
    legend_html += "</div></div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    # ── Node distribution table ──
    st.markdown("#### Dugum Dagilimi")
    dist_data = {}
    for nt in NodeType:
        nodes = network.get_nodes_by_type(nt)
        if nodes:
            dist_data[NODE_STYLE.get(nt, {}).get("label", str(nt))] = len(nodes)
    st.bar_chart(dist_data)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL TEST & SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def _run_algorithm(
    env: DisasterWasteEnv,
    algo_name: str,
    model_path: Optional[Path],
    n_episodes: int,
    seed: int,
    progress_bar,
) -> Optional[Dict[str, Any]]:
    """Execute the selected algorithm, updating a Streamlit progress bar.

    Parameters
    ----------
    env : DisasterWasteEnv
        Environment instance to evaluate on.
    algo_name : str
        Display name of the chosen algorithm.
    model_path : Path or None
        Path to trained .pt model (MAPPO only).
    n_episodes : int
        Number of evaluation episodes.
    seed : int
        Random seed.
    progress_bar : st.progress
        Streamlit progress bar widget to update.

    Returns
    -------
    dict or None
        Aggregated metrics dictionary, or None on failure.
    """
    import torch

    try:
        if "MAPPO" in algo_name:
            if model_path is None or not Path(model_path).exists():
                st.error(
                    "MAPPO model dosyası bulunamadı. "
                    "Lütfen `train.py` ile egitim yapın."
                )
                return None

            from src.agents import MAPPO
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            progress_bar.progress(10, text="Model yukleniyor...")
            mappo = MAPPO(
                n_agents=len(env.possible_agents),
                obs_dim=env._local_obs_dim,
                state_dim=env._global_state_dim,
                action_dim=env._action_size,
                device=device,
            )
            mappo.load(str(model_path), load_optimizer=False)
            progress_bar.progress(30, text="Simulasyon calısıyor...")
            result = mappo.evaluate(env, n_episodes=n_episodes, deterministic=True)
            progress_bar.progress(100, text="Tamamlandı.")
            return result

        elif "Nearest" in algo_name:
            from src.baselines import NearestNeighborBaseline
            progress_bar.progress(20, text="Nearest Neighbor calısıyor...")
            nn = NearestNeighborBaseline()
            result = nn.solve_batch(env, n_episodes=n_episodes, seed=seed)
            progress_bar.progress(100, text="Tamamlandı.")
            return result

        elif "Clarke" in algo_name:
            from src.baselines import ClarkeWrightBaseline
            progress_bar.progress(20, text="Clarke-Wright calısıyor...")
            cw = ClarkeWrightBaseline()
            result = cw.solve_batch(env, n_episodes=n_episodes, seed=seed)
            progress_bar.progress(100, text="Tamamlandı.")
            return result

        elif "Genetic" in algo_name:
            from src.baselines import GeneticAlgorithmBaseline, GAConfig
            progress_bar.progress(20, text="Genetik Algoritma calısıyor...")
            ga = GeneticAlgorithmBaseline(
                config=GAConfig(population_size=20, n_generations=30, seed=seed)
            )
            result = ga.solve_batch(env, n_episodes=n_episodes, seed=seed)
            progress_bar.progress(100, text="Tamamlandı.")
            return result

        elif "MILP" in algo_name:
            from src.baselines import MILPSolver
            progress_bar.progress(10, text="MILP Solver calısıyor (max 30s)...")
            milp = MILPSolver(time_limit_seconds=30)
            result = milp.solve_batch(
                env, n_episodes=min(n_episodes, 2), seed=seed,
            )
            progress_bar.progress(100, text="Tamamlandı.")
            return result

    except Exception as exc:
        st.error(f"Simulasyon hatası: {exc}")
        return None


def tab_model_test(
    scenario_name: str, seed: int, model_path: Optional[Path],
) -> None:
    """Model Test / Simulation tab.

    Metric cards are **hidden** until a simulation has been successfully
    run.  Before that, an informational box prompts the user.
    """
    st.markdown("### Model Test ve Simulasyon")

    tier = TIER_MAP[scenario_name]
    gen = ScenarioGenerator(seed=seed)
    scenario = gen.from_tier(tier)
    env = DisasterWasteEnv(scenario=scenario, seed=seed)

    # ── Controls ──
    ctrl_left, ctrl_right = st.columns([3, 1])
    with ctrl_left:
        algo_choice = st.selectbox(
            "Algoritma",
            [
                "MAPPO (Egitilmis)",
                "Nearest Neighbor",
                "Clarke-Wright",
                "Genetic Algorithm",
                "MILP (OR-Tools)",
            ],
            label_visibility="collapsed",
        )
    with ctrl_right:
        n_episodes = st.number_input(
            "Episode", min_value=1, max_value=10, value=3,
        )

    run_clicked = st.button(
        "Simulasyonu Baslat",
        type="primary",
        use_container_width=True,
    )

    # ── State guard: show info until simulation runs ──
    if "sim_metrics" not in st.session_state:
        st.session_state["sim_metrics"] = None

    if run_clicked:
        progress = st.progress(0, text="Hazırlanıyor...")
        metrics = _run_algorithm(
            env, algo_choice, model_path, n_episodes, seed, progress,
        )
        time.sleep(0.3)  # brief pause so user sees 100%
        progress.empty()

        if metrics is not None:
            st.session_state["sim_metrics"] = metrics
        else:
            st.session_state["sim_metrics"] = None

    # ── Display results (or placeholder) ──
    metrics = st.session_state.get("sim_metrics")

    if metrics is None:
        st.info(
            "Lutfen yukarıdan bir algoritma secin ve "
            "**Simulasyonu Baslat** butonuna tıklayın. "
            "Performans metrikleri burada gorunecektir."
        )
        return

    # ── KPI cards ──
    st.markdown("#### Performans Metrikleri")
    cost = metrics.get("total_cost", metrics.get("mean_cost", 0))
    emission = metrics.get("total_emission", metrics.get("mean_emission", 0))
    service = metrics.get("service_level", metrics.get("mean_service_level", 0))
    reward = metrics.get("total_reward", metrics.get("mean_reward", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Maliyet", f"{cost:,.1f}")
    c2.metric("CO2 Emisyonu (kg)", f"{emission:,.1f}")

    if service is not None and not (isinstance(service, float) and np.isnan(service)):
        c3.metric("Hizmet Seviyesi", f"{service:.4f}")
    else:
        c3.metric("Hizmet Seviyesi", "N/A")

    c4.metric("Toplam Odul", f"{reward:,.2f}")

    # ── Detailed breakdown ──
    with st.expander("Detaylı Metrikler"):
        detail_rows = []
        for k, v in sorted(metrics.items()):
            if isinstance(v, (int, float)):
                detail_rows.append({"Metrik": k, "Deger": f"{v:.4f}"})
            else:
                detail_rows.append({"Metrik": k, "Deger": str(v)})
        st.dataframe(
            pd.DataFrame(detail_rows),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — BENCHMARK METRICS
# ═══════════════════════════════════════════════════════════════════════════

def tab_training_metrics(scenario_name: str) -> None:
    """Display benchmark results from CSV with Streamlit-native bar charts.

    Automatically detects scenario-specific CSV files generated by
    ``benchmark.py --scenario <TIER>``.
    """
    st.markdown("### Egitim ve Kıyaslama Metrikleri")

    tier = TIER_MAP[scenario_name]
    tier_key = TIER_KEYS[tier]

    # Try scenario-specific CSV first, then generic
    csv_candidates = [
        PROJECT_ROOT / "results" / f"benchmark_results_{tier_key}.csv",
        PROJECT_ROOT / "results" / "benchmark_results.csv",
    ]
    csv_path = None
    for c in csv_candidates:
        if c.exists():
            csv_path = c
            break

    if csv_path is None:
        st.warning(
            "Benchmark sonuclari bulunamadı.\n\n"
            "Asagıdaki komutu calistirin:\n\n"
            f"```\npython src/experiments/benchmark.py --scenario {tier_key}\n```"
        )
        return

    df = pd.read_csv(csv_path)
    st.caption(f"Kaynak: `{csv_path.name}`  —  {len(df)} kayıt")

    # ── Summary table ──
    st.markdown("#### Ozet Tablo")
    summary = df.groupby("algorithm").agg(
        Maliyet=("total_cost", "mean"),
        Emisyon=("total_emission", "mean"),
        Hizmet_Seviyesi=("service_level", "mean"),
        Odul=("total_reward", "mean"),
    ).round(2)
    st.dataframe(summary, use_container_width=True)

    # ── Metric selector ──
    st.markdown("#### Karsılastırmalı Grafikler")
    metric_choice = st.selectbox(
        "Metrik",
        ["total_cost", "total_emission", "service_level", "total_reward"],
        format_func=lambda x: {
            "total_cost":     "Toplam Maliyet",
            "total_emission": "CO2 Emisyonu",
            "service_level":  "Hizmet Seviyesi",
            "total_reward":   "Toplam Odul",
        }.get(x, x),
        label_visibility="collapsed",
    )

    # Aggregate per algorithm for bar chart
    chart_agg = (
        df.groupby("algorithm")[metric_choice]
        .mean()
        .reindex([a for a in ALGO_COLORS_HEX if a in df["algorithm"].values])
    )
    st.bar_chart(chart_agg)

    # ── Publication figures ──
    fig_dir = PROJECT_ROOT / "results" / "figures"
    png_files = sorted(fig_dir.glob(f"*{tier_key}*.png")) if fig_dir.exists() else []
    if png_files:
        st.markdown("#### Makale Grafikleri")
        cols = st.columns(min(len(png_files), 3))
        for i, f in enumerate(png_files[:3]):
            with cols[i % 3]:
                st.image(
                    str(f),
                    caption=f.stem.replace("_", " ").title(),
                    use_container_width=True,
                )

    # ── Raw data download ──
    with st.expander("Ham Veri"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "CSV Indir",
            csv_data,
            file_name=csv_path.name,
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Application entry point."""
    scenario_name, seed, model_path = render_sidebar()

    # ── Header ──
    st.markdown(
        "<div style='text-align:center; padding:0.3rem 0 0.8rem 0;'>"
        "<h1 style='color:#e2e8f0; margin-bottom:0.15rem;'>"
        "MAPPO — Afet Atık Yönetimi</h1>"
        "<p style='color:#64748b; font-size:1.0rem;'>"
        "Multi-Agent Proximal Policy Optimization ile Dinamik Atık Toplama"
        "</p></div>",
        unsafe_allow_html=True,
    )

    # ── Tabs (no emojis) ──
    tab1, tab2, tab3 = st.tabs([
        "Senaryo Goruntuleyici",
        "Model Test",
        "Kıyaslama Metrikleri",
    ])

    with tab1:
        tab_scenario_viewer(scenario_name, seed)

    with tab2:
        tab_model_test(scenario_name, seed, model_path)

    with tab3:
        tab_training_metrics(scenario_name)


if __name__ == "__main__":
    main()
