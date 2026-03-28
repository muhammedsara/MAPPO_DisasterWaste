import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Enforced scenario order
SCENARIO_ORDER = ['S1-Small', 'S2-Medium', 'S3-Large', 'S4-Severe']
LABEL_MAP = {
    'S1_SMALL': 'S1-Small',
    'S2_MEDIUM': 'S2-Medium',
    'S3_LARGE': 'S3-Large',
    'S4_SEVERE': 'S4-Severe',
}

def main():
    # 1. Load all scenario CSVs
    results_dir = Path("results")
    csv_files = sorted(results_dir.glob("benchmark_results_S*.csv"))

    if not csv_files:
        print("Error: No CSV files found!")
        return

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # 2. Filter MAPPO and SinglePPO only
    df_ablation = df[df['algorithm'].isin(['MAPPO', 'SinglePPO'])].copy()

    # 3. Clean scenario labels and enforce order
    df_ablation['scenario'] = df_ablation['scenario'].map(LABEL_MAP)
    df_ablation['scenario'] = pd.Categorical(
        df_ablation['scenario'], categories=SCENARIO_ORDER, ordered=True
    )
    df_ablation = df_ablation.sort_values('scenario')

    # 4. Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", font_scale=1.1)

    ax = sns.barplot(
        data=df_ablation,
        x='scenario',
        y='total_cost',
        hue='algorithm',
        order=SCENARIO_ORDER,
        hue_order=['MAPPO', 'SinglePPO'],
        palette=['#2196F3', '#F44336'],
        capsize=0.1,
        err_kws={'linewidth': 1.5},
        edgecolor='0.3',
        linewidth=0.8,
    )

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.0f}', fontsize=8, padding=3)

    plt.title(
        "Ablation Study: CTDE (MAPPO) vs. Centralized Controller (SinglePPO)",
        fontsize=14, fontweight='bold', pad=15
    )
    plt.ylabel("Total Operational Cost", fontsize=12, fontweight='bold')
    plt.xlabel("Disaster Scenario Severity →", fontsize=12, fontweight='bold')
    plt.legend(title="Architecture", fontsize=11, title_fontsize=12, loc='upper left')
    plt.tight_layout()

    # 5. Save
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_ablation_cost.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / "fig_ablation_cost.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    print(f"✓ Saved: {out_dir / 'fig_ablation_cost.png'}")

if __name__ == "__main__":
    main()