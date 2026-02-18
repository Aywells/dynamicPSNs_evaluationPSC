import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
import textwrap

# -------------------------------
# Configuration
# -------------------------------
EXCEL_FILE = "results_PSC_DL.xlsx"
OUTPUT_DIR = "figures"
OUTPUT_RANKING_FILE = os.path.join(OUTPUT_DIR, "misclassification_rankings.xlsx")
OUTPUT_PLOT_FILE = os.path.join(OUTPUT_DIR, "misclassification_ranking_plot.png")

ROUND_DECIMALS = 3
LABEL_WRAP = 20  # characters per line
TOLERANCES = [0.0, 0.01, 0.02, 0.05, 0.10]
# -------------------------------
# Create output directory
# -------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Helper functions
# -------------------------------
def round_half_up(value, decimals=3):
    quant = Decimal(f"1.{'0'*decimals}")
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))

def wrap_label(label, width=20):
    return "\n".join(textwrap.wrap(label, width=width))

def compute_tolerance_competition_ranks(values, tolerance):
    """
    Anchored tolerance + competition ranking.
    values: pandas Series (method -> rounded value)
    """
    sorted_vals = values.sort_values()
    ranks = {}

    current_rank = 1
    group_start_value = None
    group_count = 0
    
    # Small epsilon for floating point comparison
    epsilon = 1e-9

    for method, val in sorted_vals.items():

        if group_start_value is None:
            # Start first group
            group_start_value = val
            group_count = 1
            ranks[method] = current_rank

        elif abs(val - group_start_value) <= tolerance + epsilon:  # Add epsilon here
            # Same anchored group
            group_count += 1
            ranks[method] = current_rank

        else:
            # New group → COMPETITION RANK jump
            current_rank = current_rank + group_count
            group_start_value = val
            group_count = 1
            ranks[method] = current_rank

    return ranks

# -------------------------------
# Load data
# -------------------------------
df = pd.read_excel(EXCEL_FILE)

METHOD_GROUPS = {
    "BEST_OF_DL": [
        "Dynamic graphlets + regular deep learning (2,3)",
        "Dynamic graphlets + regular deep learning (3,1,ReLu)",
        "Dynamic graphlets + regular deep learning (3,3)",
    ],
    "BEST_OF_GRAPH": [
        "Dynamic graphlets + DGCN",
        "Dynamic graphlets + SGCN",
    ],
    "LR": [
        "Dynamic graphlets + LR",
    ],
}

dataset_col = "Dataset"
misclass_cols = [c for c in df.columns if c.endswith("_agg_misclass")]
selected = misclass_cols[6:8] if len(misclass_cols) > 5 else []
misclass_cols = selected

# -------------------------------
# Round misclassification values
# -------------------------------
df_rounded = df.copy()
for col in misclass_cols:
    df_rounded[col] = df_rounded[col].apply(
        lambda x: round_half_up(x, ROUND_DECIMALS)
    )

# -------------------------------
# Compute competition rankings
# -------------------------------
all_rank_dfs = {}

for tol in TOLERANCES:
    rank_records = []

    for _, row in df_rounded.iterrows():
        dataset = row[dataset_col]
        values = row[misclass_cols]  # already rounded

        ranks = compute_tolerance_competition_ranks(values, tol)

        for method, rank in ranks.items():
            rank_records.append({
                "Dataset": dataset,
                "Method": method,
                "Misclassification_rounded": values[method],
                "Rank": rank,
                "Tolerance": tol
            })

    rank_df = pd.DataFrame(rank_records)
    all_rank_dfs[tol] = rank_df

    # -------------------------------
    # Output table (long-form, paper-safe)
    # -------------------------------
    rank_df = rank_df.sort_values(
        ["Dataset", "Rank", "Misclassification_rounded"]
    )

    rank_df.to_excel(
        os.path.join(
            OUTPUT_DIR,
            f"misclassification_rankings_tol_{tol:.2f}.xlsx"
        ),
        index=False
    )

    # -------------------------------
    # Output ranking table (per tolerance)
    # -------------------------------
    ranking_table = rank_df.sort_values(
        ["Dataset", "Rank", "Misclassification_rounded"]
    )

    ranking_table.to_excel(
        os.path.join(
            OUTPUT_DIR,
            f"misclassification_rankings_tol_{tol:.2f}.xlsx"
        ),
        index=False
    )

# -------------------------------
# Prepare data for stacked ranking plot
# -------------------------------
max_rank = rank_df["Rank"].max()
methods = misclass_cols

rank_props = defaultdict(float)

for method in methods:
    method_ranks = rank_df[rank_df["Method"] == method]["Rank"]
    total = len(method_ranks)

    for r in range(1, max_rank + 1):
        rank_props[(method, r)] = (method_ranks == r).sum() / total

plot_data = np.array([
    [rank_props[(m, r)] for m in methods]
    for r in range(1, max_rank + 1)
])

# -------------------------------
# Plot stacked ranking distribution
# -------------------------------
for tol, rank_df in all_rank_dfs.items():
    max_rank = rank_df["Rank"].max()
    methods = misclass_cols

    rank_props = defaultdict(float)

    for method in methods:
        method_ranks = rank_df[rank_df["Method"] == method]["Rank"]
        total = len(method_ranks)

        for r in range(1, max_rank + 1):
            rank_props[(method, r)] = (method_ranks == r).sum() / total

    plot_data = np.array([
        [rank_props[(m, r)] for m in methods]
        for r in range(1, max_rank + 1)
    ])

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(methods))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max_rank))

    for i in range(max_rank):
        ax.bar(
            range(len(methods)),
            plot_data[i],
            bottom=bottom,
            label=f"Rank {i+1}",
            color=colors[i]
        )
        bottom += plot_data[i]

    wrapped_labels = [wrap_label(m, LABEL_WRAP) for m in methods]

    ax.set_ylabel("Proportion of datasets")
    ax.set_title(f"Misclassification Ranking (Tolerance ≤ {tol})")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(wrapped_labels, rotation=45, ha="right")

    ax.legend(title="Ranking", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"misclassification_ranking_plot_tol_{tol:.2f}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

