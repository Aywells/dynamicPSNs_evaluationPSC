import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
from scipy.stats import wilcoxon
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
import textwrap
import os
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,   # editable text
    "ps.fonttype": 42,
})

MISCLASS_YMAX = 0.5     # adjust if needed
RUNTIME_YMAX = 20000    # set to a number (e.g. 500) if you want a fixed max

COLOR_CYCLE = plt.get_cmap("Set1").colors

EXCEL_FILE = "results_PSC_DL (version 1).xlsx"
ALPHA = 0.05
OUTPUT_DIR_1 = "figures/ALL_METHODS"
OUTPUT_DIR_2 = "figures/FRANCIS_4_METHODS"
OUTPUT_DIR_3 = "figures/FRANCIS_3_METHODS"
OUTPUT_DIR_4 = "figures/AYDIN_3_METHODS"
OUTPUT_DIR_5 = "figures/AYDIN_2_METHODS_S_D_GCN"
OUTPUT_DIR_6 = "figures/BEST_METHODS_OVERALL"
OUTPUT_DIR_7 = "figures/FRANCIS_2_METHODS"
OUTPUT_DIR_8 = "figures/AYDIN_2_METHODS_DGCN_ONLY"

METHODS_1 = [
    "Dynamic graphlets + LR",
    "Dynamic graphlets + regular deep learning (2,3)",
    "Dynamic graphlets + regular deep learning (3,1,ReLu)",
    "Dynamic graphlets + regular deep learning (3,1,leakyReLu)",
    "Dynamic graphlets + regular deep learning (3,3)",
    "Default features + DGCN",
    "Dynamic graphlets + DGCN",
    "Dynamic graphlets + SGCN"
]

METHODS_2 = [
    "Dynamic graphlets + regular deep learning (2,3)",
    "Dynamic graphlets + regular deep learning (3,1,ReLu)",
    "Dynamic graphlets + regular deep learning (3,1,leakyReLu)",
    "Dynamic graphlets + regular deep learning (3,3)"
]

METHODS_3 = [
    "Dynamic graphlets + regular deep learning (2,3)",
    "Dynamic graphlets + regular deep learning (3,1,ReLu)",
    "Dynamic graphlets + regular deep learning (3,3)"
]

METHODS_4 = [
    "Default features + DGCN",
    "Dynamic graphlets + DGCN",
    "Dynamic graphlets + SGCN"
]

METHODS_5 = [
    "Dynamic graphlets + DGCN",
    "Dynamic graphlets + SGCN"
]

METHODS_6 = [
    "Dynamic graphlets + LR",
    "Dynamic graphlets + regular deep learning (2,3)",
    "Dynamic graphlets + SGCN"
]

METHODS_7 = [
    "Dynamic graphlets + regular deep learning (3,1,ReLu)",
    "Dynamic graphlets + regular deep learning (3,1,leakyReLu)"
]

METHODS_8 = [
    "Default features + DGCN",
    "Dynamic graphlets + DGCN"
]

MISCLASS_SUFFIX = "_agg_misclass"
RUNTIME_SUFFIX = "_run_time"

df = pd.read_excel(EXCEL_FILE)
datasets = df["Dataset"]

#-------------------------------

def plot_misclassification_lines(df, methods, OUTPUT_DIR, save=True):
    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(datasets))

    for i, m in enumerate(methods):
        ax.plot(
            x,
            df[f"{m}{MISCLASS_SUFFIX}"],
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=m,
            color=COLOR_CYCLE[i % len(COLOR_CYCLE)]
        )

    ax.set_ylabel("Misclassification rate")
    ax.set_xlabel("Dataset")

    ax.set_ylim(0, MISCLASS_YMAX)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=90, ha="center", va="top")

    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()

    if save and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(f"{OUTPUT_DIR}/misclassification_lineplot.pdf")

    plt.show()

def plot_runtime_lines(df, methods, OUTPUT_DIR, save=True):
    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(datasets))

    for i, m in enumerate(methods):
        ax.plot(
            x,
            df[f"{m}{RUNTIME_SUFFIX}"],
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=m,
            color=COLOR_CYCLE[i % len(COLOR_CYCLE)]
        )

    ax.set_ylabel("Runtime (min)")
    ax.set_xlabel("Dataset")

    if RUNTIME_YMAX is not None:
        ax.set_ylim(0, RUNTIME_YMAX)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=90, ha="center", va="top")

    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()

    if save:
        fig.savefig(f"{OUTPUT_DIR}/runtime_lineplot.pdf")

    plt.show()

plot_misclassification_lines(df,METHODS_1,OUTPUT_DIR_1)
plot_misclassification_lines(df,METHODS_2,OUTPUT_DIR_2)
plot_misclassification_lines(df,METHODS_3,OUTPUT_DIR_3)
plot_misclassification_lines(df,METHODS_4,OUTPUT_DIR_4)
plot_misclassification_lines(df,METHODS_5,OUTPUT_DIR_5)
plot_misclassification_lines(df,METHODS_6,OUTPUT_DIR_6)
plot_misclassification_lines(df,METHODS_7,OUTPUT_DIR_7)
plot_misclassification_lines(df,METHODS_8,OUTPUT_DIR_8)

plot_runtime_lines(df,METHODS_1,OUTPUT_DIR_1)
plot_runtime_lines(df,METHODS_2,OUTPUT_DIR_2)
plot_runtime_lines(df,METHODS_3,OUTPUT_DIR_3)
plot_runtime_lines(df,METHODS_4,OUTPUT_DIR_4)
plot_runtime_lines(df,METHODS_5,OUTPUT_DIR_5)
plot_runtime_lines(df,METHODS_6,OUTPUT_DIR_6)
plot_runtime_lines(df,METHODS_7,OUTPUT_DIR_7)
plot_runtime_lines(df,METHODS_8,OUTPUT_DIR_8)

#-------------------------------

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

def compute_misclassification_rankings(df, METHODS, OUTPUT_DIR):

    ROUND_DECIMALS = 3
    LABEL_WRAP = 20
    TOLERANCES = [0.0, 0.01, 0.02, 0.05, 0.10]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_col = "Dataset"
    misclass_cols = [c for c in df.columns if c.endswith(MISCLASS_SUFFIX)]
    selected = [c for c in misclass_cols if c[:-len(MISCLASS_SUFFIX)] in METHODS]
    misclass_cols = [f"{m}{MISCLASS_SUFFIX}" for m in METHODS if f"{m}{MISCLASS_SUFFIX}" in selected]

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
            dataset_col = "Dataset"
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
    # For plotting we only show Rank 1 (including ties) and shade the portion that is absolutely Rank 1
    for tol, rank_df in all_rank_dfs.items():
        methods = misclass_cols  # these are full column names like "<method>_agg_misclass"
        n_datasets = rank_df["Dataset"].nunique()

        # Count datasets where each method has Rank == 1 (including ties)
        total_rank1_counts = {
            m: rank_df[(rank_df["Method"] == m) & (rank_df["Rank"] == 1)]["Dataset"].nunique()
            for m in methods
        }

        # Identify datasets that have exactly one method at Rank == 1 (absolute winners)
        rank1_per_dataset = rank_df[rank_df["Rank"] == 1].groupby("Dataset").size()
        abs_rank1_datasets = set(rank1_per_dataset[rank1_per_dataset == 1].index.tolist())

        absolute_rank1_counts = {
            m: rank_df[
                (rank_df["Method"] == m)
                & (rank_df["Rank"] == 1)
                & (rank_df["Dataset"].isin(abs_rank1_datasets))
            ]["Dataset"].nunique()
            for m in methods
        }

        total_frac = np.array([total_rank1_counts[m] / max(1, n_datasets) for m in methods])
        abs_frac = np.array([absolute_rank1_counts[m] / max(1, n_datasets) for m in methods])

        # Clean labels: remove suffix like "_agg_misclass" for x-axis
        cleaned_labels = [m.replace(MISCLASS_SUFFIX, "") for m in methods]
        wrapped_labels = [wrap_label(lbl, LABEL_WRAP) for lbl in cleaned_labels]

        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.2), 6))

        # Main bar: total fraction of datasets where method is rank 1 (including ties)
        bar_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
        ax.bar(x, total_frac, color=bar_colors, edgecolor="black", linewidth=0.4, label="Rank 1 (incl. ties)")

        # Overlay shaded region representing absolute rank-1 (not tied)
        ax.bar(x, abs_frac, color="gray", alpha=0.6, edgecolor=None, label="Absolute rank 1 (no tie)")

        ax.set_ylabel("Proportion of datasets")
        ax.set_title(f"Proportion of rank 1 (tolerance ≤ {tol})")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped_labels, rotation=45, ha="right")

        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        plt.tight_layout()

        if OUTPUT_DIR:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            plt.savefig(
                os.path.join(OUTPUT_DIR, f"misclassification_rank1_plot_tol_{tol:.2f}.png"),
                dpi=300,
                bbox_inches="tight"
            )
        plt.close()

    return all_rank_dfs

compute_misclassification_rankings(df,METHODS_1,OUTPUT_DIR_1)
compute_misclassification_rankings(df,METHODS_2,OUTPUT_DIR_2)
compute_misclassification_rankings(df,METHODS_3,OUTPUT_DIR_3)
compute_misclassification_rankings(df,METHODS_4,OUTPUT_DIR_4)
compute_misclassification_rankings(df,METHODS_5,OUTPUT_DIR_5)
compute_misclassification_rankings(df,METHODS_6,OUTPUT_DIR_6)
compute_misclassification_rankings(df,METHODS_7,OUTPUT_DIR_7)
compute_misclassification_rankings(df,METHODS_8,OUTPUT_DIR_8)

#-------------------------------

def pairwise_wilcoxon_bonferroni(df, methods, suffix=MISCLASS_SUFFIX, output_excel="wilcoxon_pairwise.xlsx", alpha=ALPHA):
    
    methods = list(methods)
    n = len(methods)
    # Initialize matrices
    p_mat = pd.DataFrame(np.ones((n, n)), index=methods, columns=methods)
    p_adj_mat = pd.DataFrame(np.ones((n, n)), index=methods, columns=methods)

    all_p = []
    pair_keys = []

    # Collect p-values for every ordered pair (i != j)
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if i == j:
                p = 1.0
            else:
                col_i = f"{mi}{suffix}"
                col_j = f"{mj}{suffix}"
                if col_i not in df.columns or col_j not in df.columns:
                    p = 1.0
                else:
                    # paired, drop NA rows
                    paired = df[[col_i, col_j]].dropna()
                    if len(paired) < 1:
                        p = 1.0
                    else:
                        x = paired[col_i].values
                        y = paired[col_j].values
                        try:
                            # one-sided: test whether x < y (i has lower misclassification than j)
                            stat, p = wilcoxon(x, y, alternative="less")
                        except Exception:
                            # fallback when wilcoxon cannot compute (e.g. all zero diffs)
                            p = 1.0
            p_mat.at[mi, mj] = p
            if i != j:
                all_p.append(p)
                pair_keys.append((mi, mj))

    # Bonferroni correction across ordered pairs (count = n*(n-1))
    m_tests = n * (n - 1)
    all_p = np.array(all_p)
    adj_all = np.minimum(all_p * m_tests, 1.0)

    # Fill adjusted matrix
    for (mi, mj), p_adj in zip(pair_keys, adj_all):
        p_adj_mat.at[mi, mj] = p_adj

    # Interpretation matrix:
    # For each ordered pair (i,j): mark '<' if adjusted p(i<j) < alpha (i significantly lower than j),
    # '>' if adjusted p(j<i) < alpha (i significantly greater than j),
    # otherwise 'ns' (not significant). If both directions significant (rare), mark '<>'.
    interp = pd.DataFrame("", index=methods, columns=methods)
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if mi == mj:
                interp.at[mi, mj] = "-"
                continue
            p_i_j = p_adj_mat.at[mi, mj]
            p_j_i = p_adj_mat.at[mj, mi]
            sig_i_j = p_i_j < alpha
            sig_j_i = p_j_i < alpha
            if sig_i_j and not sig_j_i:
                interp.at[mi, mj] = "<"   # mi significantly better (lower) than mj
            elif sig_j_i and not sig_i_j:
                interp.at[mi, mj] = ">"   # mi significantly worse (higher) than mj
            elif sig_i_j and sig_j_i:
                interp.at[mi, mj] = "<>"  # both directions significant (uncommon)
            else:
                interp.at[mi, mj] = "ns"

    # Write results to Excel with 3 sheets
    with pd.ExcelWriter(output_excel) as w:
        p_mat.to_excel(w, sheet_name="p_values")
        p_adj_mat.to_excel(w, sheet_name="p_values_bonferroni")
        interp.to_excel(w, sheet_name="interpretation")

    return p_mat, p_adj_mat, interp

out_file = os.path.join(OUTPUT_DIR_1, "statistical_comparison_results.xlsx")
pvals, pvals_adj, interpretation = pairwise_wilcoxon_bonferroni(df, METHODS_1, output_excel=out_file)
print(f"Pairwise Wilcoxon p-values written to: {out_file}")

#-------------------------------

METHODS_DL = [
    "Dynamic graphlets + regular deep learning (2,3)",
    "Dynamic graphlets + regular deep learning (3,1,ReLu)",
    "Dynamic graphlets + regular deep learning (3,3)"
]

METHODS_GRAPH = [
    "Dynamic graphlets + DGCN",
    "Dynamic graphlets + SGCN"
]

FIXED_METHOD = "Dynamic graphlets + LR"

OUTPUT_DIR_6 = "figures/BEST_METHODS_PER_DATASET"

def select_best_methods(df):
    """
    For each dataset choose:
      - fixed method (FIXED_METHOD) value
      - best DL method (min misclassification among METHODS_DL)
      - best GCN method (min misclassification among METHODS_GRAPH)
    Returns a DataFrame with misclassification and runtime columns (no suffixes)
    and runtime columns for the chosen best methods.
    """
    rows = []
    for _, row in df.iterrows():
        # choose best by misclassification
        best_dl = min(METHODS_DL, key=lambda m: row[f"{m}{MISCLASS_SUFFIX}"])
        best_graph = min(METHODS_GRAPH, key=lambda m: row[f"{m}{MISCLASS_SUFFIX}"])
        rows.append({
            "Dataset": row["Dataset"],
            "Dynamic graphlets + LR": row[f"{FIXED_METHOD}{MISCLASS_SUFFIX}"],
            "Best of regular deep learning": row[f"{best_dl}{MISCLASS_SUFFIX}"],
            "Best of graph-based deep learning": row[f"{best_graph}{MISCLASS_SUFFIX}"],
            "Runtime Dynamic graphlets + LR": row[f"{FIXED_METHOD}{RUNTIME_SUFFIX}"],
            "Runtime Best of regular deep learning": row[f"{best_dl}{RUNTIME_SUFFIX}"],
            "Runtime Best of graph-based deep learning": row[f"{best_graph}{RUNTIME_SUFFIX}"],
        })
    return pd.DataFrame(rows)

def plot_best_misclassification_lines(df_best, OUTPUT_DIR, save=True):
    methods = ["Dynamic graphlets + LR", "Best of regular deep learning", "Best of graph-based deep learning"]
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(df_best["Dataset"]))
    for i, m in enumerate(methods):
        ax.plot(x, df_best[m], marker="o", markersize=6, linewidth=2, label=m, color=COLOR_CYCLE[i % len(COLOR_CYCLE)])
    ax.set_xticks(x)
    ax.set_xticklabels(df_best["Dataset"], rotation=90, ha="center", va="top")
    ax.set_ylabel("Misclassification rate")
    ax.set_ylim(0, MISCLASS_YMAX)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, "best_misclassification_lineplot.pdf"))
    plt.show()

def plot_best_runtime_lines(df_best, OUTPUT_DIR, save=True):
    methods = ["Runtime Dynamic graphlets + LR", "Runtime Best of regular deep learning", "Runtime Best of graph-based deep learning"]
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(df_best["Dataset"]))
    for i, m in enumerate(methods):
        ax.plot(x, df_best[m], marker="o", markersize=6, linewidth=2, label=m.replace("Runtime ", ""), color=COLOR_CYCLE[i % len(COLOR_CYCLE)])
    ax.set_xticks(x)
    ax.set_xticklabels(df_best["Dataset"], rotation=90, ha="center", va="top")
    ax.set_ylabel("Runtime (min)")
    if RUNTIME_YMAX is not None:
        ax.set_ylim(0, RUNTIME_YMAX)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, "best_runtime_lineplot.pdf"))
    plt.show()

# Create best-per-dataset DataFrame and plot
df_best = select_best_methods(df)
OUT_DIR_BEST = OUTPUT_DIR_6  # reuse defined OUTPUT_DIR_6 variable
os.makedirs(OUT_DIR_BEST, exist_ok=True)
plot_best_misclassification_lines(df_best, OUT_DIR_BEST)
plot_best_runtime_lines(df_best, OUT_DIR_BEST)

# Prepare DataFrame compatible with compute_misclassification_rankings:
# compute_misclassification_rankings expects columns named "<method>_agg_misclass"
methods_3 = ["Dynamic graphlets + LR", "Best of regular deep learning", "Best of graph-based deep learning"]
df_best_for_ranking = df_best.copy()
for m in methods_3:
    df_best_for_ranking[f"{m}{MISCLASS_SUFFIX}"] = df_best_for_ranking[m]
# keep Dataset column as expected
df_best_for_ranking = df_best_for_ranking[[ "Dataset" ] + [f"{m}{MISCLASS_SUFFIX}" for m in methods_3]]

# Run the same ranking pipeline (will produce ranking xlsx and Rank-1 plots)
compute_misclassification_rankings(df_best_for_ranking, methods_3, OUT_DIR_BEST)
print("Best-method plots and ranking outputs written to:", OUT_DIR_BEST)