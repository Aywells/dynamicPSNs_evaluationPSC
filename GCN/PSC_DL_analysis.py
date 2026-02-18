import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
from scipy.stats import wilcoxon
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
import textwrap
import os

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

MISCLASS_YMAX = 0.4     # adjust if needed
RUNTIME_YMAX = 16000    # set to a number (e.g. 500) if you want a fixed max

COLOR_CYCLE = plt.get_cmap("Set1").colors

EXCEL_FILE = "results_PSC_DL.xlsx"
ALPHA = 0.05
OUTPUT_DIR = "figures"

METHODS = [
    "Dynamic graphlets + LR",
    "Dynamic graphlets + regular deep learning (2,3)",
    "Dynamic graphlets + regular deep learning (3,1,ReLu)",
    "Dynamic graphlets + regular deep learning (3,1,leakyReLu)",
    "Dynamic graphlets + regular deep learning (3,3)",
    "Default features + DGCN",
    "Dynamic graphlets + DGCN",
    "Dynamic graphlets + SGCN"
]

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
MISCLASS_SUFFIX = "_agg_misclass"
RUNTIME_SUFFIX = "_run_time"

df = pd.read_excel(EXCEL_FILE)
datasets = df["Dataset"]

def plot_misclassification_lines(df, methods, save=True):
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

    if save:
        fig.savefig(f"{OUTPUT_DIR}/misclassification_lineplot.pdf")
        fig.savefig(f"{OUTPUT_DIR}/misclassification_lineplot.svg")

    plt.show()

def plot_runtime_lines(df, methods, save=True):
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
        fig.savefig(f"{OUTPUT_DIR}/runtime_lineplot.svg")

    plt.show()

def plot_rank_distribution(rank_counts, title="", save_path=None):
    methods = list(rank_counts.keys())
    ranks = sorted(rank_counts[methods[0]].keys())

    data = np.array([[rank_counts[m][r] for r in ranks] for m in methods], dtype=float)

    # Normalize to proportions (ignoring datasets where method is missing)
    data_sum = data.sum(axis=1, keepdims=True)
    # Avoid division by zero
    data_sum[data_sum == 0] = 1
    data /= data_sum

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(methods))

    for i, r in enumerate(ranks):
        ax.bar(
            methods,
            data[:, i],
            bottom=bottom,
            label=f"Rank {r}",
            edgecolor="black",
            linewidth=0.4
        )
        bottom += data[:, i]

    ax.set_ylabel("Proportion of datasets")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(wrap_labels(methods, width=25))
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )
    # Rotate x-axis labels 30 degrees
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path + ".pdf")
        fig.savefig(save_path + ".svg")

    plt.show()

def corrected_wilcoxon_pvalue_matrices(
    df,
    METHODS,
    MISCLASS_SUFFIX,
    alternative="two-sided",
    correction="holm",
    alpha=0.05,
    min_pairs=2
):
    """
    Computes pairwise corrected p-value matrices using the paired
    Wilcoxon signed-rank test.

    Parameters
    ----------
    df : pd.DataFrame
        Rows = datasets
    METHODS : list[str]
        Method names (without suffix)
    MISCLASS_SUFFIX : str
        Column suffix (e.g. "_agg_misclass")
    alternative : {"two-sided", "less", "greater"}
        Wilcoxon alternative hypothesis
    correction : {"holm", "bonferroni", "fdr_bh", ...}
        Multiple-testing correction method
    alpha : float
        Significance level
    min_pairs : int
        Minimum number of paired samples required

    Returns
    -------
    raw_pvals : pd.DataFrame
        Raw p-value matrix
    corrected_pvals : pd.DataFrame
        Corrected p-value matrix
    significant : pd.DataFrame (bool)
        Significance matrix after correction
    """

    # Initialize matrices
    raw_pvals = pd.DataFrame(
        np.nan, index=METHODS, columns=METHODS, dtype=float
    )

    comparisons = []
    index_pairs = []

    # Compute raw p-values
    for m1 in METHODS:
        for m2 in METHODS:
            if m1 == m2:
                continue

            pair_df = df[
                [f"{m1}{MISCLASS_SUFFIX}", f"{m2}{MISCLASS_SUFFIX}"]
            ].dropna()

            if len(pair_df) < min_pairs:
                continue

            x = pair_df.iloc[:, 0].values
            y = pair_df.iloc[:, 1].values

            try:
                _, p = wilcoxon(x, y, alternative=alternative)
            except ValueError:
                p = np.nan

            raw_pvals.loc[m1, m2] = p

            if not np.isnan(p):
                comparisons.append(p)
                index_pairs.append((m1, m2))

    # Multiple-testing correction
    corrected = np.full(len(comparisons), np.nan)

    if len(comparisons) > 0:
        corrected = multipletests(
            comparisons,
            alpha=alpha,
            method=correction
        )[1]

    # Fill corrected matrix
    corrected_pvals = raw_pvals.copy()

    for (m1, m2), p_corr in zip(index_pairs, corrected):
        corrected_pvals.loc[m1, m2] = p_corr

    significant = corrected_pvals < alpha

    return raw_pvals, corrected_pvals, significant

def generate_ranking_table(df, methods, decimals=2, suffix="_agg_misclass"):
    """
    Generates a long-form table with:
        Dataset | Method | Value | Rank
    Uses competition-based ranking (1-1-1-4 tie handling)
    and rounds values to specified decimal places.
    """
    rows = []

    for _, row in df.iterrows():
        # Round misclassification values for ranking
        rounded_values = pd.Series({
            m: round(row[f"{m}{suffix}"], decimals)
            if not pd.isna(row[f"{m}{suffix}"]) else np.nan
            for m in methods
        })

        # Compute ranks, ignoring NaNs
        ranks = competition_ranking(rounded_values)

        # Store results
        for m in methods:
            value = rounded_values[m] if not pd.isna(rounded_values[m]) else np.nan
            rank = ranks[m] if m in ranks else np.nan
            rows.append({
                "Dataset": row["Dataset"],
                "Method": m,
                "Value": value,
                "Rank": rank
            })

    ranking_table = pd.DataFrame(rows)
    return ranking_table

def competition_ranking(values: pd.Series):
    """
    Competition ranking (1,1,1,4 style).
    Lower values are better.
    """
    sorted_vals = values.sort_values()
    ranks = {}

    current_rank = 1
    i = 0

    while i < len(sorted_vals):
        tied_mask = sorted_vals == sorted_vals.iloc[i]
        tied_methods = sorted_vals.index[tied_mask]

        for m in tied_methods:
            ranks[m] = current_rank

        tie_count = len(tied_methods)
        current_rank += tie_count
        i += tie_count

    return ranks

def compute_rank_counts_with_precision(df, methods, decimals):
    """
    Compute competition-based rank counts after rounding values.
    Ignores missing misclassification values.
    Returns a dict: {method -> {rank -> count}}
    """
    max_rank = len(methods)
    rank_counts = {m: {r: 0 for r in range(1, max_rank + 1)} for m in methods}

    for _, row in df.iterrows():
        rounded = pd.Series({
            m: round(row[f"{m}{MISCLASS_SUFFIX}"], decimals)
            if not pd.isna(row[f"{m}{MISCLASS_SUFFIX}"]) else np.nan
            for m in methods
        })
        ranks = competition_ranking(rounded)
        for m, r in ranks.items():
            rank_counts[m][r] += 1

    return rank_counts

def select_best_methods_per_dataset(df):
    rows = []

    for _, row in df.iterrows():
        # baseline
        baseline = BASELINE_METHOD

        # best DL method
        dl_values = {
            m: row[f"{m}{MISCLASS_SUFFIX}"] for m in DL_GROUP
        }
        best_dl = min(dl_values, key=dl_values.get)

        # best graph-based method
        graph_values = {
            m: row[f"{m}{MISCLASS_SUFFIX}"] for m in GRAPH_GROUP
        }
        best_graph = min(graph_values, key=graph_values.get)

        selected = [baseline, best_dl, best_graph]

        for m in selected:
            rows.append({
                "Dataset": row["Dataset"],
                "Method": m,
                "Misclassification": row[f"{m}{MISCLASS_SUFFIX}"]
            })

    return pd.DataFrame(rows)

def plot_selected_misclassification(df_selected, save=True):
    fig, ax = plt.subplots(figsize=(14, 5))

    datasets = df_selected["Dataset"].unique()
    x = np.arange(len(datasets))

    for method in df_selected["Method"].unique():
        y = (
            df_selected[df_selected["Method"] == method]
            .set_index("Dataset")
            .reindex(datasets)["Misclassification"]
        )

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            label=method
        )

    ax.set_ylabel("Misclassification rate")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, MISCLASS_YMAX)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=90)

    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        fig.savefig(f"{OUTPUT_DIR}/misclassification_selected.pdf")
        fig.savefig(f"{OUTPUT_DIR}/misclassification_selected.svg")

    plt.show()

def compute_selected_rank_counts(df_selected):
    methods = df_selected["Method"].unique().tolist()
    max_rank = 3

    rank_counts = {
        m: {r: 0 for r in range(1, max_rank + 1)}
        for m in methods
    }

    for dataset in df_selected["Dataset"].unique():
        subset = (
            df_selected[df_selected["Dataset"] == dataset]
            .set_index("Method")["Misclassification"]
        )

        ranks = competition_ranking(subset)

        for m, r in ranks.items():
            rank_counts[m][r] += 1

    return rank_counts

def plot_selected_rank_distribution(rank_counts, save=True):
    methods = list(rank_counts.keys())
    ranks = [1, 2, 3]

    data = np.array([
        [rank_counts[m][r] for r in ranks]
        for m in methods
    ], dtype=float)

    data /= data.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(methods))

    for i, r in enumerate(ranks):
        ax.bar(
            methods,
            data[:, i],
            bottom=bottom,
            label=f"Rank {r}",
            edgecolor="black",
            linewidth=0.4
        )
        bottom += data[:, i]

    ax.set_ylabel("Proportion of datasets")
    ax.set_ylim(0, 1.0)
    ax.set_title("Rank distribution (best-per-dataset selection)")

    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        fig.savefig(f"{OUTPUT_DIR}/rank_distribution_selected.pdf")
        fig.savefig(f"{OUTPUT_DIR}/rank_distribution_selected.svg")

    plt.show()

def wrap_labels(labels, max_len=20):
    """
    Break long labels into multiple lines for plotting.
    """
    new_labels = []
    for lab in labels:
        if len(lab) > max_len:
            # split into chunks of max_len characters
            chunks = [lab[i:i+max_len] for i in range(0, len(lab), max_len)]
            new_labels.append("\n".join(chunks))
        else:
            new_labels.append(lab)
    return new_labels

def select_best_methods(df):
    """
    Creates a new dataframe with 3 columns:
    - 'Dynamic graphlets + LR'
    - 'Best of regular deep learning'
    - 'Best of GCN'
    """
    new_df = pd.DataFrame()
    new_df['Dataset'] = df['Dataset']

    # Always include Dynamic graphlets + LR
    new_df['Dynamic graphlets + LR'] = df['Dynamic graphlets + LR_agg_misclass']

    # Best of regular deep learning (minimum misclassification)
    reg_dl_methods = [
        "Dynamic graphlets + regular deep learning (2,3)",
        "Dynamic graphlets + regular deep learning (3,1,ReLu)",
        "Dynamic graphlets + regular deep learning (3,1,leakyReLu)",
        "Dynamic graphlets + regular deep learning (3,3)"
    ]
    new_df['Best of regular deep learning'] = df[[f"{m}_agg_misclass" for m in reg_dl_methods]].min(axis=1)

    # Best of GCN/DGCN methods
    gcn_methods = [
        "Default features + DGCN",
        "Dynamic graphlets + DGCN",
        "Dynamic graphlets + SGCN"
    ]
    new_df['Best of GCN'] = df[[f"{m}_agg_misclass" for m in gcn_methods]].min(axis=1)

    return new_df

def compute_ranks_for_selected(df_selected):
    """
    Computes competition ranks for the 3-method dataset.
    Returns a rank table compatible with plotting.
    """
    methods = ['Dynamic graphlets + LR', 'Best of regular deep learning', 'Best of GCN']
    rank_counts = {
        m: {r: 0 for r in range(1, len(methods)+1)}
        for m in methods
    }

    for _, row in df_selected.iterrows():
        values = row[methods]
        ranks = competition_ranking(values)  # using your earlier competition_ranking function
        for m, r in ranks.items():
            rank_counts[m][r] += 1

    return rank_counts

def plot_selected_misclassification_lines(df_selected, save=True):
    methods = ['Dynamic graphlets + LR', 'Best of regular deep learning', 'Best of GCN']
    fig, ax = plt.subplots(figsize=(12,5))
    x = np.arange(len(df_selected['Dataset']))

    for i, m in enumerate(methods):
        ax.plot(
            x,
            df_selected[m],
            marker='o',
            markersize=6,
            linewidth=2,
            label=m,
            color=COLOR_CYCLE[i % len(COLOR_CYCLE)]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df_selected['Dataset'], rotation=90, ha='center', va='top')
    ax.set_ylabel("Misclassification rate")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, MISCLASS_YMAX)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        fig.savefig(f"{OUTPUT_DIR}/misclassification_selected_lineplot.pdf")
        fig.savefig(f"{OUTPUT_DIR}/misclassification_selected_lineplot.svg")

    plt.show()

def plot_selected_rank_distribution(rank_counts, save=True):
    methods = list(rank_counts.keys())
    ranks = sorted(rank_counts[methods[0]].keys())

    data = np.array([[rank_counts[m][r] for r in ranks] for m in methods], dtype=float)
    data /= data.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8,4))
    bottom = np.zeros(len(methods))
    for i, r in enumerate(ranks):
        ax.bar(
            methods,
            data[:,i],
            bottom=bottom,
            label=f"Rank {r}",
            edgecolor='black',
            linewidth=0.4
        )
        bottom += data[:,i]

    ax.set_ylabel("Proportion of datasets")
    ax.set_ylim(0,1.0)
    ax.set_title("Rank distribution (selected methods)")

    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', frameon=False)
    fig.tight_layout()

    if save:
        fig.savefig(f"{OUTPUT_DIR}/rank_distribution_selected.pdf")
        fig.savefig(f"{OUTPUT_DIR}/rank_distribution_selected.svg")

    plt.show()

def select_best_methods(df):
    """
    Create a new dataframe with 3 columns:
    - 'Dynamic graphlets + LR'
    - 'Best of regular deep learning'
    - 'Best of GCN'
    """
    new_df = pd.DataFrame()
    new_df['Dataset'] = df['Dataset']

    # Always include Dynamic graphlets + LR
    new_df['Dynamic graphlets + LR'] = df['Dynamic graphlets + LR_agg_misclass']

    # Best of regular deep learning
    reg_dl_methods = [
        "Dynamic graphlets + regular deep learning (2,3)",
        "Dynamic graphlets + regular deep learning (3,1,ReLu)",
        "Dynamic graphlets + regular deep learning (3,1,leakyReLu)",
        "Dynamic graphlets + regular deep learning (3,3)"
    ]
    new_df['Best of regular deep learning'] = df[[f"{m}_agg_misclass" for m in reg_dl_methods]].min(axis=1)

    # Best of GCN/DGCN methods
    gcn_methods = [
        "Default features + DGCN",
        "Dynamic graphlets + DGCN",
        "Dynamic graphlets + SGCN"
    ]
    new_df['Best of GCN'] = df[[f"{m}_agg_misclass" for m in gcn_methods]].min(axis=1)

    return new_df

def compute_ranks_selected(df_selected, decimals=3):
    """
    Compute competition rank counts for the 3 selected methods.
    decimals: int or None, rounds values before ranking
    """
    methods = ['Dynamic graphlets + LR', 'Best of regular deep learning', 'Best of GCN']
    rank_counts = {m: {r: 0 for r in range(1, len(methods)+1)} for m in methods}

    for _, row in df_selected.iterrows():
        values = row[methods]
        ranks = competition_ranking(values, decimals=decimals)
        for m, r in ranks.items():
            rank_counts[m][r] += 1

    return rank_counts

def generate_rank_table_selected(df_selected, decimals=None):
    """
    Long-form rank table with competition ranking for the 3 selected methods
    Columns: Dataset | Method | Value | Rounded_Value | Rank
    """
    methods = ['Dynamic graphlets + LR', 'Best of regular deep learning', 'Best of GCN']
    rows = []

    for _, row in df_selected.iterrows():
        values = pd.Series({m: row[m] for m in methods})
        if decimals is not None:
            rounded = values.round(decimals)
        else:
            rounded = values.copy()

        ranks = competition_ranking(rounded, decimals=None)  # rounding already applied
        for m in methods:
            rows.append({
                'Dataset': row['Dataset'],
                'Method': m,
                'Value': values[m],
                'Rounded_Value': rounded[m],
                'Rank': ranks[m]
            })

    return pd.DataFrame(rows)

def competition_ranking_absolute(values):
    """
    Competition ranking based on absolute difference to the best value.
    Lower absolute difference = better.

    Parameters
    ----------
    values : pd.Series (method -> value)

    Returns
    -------
    dict(method -> rank)
    """
    best = values.min()
    abs_diff = (values - best).abs()

    sorted_diff = abs_diff.sort_values()
    ranks = {}

    current_rank = 1
    i = 0
    n = len(sorted_diff)

    while i < n:
        tied_methods = sorted_diff.index[
            sorted_diff == sorted_diff.iloc[i]
        ]

        for m in tied_methods:
            ranks[m] = current_rank

        tie_count = len(tied_methods)
        current_rank += tie_count
        i += tie_count

    return ranks

def compute_rank_counts_absolute_precision(df, METHODS, suffix, decimals):
    """
    Computes rank counts using absolute differences with rounding.

    Parameters
    ----------
    decimals : int (2 or 3)

    Returns
    -------
    rank_counts : dict
        method -> {rank -> count}
    """
    max_rank = len(METHODS)

    rank_counts = {
        m: {r: 0 for r in range(1, max_rank + 1)}
        for m in METHODS
    }

    for _, row in df.iterrows():
        rounded = pd.Series({
            m: round(row[f"{m}{suffix}"], decimals)
            for m in METHODS
        })

        ranks = competition_ranking_absolute(rounded)

        for m, r in ranks.items():
            rank_counts[m][r] += 1

    return rank_counts

def generate_absolute_rank_table(df, METHODS, suffix, decimals, metric_name):
    """
    Generates a detailed ranking table using absolute-difference ranking.

    Columns:
    Dataset | Method | Value | AbsoluteDifference | Rank
    """
    rows = []

    for _, row in df.iterrows():
        rounded = pd.Series({
            m: round(row[f"{m}{suffix}"], decimals)
            for m in METHODS
        })

        best = rounded.min()
        abs_diff = (rounded - best).abs()
        ranks = competition_ranking_absolute(rounded)

        for m in METHODS:
            rows.append({
                "Dataset": row["Dataset"],
                "Method": m,
                metric_name: rounded[m],
                "AbsoluteDifference": abs_diff[m],
                "Rank": ranks[m]
            })

    return pd.DataFrame(rows)

def rank_by_absolute_rounding(df, methods, suffix, decimals):
    """
    Absolute-difference ranking after rounding values.
    """
    rank_rows = []

    for _, row in df.iterrows():
        values = pd.Series({
            m: round(row[f"{m}{suffix}"], decimals)
            for m in methods
        })

        ranks = competition_ranking(values)

        for m in methods:
            rank_rows.append({
                "Dataset": row["Dataset"],
                "Method": m,
                "Value": row[f"{m}{suffix}"],
                "Rounded_Value": values[m],
                "Rank": ranks[m],
                "Criterion": f"abs_round_{decimals}dp"
            })

    return pd.DataFrame(rank_rows)

def rank_by_relative_tolerance(df, methods, suffix, tol):
    """
    Absolute-difference ranking using relative tolerance.
    Robust to zero values.

    tol = 0.01, 0.02, 0.05, 0.10
    """
    rank_rows = []

    for _, row in df.iterrows():
        raw_vals = pd.Series({
            m: row[f"{m}{suffix}"]
            for m in methods
        }).sort_values()

        grouped = []
        used = set()

        for m in raw_vals.index:
            if m in used:
                continue

            group = [m]
            used.add(m)

            for n in raw_vals.index:
                if n in used:
                    continue

                x = raw_vals[m]
                y = raw_vals[n]

                # ---- ZERO-SAFE RELATIVE DIFFERENCE ----
                if x == 0 and y == 0:
                    is_tie = True
                elif x == 0 or y == 0:
                    is_tie = False
                else:
                    rel_diff = abs(x - y) / min(x, y)
                    is_tie = rel_diff <= tol
                # --------------------------------------

                if is_tie:
                    group.append(n)
                    used.add(n)

            grouped.append(group)

        # Competition ranking
        rank = 1
        ranks = {}

        for group in grouped:
            for m in group:
                ranks[m] = rank
            rank += len(group)

        for m in methods:
            rank_rows.append({
                "Dataset": row["Dataset"],
                "Method": m,
                "Value": row[f"{m}{suffix}"],
                "Rank": ranks[m],
                "Criterion": f"rel_tol_{int(tol * 100)}pct"
            })

    return pd.DataFrame(rank_rows)

def plot_rank_distribution_from_table(rank_df, title, save_path=None):
    rank_counts = (
        rank_df
        .groupby(["Method", "Rank"])
        .size()
        .unstack(fill_value=0)
    )

    proportions = rank_counts.div(rank_counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(proportions))

    for r in proportions.columns:
        ax.bar(
            proportions.index,
            proportions[r],
            bottom=bottom,
            label=f"Rank {r}",
            edgecolor="black",
            linewidth=0.4
        )
        bottom += proportions[r].values

    ax.set_ylabel("Proportion of datasets")
    ax.set_ylim(0, 1)
    ax.set_title(title)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        fig.savefig(save_path.replace(".pdf", ".svg"))

    plt.show()

def plot_rank_distribution_by_criterion(
    rank_table,
    criterion,
    save_dir="figures"
):
    """
    Generates a stacked rank-distribution plot
    for a specific ranking criterion.
    """

    sub = rank_table[rank_table["Criterion"] == criterion]

    rank_counts = (
        sub
        .groupby(["Method", "Rank"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )

    proportions = rank_counts.div(rank_counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(proportions))

    for r in proportions.columns:
        ax.bar(
            proportions.index,
            proportions[r],
            bottom=bottom,
            label=f"Rank {r}",
            edgecolor="black",
            linewidth=0.4
        )
        bottom += proportions[r].values

    ax.set_ylabel("Proportion of datasets")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Rank distribution ({criterion})")

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(f"{save_dir}/rank_distribution_{criterion}.pdf")
    fig.savefig(f"{save_dir}/rank_distribution_{criterion}.svg")

    plt.show()

def competition_ranking_absolute(values, tolerance_pct):
    """
    values: pd.Series (method -> value), lower is better
    tolerance_pct: float (e.g. 0.01 for 1%)

    Returns: dict(method -> rank)
    """
    best = values.min()
    abs_diff_pct = ((values - best).abs()) / best

    df_rank = pd.DataFrame({
        "method": values.index,
        "abs_diff_pct": abs_diff_pct.values
    }).sort_values("abs_diff_pct")

    ranks = {}
    current_rank = 1

    while not df_rank.empty:
        base_diff = df_rank.iloc[0]["abs_diff_pct"]

        # Correct tie definition
        tied_mask = abs(df_rank["abs_diff_pct"] - base_diff) <= tolerance_pct
        tied = df_rank[tied_mask]

        for m in tied["method"]:
            ranks[m] = current_rank

        current_rank += len(tied)
        df_rank = df_rank.loc[~tied_mask]

    return ranks

def compute_rank_counts_absolute(
    df,
    METHODS,
    MISCLASS_SUFFIX,
    decimals,
    tolerance_pct
):
    rank_counts = {m: {} for m in METHODS}
    rows = []

    for _, row in df.iterrows():
        values = {
            m: round(row[f"{m}{MISCLASS_SUFFIX}"], decimals)
            for m in METHODS
            if pd.notna(row[f"{m}{MISCLASS_SUFFIX}"])
        }

        if len(values) < 2:
            continue

        values = pd.Series(values)
        ranks = competition_ranking_absolute(values, tolerance_pct)

        for m, r in ranks.items():
            rank_counts[m][r] = rank_counts[m].get(r, 0) + 1

            rows.append({
                "Dataset": row["Dataset"],
                "Method": m,
                "Value": values[m],
                "Rank": r,
                "Decimals": decimals,
                "Tolerance (%)": tolerance_pct * 100
            })

    # Normalize rank keys
    max_rank = max(
        (max(rc.keys()) for rc in rank_counts.values() if rc),
        default=0
    )

    for m in rank_counts:
        for r in range(1, max_rank + 1):
            rank_counts[m].setdefault(r, 0)

    return rank_counts, pd.DataFrame(rows)

def abs_diff_ranking(values, abs_thresh):
    """
    values: pandas Series (method -> value)
    abs_thresh: maximum difference allowed to consider methods tied
                Can be absolute (e.g., 0.01 = 1%) or relative fraction (0.01 = 1% of value)
    Returns: dict of method -> rank
    """
    # Sort methods
    sorted_methods = values.sort_values()
    methods = sorted_methods.index.tolist()
    sorted_vals = sorted_methods.values
    
    ranks = {}
    current_rank = 1
    i = 0
    n = len(sorted_vals)
    
    while i < n:
        # Methods tied with first in this block
        tie_indices = [i]
        for j in range(i + 1, n):
            # Relative difference vs first in block
            if abs(sorted_vals[j] - sorted_vals[i]) <= abs_thresh:
                tie_indices.append(j)
            else:
                break
        # Assign current_rank to all tied methods
        for idx in tie_indices:
            ranks[methods[idx]] = current_rank
        # Advance
        current_rank += len(tie_indices)
        i += len(tie_indices)
    
    return ranks

# ===========================
# Compute rank counts for all datasets and thresholds
# ===========================
def compute_rank_counts_absdiff(df, methods, decimals, abs_percent):
    """
    abs_percent: fraction (0.01=1%, 0.02=2%, 0.05=5%, 0.10=10%)
    decimals: 2 or 3
    """
    max_rank = len(methods)
    rank_counts = {m: {r:0 for r in range(1, max_rank+1)} for m in methods}
    rank_table = []  # For output table
    
    for _, row in df.iterrows():
        # Use exact values, rounded only for display
        values = pd.Series({m: row[f"{m}{MISCLASS_SUFFIX}"] for m in methods})
        # Absolute threshold = fraction of the best value
        best_val = values.min()
        abs_thresh = best_val * abs_percent
        
        ranks = abs_diff_ranking(values, abs_thresh)
        
        for m, r in ranks.items():
            rank_counts[m][r] += 1
            rank_table.append({
                "Dataset": row["Dataset"],
                "Method": m,
                "Value": round(values[m], decimals),
                "Rank": r,
                "Decimals": decimals,
                "AbsDiffThreshold": abs_percent
            })
    
    return rank_counts, pd.DataFrame(rank_table)

# ===========================
# Plot rank distributions
# ===========================
def plot_rank_distribution(rank_counts, title, filename):
    methods = list(rank_counts.keys())
    ranks = sorted(rank_counts[methods[0]].keys())
    
    data = np.array([[rank_counts[m][r] for r in ranks] for m in methods], dtype=float)
    data /= data.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(10,5))
    bottom = np.zeros(len(methods))
    
    for i, r in enumerate(ranks):
        ax.bar(methods, data[:,i], bottom=bottom, label=f"Rank {r}", edgecolor="black", linewidth=0.4)
        bottom += data[:,i]
    
    ax.set_ylabel("Proportion of datasets")
    ax.set_ylim(0,1.0)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(filename + ".pdf")
    fig.savefig(filename + ".svg")
    plt.show()

def compressed_absdiff_ranking(values, abs_thresh=0):
    """
    Compute rankings with "compressed ranks":
    - values: pandas Series (method -> value)
    - abs_thresh: absolute difference threshold for tying
    Returns: dict(method -> rank)
    
    Rules:
    - Methods within abs_thresh are tied
    - Next distinct group always increments rank by 1
    """
    # Sort methods by value (lower is better)
    sorted_methods = values.sort_values()
    methods = sorted_methods.index.tolist()
    sorted_vals = sorted_methods.values
    
    ranks = {}
    current_rank = 1
    i = 0
    n = len(sorted_vals)
    
    while i < n:
        # Identify all methods tied with sorted_vals[i]
        tie_indices = [i]
        for j in range(i+1, n):
            if abs(sorted_vals[j] - sorted_vals[i]) <= abs_thresh:
                tie_indices.append(j)
            else:
                break
        # Assign current rank to all tied methods
        for idx in tie_indices:
            ranks[methods[idx]] = current_rank
        # Move to next group, increment rank by 1 (compressed)
        current_rank += 1
        i += len(tie_indices)
    
    return ranks

def compute_rank_counts_compressed(df, methods, decimals, abs_percent):
    """
    Compute compressed ranks for all datasets and return:
    - rank_counts: dict[method][rank] = count
    - rank_table: long-form DataFrame
    """
    max_rank = len(methods)
    rank_counts = {m: {r:0 for r in range(1,max_rank+1)} for m in methods}
    rank_table = []

    for _, row in df.iterrows():
        values = pd.Series({m: row[f"{m}{MISCLASS_SUFFIX}"] for m in methods})
        best_val = values.min()
        abs_thresh = best_val * abs_percent
        ranks = compressed_absdiff_ranking(values, abs_thresh)
        for m, r in ranks.items():
            rank_counts[m][r] += 1
            rank_table.append({
                "Dataset": row["Dataset"],
                "Method": m,
                "Value": round(values[m], decimals),
                "Rank": r,
                "Decimals": decimals,
                "AbsDiffThreshold": abs_percent
            })

    return rank_counts, pd.DataFrame(rank_table)

# ===========================
# Plot rank distributions with tie shading
# ===========================
def plot_rank_distribution_with_ties(rank_table, methods, title, filename, max_label_len=20):
    """
    Rank distribution plot with shaded ties and wrapped x-axis labels.
    """
    max_rank = rank_table['Rank'].max()
    data_total = np.zeros((len(methods), max_rank))
    data_ties  = np.zeros((len(methods), max_rank))
    
    for i, m in enumerate(methods):
        df_m = rank_table[rank_table['Method'] == m]
        n_datasets = df_m['Dataset'].nunique()
        for r in range(1, max_rank+1):
            df_r = df_m[df_m['Rank'] == r]
            data_total[i, r-1] = len(df_r)/n_datasets
            tie_count = 0
            for ds in df_r['Dataset'].unique():
                df_ds = rank_table[(rank_table['Dataset']==ds) & (rank_table['Rank']==r)]
                if len(df_ds) > 1:
                    tie_count += 1
            data_ties[i, r-1] = tie_count/n_datasets
    
    fig, ax = plt.subplots(figsize=(max(12, len(methods)*1.5),5))
    bottom = np.zeros(len(methods))
    
    for r in range(max_rank):
        ax.bar(methods, data_total[:,r], bottom=bottom, color=COLOR_CYCLE[r%len(COLOR_CYCLE)],
               edgecolor='black', linewidth=0.4)
        ax.bar(methods, data_ties[:,r], bottom=bottom, color='k', alpha=0.3, edgecolor=None)
        bottom += data_total[:,r]
    
    ax.set_ylabel("Proportion of datasets")
    ax.set_ylim(0,1.0)
    ax.set_title(title)
    
    # Wrap labels
    wrapped_labels = wrap_labels(methods, max_len=max_label_len)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(wrapped_labels, rotation=45, ha="center")
    
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='gray', alpha=0.3, label='Tie fraction')],
              loc='upper right', frameon=False)
    
    fig.tight_layout()
    fig.savefig(filename + ".pdf")
    fig.savefig(filename + ".svg")
    plt.show()

def group_ranking(values, abs_thresh=0):
    """
    Compressed ranking by group (rank increments by 1 per group)
    
    Parameters:
    - values: pd.Series (method -> value), lower is better
    - abs_thresh: absolute difference threshold for grouping ties
    
    Returns:
    - dict(method -> rank)
    """
    # Sort methods by value
    sorted_vals = values.sort_values()
    methods = sorted_vals.index.tolist()
    sorted_values = sorted_vals.values
    
    ranks = {}
    current_rank = 1
    i = 0
    n = len(sorted_values)
    
    while i < n:
        # Start a new group
        group_indices = [i]
        for j in range(i + 1, n):
            if abs(sorted_values[j] - sorted_values[i]) <= abs_thresh:
                group_indices.append(j)
            else:
                break
        # Assign the same rank to all in the group
        for idx in group_indices:
            ranks[methods[idx]] = current_rank
        # Increment rank by 1 for the next group (compressed)
        current_rank += 1
        i += len(group_indices)
    
    return ranks

def compute_rank_counts_group(df, methods, decimals, abs_percent):
    """
    Compute rank counts and rank table for all datasets using group ranking.
    
    Parameters:
    - decimals: display precision for values (2 or 3)
    - abs_percent: fraction of the best value to allow ties (e.g., 0.01 = 1%)
    
    Returns:
    - rank_counts: dict[method][rank] = count
    - rank_table: pd.DataFrame long format
    """
    max_rank = len(methods)
    rank_counts = {m: {r: 0 for r in range(1, max_rank+1)} for m in methods}
    rank_table = []

    for _, row in df.iterrows():
        values = pd.Series({m: row[f"{m}{MISCLASS_SUFFIX}"] for m in methods})
        best_val = values.min()
        abs_thresh = best_val * abs_percent  # tie threshold
        ranks = group_ranking(values, abs_thresh)
        
        for m, r in ranks.items():
            rank_counts[m][r] += 1
            rank_table.append({
                "Dataset": row["Dataset"],
                "Method": m,
                "Value": round(values[m], decimals),
                "Rank": r,
                "Decimals": decimals,
                "AbsDiffThreshold": abs_percent
            })
    
    return rank_counts, pd.DataFrame(rank_table)

def group_ranking_after_decimals(values, decimals=2, abs_percent=0):
    """
    Compute compressed group ranks **after rounding** to specified decimals.
    
    Parameters:
    - values: pd.Series (method -> value), lower is better
    - decimals: int, number of decimal places to round before ranking
    - abs_percent: fraction of the best value to allow ties (e.g., 0.01 = 1%)
    
    Returns:
    - dict(method -> rank)
    """
    # Round values to specified decimals first
    rounded_vals = values.round(decimals)
    
    # Compute absolute difference threshold after rounding
    if abs_percent > 0:
        best_val = rounded_vals.min()
        abs_thresh = best_val * abs_percent
    else:
        abs_thresh = 0
    
    # Sort rounded values
    sorted_vals = rounded_vals.sort_values()
    methods = sorted_vals.index.tolist()
    sorted_values = sorted_vals.values
    
    ranks = {}
    current_rank = 1
    i = 0
    n = len(sorted_values)
    
    while i < n:
        # Group all methods tied with abs_thresh
        group_indices = [i]
        for j in range(i + 1, n):
            if abs(sorted_values[j] - sorted_values[i]) <= abs_thresh:
                group_indices.append(j)
            else:
                break
        # Assign same rank to all in the group
        for idx in group_indices:
            ranks[methods[idx]] = current_rank
        current_rank += 1  # next group = next integer
        i += len(group_indices)
    
    return ranks

def compute_rank_counts_group_decimals(df, methods, decimals=2, abs_percent=0.01):
    """
    Compute rank counts and table using **rounded values first**, then compressed group ranking.
    
    Returns:
    - rank_counts: dict[method][rank] = count
    - rank_table: long-form DataFrame
    """
    max_rank = len(methods)
    rank_counts = {m: {r:0 for r in range(1, max_rank+1)} for m in methods}
    rank_table = []
    
    for _, row in df.iterrows():
        values = pd.Series({m: row[f"{m}{MISCLASS_SUFFIX}"] for m in methods})
        ranks = group_ranking_after_decimals(values, decimals=decimals, abs_percent=abs_percent)
        
        for m, r in ranks.items():
            rank_counts[m][r] += 1
            rank_table.append({
                "Dataset": row["Dataset"],
                "Method": m,
                "Value": round(values[m], decimals),
                "Rank": r,
                "Decimals": decimals,
                "AbsDiffThreshold": abs_percent
            })
    
    return rank_counts, pd.DataFrame(rank_table)


#--------------------

plot_misclassification_lines(df, METHODS)
plot_runtime_lines(df, METHODS)

#--------------------

raw_pvals, corrected_pvals, significant = corrected_wilcoxon_pvalue_matrices(
    df,
    METHODS,
    MISCLASS_SUFFIX,
    alternative="two-sided",
    correction="holm"
)

raw_pvals.to_csv(f"{OUTPUT_DIR}/wilcoxon_raw_pvalues.csv")
corrected_pvals.to_csv(f"{OUTPUT_DIR}/wilcoxon_corrected_pvalues_holm.csv")
significant.to_csv(f"{OUTPUT_DIR}/wilcoxon_significance_holm.csv")

# pval_matrix_raw pval_matrix_corrected, sig_matrix = wilcoxon_corrected_matrix(df, METHODS, MISCLASS_SUFFIX)
# pval_matrix_corrected, sig_matrix = bonferroni_correct_pval_matrix(pval_matrix_raw, alpha=0.05)

# pval_matrix_corrected.to_csv(f"{OUTPUT_DIR}/pvalues_corrected_bonferroni.csv")
# sig_matrix.to_csv(f"{OUTPUT_DIR}/pvalues_significant_mask.csv")
# pval_matrix_raw.to_csv(f"{OUTPUT_DIR}/pvalues_raw.csv")

#--------------------

decimals_list = [2,3]
abs_diff_list = [0.00, 0.01,0.02,0.05,0.10]

all_rank_tables = []

for dec, abs_perc in product(decimals_list, abs_diff_list):
    rank_counts, rank_table = compute_rank_counts_group_decimals(df, METHODS, dec, abs_perc)
    all_rank_tables.append(rank_table)
    
    title = f"Rank distribution (Decimals={dec}, AbsDiff={int(abs_perc*100)}%)"
    filename = f"{OUTPUT_DIR}/rank_distribution_grouped_dec{dec}_abs{int(abs_perc*100)}"
    plot_rank_distribution_with_ties(rank_table, METHODS, title, filename)

# Save combined rank table
combined_rank_table = pd.concat(all_rank_tables, ignore_index=True)
combined_rank_table.to_csv(f"{OUTPUT_DIR}/rank_table_compressed_absdiff.csv", index=False)
print("Rank table saved:", f"{OUTPUT_DIR}/rank_table_compressed_absdiff.csv")

# rank_counts_2dp = compute_rank_counts_absolute_precision(
#     df, METHODS, MISCLASS_SUFFIX, decimals=2
# )

# rank_counts_3dp = compute_rank_counts_absolute_precision(
#     df, METHODS, MISCLASS_SUFFIX, decimals=3
# )
# plot_rank_distribution(rank_counts_2dp, title="Competition Ranking (2 decimal precision)", 
#                        save_path=f"{OUTPUT_DIR}/rank_distribution_2dp")
# plot_rank_distribution(rank_counts_3dp, title="Competition Ranking (3 decimal precision)", 
#                        save_path=f"{OUTPUT_DIR}/rank_distribution_3dp")

# rank_table_2dp = generate_absolute_rank_table(
#     df,
#     METHODS,
#     MISCLASS_SUFFIX,
#     decimals=2,
#     metric_name="Misclassification (2dp)"
# )

# rank_table_3dp = generate_absolute_rank_table(
#     df,
#     METHODS,
#     MISCLASS_SUFFIX,
#     decimals=3,
#     metric_name="Misclassification (3dp)"
# )

# rank_table_2dp.to_csv(f"{OUTPUT_DIR}/rank_table_absolute_2dp.csv", index=False)
# rank_table_3dp.to_csv(f"{OUTPUT_DIR}/rank_table_absolute_3dp.csv", index=False)

# Generate ranking tables
# ranking_table_2dp = generate_ranking_table(df, METHODS, decimals=2)
# ranking_table_3dp = generate_ranking_table(df, METHODS, decimals=3)

# # Save to CSV
# ranking_table_2dp.to_csv(f"{OUTPUT_DIR}/ranking_table_2dp.csv", index=False)
# ranking_table_3dp.to_csv(f"{OUTPUT_DIR}/ranking_table_3dp.csv", index=False)

#--------------------
best_methods_per_dataset = []
for idx, row in df.iterrows():
    best_dl = min(METHODS_DL, key=lambda m: row[f"{m}{MISCLASS_SUFFIX}"])
    best_graph = min(METHODS_GRAPH, key=lambda m: row[f"{m}{MISCLASS_SUFFIX}"])
    best_methods_per_dataset.append({
        "Dataset": row["Dataset"],
        "Dynamic graphlets + LR": row[f"{FIXED_METHOD}{MISCLASS_SUFFIX}"],
        "Best of regular deep learning": row[f"{best_dl}{MISCLASS_SUFFIX}"],
        "Best of graph-based deep learning": row[f"{best_graph}{MISCLASS_SUFFIX}"],
        "Runtime Dynamic graphlets + LR": row[f"{FIXED_METHOD}{RUNTIME_SUFFIX}"],
        "Runtime Best of regular deep learning": row[f"{best_dl}{RUNTIME_SUFFIX}"],
        "Runtime Best of graph-based deep learning": row[f"{best_graph}{RUNTIME_SUFFIX}"],
    })

df_selected = pd.DataFrame(best_methods_per_dataset)

selected_methods_misclass = ["Dynamic graphlets + LR", 
                             "Best of regular deep learning", 
                             "Best of graph-based deep learning"]

selected_methods_runtime = ["Runtime Dynamic graphlets + LR", 
                            "Runtime Best of regular deep learning", 
                            "Runtime Best of graph-based deep learning"]

# ===========================
# Plot misclassification line plot (vertical x-axis labels)
# ===========================
x = np.arange(len(datasets))
fig, ax = plt.subplots(figsize=(14,5))
for i, m in enumerate(selected_methods_misclass):
    ax.plot(x, df_selected[m], marker="o", markersize=6, linewidth=2, color=COLOR_CYCLE[i], label=m)

ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=90, ha="center")  # vertical labels
ax.set_ylabel("Misclassification rate")
ax.set_xlabel("Dataset")
ax.set_title("Misclassification rates for selected methods")
ax.legend(ncol=1, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/misclassification_selected_methods.pdf")
fig.savefig(f"{OUTPUT_DIR}/misclassification_selected_methods.svg")
plt.show()

# ===========================
# Plot runtime line plot (vertical x-axis labels)
# ===========================
fig, ax = plt.subplots(figsize=(14,5))
for i, m in enumerate(selected_methods_runtime):
    ax.plot(x, df_selected[m], marker="o", markersize=6, linewidth=2, color=COLOR_CYCLE[i], label=m.replace("Runtime ", ""))

ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=90, ha="center")  # vertical labels
ax.set_ylabel("Runtime (min)")
ax.set_xlabel("Dataset")
ax.set_title("Runtime for selected methods")
ax.legend(ncol=1, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/runtime_selected_methods.pdf")
fig.savefig(f"{OUTPUT_DIR}/runtime_selected_methods.svg")
plt.show()

# ===========================
# Prepare for ranking plots
# ===========================
# Convert to long-form DataFrame for ranking
rank_table = []
decimals = 3
abs_percent = 0.01  # 1% threshold for ties

for idx, row in df_selected.iterrows():
    # Prepare values for ranking
    values = pd.Series({
        "Dynamic graphlets + LR": row["Dynamic graphlets + LR"],
        "Best of regular deep learning": row["Best of regular deep learning"],
        "Best of graph-based deep learning": row["Best of graph-based deep learning"]
    })
    # Round first
    values_rounded = values.round(decimals)
    # Compressed rank by group
    best_val = values_rounded.min()
    abs_thresh = best_val * abs_percent
    sorted_vals = values_rounded.sort_values()
    methods = sorted_vals.index.tolist()
    
    ranks = {}
    current_rank = 1
    i = 0
    while i < len(sorted_vals):
        tie_indices = [i]
        for j in range(i+1, len(sorted_vals)):
            if abs(sorted_vals[j] - sorted_vals[i]) <= abs_thresh:
                tie_indices.append(j)
            else:
                break
        for idx2 in tie_indices:
            ranks[methods[idx2]] = current_rank
        current_rank += 1
        i += len(tie_indices)
    
    for m in methods:
        rank_table.append({
            "Dataset": row["Dataset"],
            "Method": m,
            "Value": values_rounded[m],
            "Rank": ranks[m],
            "Decimals": decimals,
            "AbsDiffThreshold": abs_percent
        })

df_rank = pd.DataFrame(rank_table)

# ===========================
# Plot rank distribution with ties
# ===========================
def plot_rank_distribution_with_ties(rank_table, methods, title, filename):
    max_rank = rank_table['Rank'].max()
    data_total = np.zeros((len(methods), max_rank))
    data_ties  = np.zeros((len(methods), max_rank))
    
    for i, m in enumerate(methods):
        df_m = rank_table[rank_table['Method'] == m]
        n_datasets = df_m['Dataset'].nunique()
        for r in range(1, max_rank+1):
            df_r = df_m[df_m['Rank'] == r]
            data_total[i, r-1] = len(df_r)/n_datasets
            tie_count = 0
            for ds in df_r['Dataset'].unique():
                df_ds = rank_table[(rank_table['Dataset']==ds) & (rank_table['Rank']==r)]
                if len(df_ds) > 1:
                    tie_count += 1
            data_ties[i, r-1] = tie_count/n_datasets
    
    fig, ax = plt.subplots(figsize=(10,5))
    bottom = np.zeros(len(methods))
    
    for r in range(max_rank):
        ax.bar(methods, data_total[:,r], bottom=bottom, color=COLOR_CYCLE[r%len(COLOR_CYCLE)], edgecolor='black', linewidth=0.4)
        ax.bar(methods, data_ties[:,r], bottom=bottom, color='k', alpha=0.3, edgecolor=None)
        bottom += data_total[:,r]
    
    ax.set_ylabel("Proportion of datasets")
    ax.set_ylim(0,1.0)
    ax.set_title(title)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='gray', alpha=0.3, label='Tie fraction')], loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(filename + ".pdf")
    fig.savefig(filename + ".svg")
    plt.show()

plot_rank_distribution_with_ties(df_rank, selected_methods_misclass,
                                 "Rank distribution for selected methods",
                                 f"{OUTPUT_DIR}/rank_distribution_selected_methods")

# ===========================
# Helper: group ranking after rounding
# ===========================
def group_ranking_after_decimals(values, decimals=2, abs_percent=0):
    rounded_vals = values.round(decimals)
    if abs_percent > 0:
        best_val = rounded_vals.min()
        abs_thresh = best_val * abs_percent
    else:
        abs_thresh = 0

    sorted_vals = rounded_vals.sort_values()
    methods = sorted_vals.index.tolist()
    sorted_values = sorted_vals.values

    ranks = {}
    current_rank = 1
    i = 0
    n = len(sorted_values)

    while i < n:
        group_indices = [i]
        for j in range(i+1, n):
            if abs(sorted_values[j] - sorted_values[i]) <= abs_thresh:
                group_indices.append(j)
            else:
                break
        for idx2 in group_indices:
            ranks[methods[idx2]] = current_rank
        current_rank += 1
        i += len(group_indices)

    return ranks

# ===========================
# Compute rank table for given decimals and abs_diff
# ===========================
def compute_rank_counts_group_decimals(df_selected, methods, decimals, abs_percent):
    max_rank = len(methods)
    rank_counts = {m: {r:0 for r in range(1,max_rank+1)} for m in methods}
    rank_table = []

    for _, row in df_selected.iterrows():
        values = pd.Series({m: row[m] for m in methods})
        ranks = group_ranking_after_decimals(values, decimals, abs_percent)

        for m, r in ranks.items():
            rank_counts[m][r] += 1
            rank_table.append({
                "Dataset": row["Dataset"],
                "Method": m,
                "Value": round(values[m], decimals),
                "Rank": r,
                "Decimals": decimals,
                "AbsDiffThreshold": abs_percent
            })

    return rank_counts, pd.DataFrame(rank_table)

# ===========================
# Wrap labels helper
# ===========================
def wrap_labels(labels, max_len=20):
    new_labels = []
    for lab in labels:
        if len(lab) > max_len:
            chunks = [lab[i:i+max_len] for i in range(0, len(lab), max_len)]
            new_labels.append("\n".join(chunks))
        else:
            new_labels.append(lab)
    return new_labels

# ===========================
# Rank distribution plot with tie shading
# ===========================
def plot_rank_distribution_with_ties(rank_table, methods, title, filename, max_label_len=20):
    max_rank = rank_table['Rank'].max()
    data_total = np.zeros((len(methods), max_rank))
    data_ties  = np.zeros((len(methods), max_rank))

    for i, m in enumerate(methods):
        df_m = rank_table[rank_table['Method'] == m]
        n_datasets = df_m['Dataset'].nunique()
        for r in range(1, max_rank+1):
            df_r = df_m[df_m['Rank']==r]
            data_total[i,r-1] = len(df_r)/n_datasets
            tie_count = 0
            for ds in df_r['Dataset'].unique():
                df_ds = rank_table[(rank_table['Dataset']==ds) & (rank_table['Rank']==r)]
                if len(df_ds) > 1:
                    tie_count += 1
            data_ties[i,r-1] = tie_count/n_datasets

    fig, ax = plt.subplots(figsize=(max(12,len(methods)*1.5),5))
    bottom = np.zeros(len(methods))

    for r in range(max_rank):
        ax.bar(methods, data_total[:,r], bottom=bottom, color=COLOR_CYCLE[r%len(COLOR_CYCLE)],
               edgecolor='black', linewidth=0.4)
        ax.bar(methods, data_ties[:,r], bottom=bottom, color='k', alpha=0.3, edgecolor=None)
        bottom += data_total[:,r]

    ax.set_ylabel("Proportion of datasets")
    ax.set_ylim(0,1.0)
    ax.set_title(title)
    wrapped_labels = wrap_labels(methods, max_label_len)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(wrapped_labels, rotation=0, ha="center")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='gray', alpha=0.3, label='Tie fraction')],
              loc='upper right', frameon=False)

    fig.tight_layout()
    fig.savefig(filename + ".pdf")
    fig.savefig(filename + ".svg")
    plt.show()

# ===========================
# Compute and plot all combinations of decimals and abs_diff
# ===========================
decimals_list = [2,3]
abs_diff_list = [0.00, 0.01, 0.02, 0.05, 0.10]

all_rank_tables = []

for dec, abs_perc in product(decimals_list, abs_diff_list):
    rank_counts, rank_table = compute_rank_counts_group_decimals(df_selected, selected_methods_misclass, dec, abs_perc)
    all_rank_tables.append(rank_table)

    title = f"Rank distribution (Decimals={dec}, AbsDiff={int(abs_perc*100)}%)"
    filename = f"{OUTPUT_DIR}/rank_distribution_grouped_dec{dec}_abs{int(abs_perc*100)}"
    plot_rank_distribution_with_ties(rank_table, selected_methods_misclass, title, filename)

# Save combined rank table
combined_rank_table = pd.concat(all_rank_tables, ignore_index=True)
combined_rank_table.to_csv(f"{OUTPUT_DIR}/rank_table_compressed_absdiff_selected_methods.csv", index=False)
print("Rank table saved:", f"{OUTPUT_DIR}/rank_table_compressed_absdiff_selected_methods.csv")

# df_selected = select_best_methods_per_dataset(df)
# plot_selected_misclassification(df_selected)

# df_selected = select_best_methods(df)
# plot_selected_misclassification_lines(df_selected)

# # 1. Compute ranks for selected methods
# rank_counts_selected = compute_ranks_for_selected(df_selected)

# # 2. Compute rank counts
# rank_counts_2dp = compute_ranks_selected(df_selected, decimals=2)
# rank_counts_3dp = compute_ranks_selected(df_selected, decimals=3)
# rank_counts_4dp = compute_ranks_selected(df_selected, decimals=4)

# # 3. Generate rank tables for CSV
# rank_table_2dp = generate_rank_table_selected(df_selected, decimals=2)
# rank_table_3dp = generate_rank_table_selected(df_selected, decimals=3)

# rank_table_2dp.to_csv(f"{OUTPUT_DIR}/rank_table_selected_2dp.csv", index=False)
# rank_table_3dp.to_csv(f"{OUTPUT_DIR}/rank_table_selected_3dp.csv", index=False)

# df_selected.to_csv(f"{OUTPUT_DIR}/misclassification_selected.csv", index=False)

# # 5. Plot rank distributions
# plot_selected_rank_distribution(rank_counts_2dp)
# plot_selected_rank_distribution(rank_counts_3dp)
# plot_selected_rank_distribution(rank_counts_4dp)