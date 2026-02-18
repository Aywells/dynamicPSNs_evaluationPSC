import os
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

# -------------------------------
# Configuration
# -------------------------------
EXCEL_FILE = "results_PSC_DL (version 1).xlsx"
OUTPUT_DIR = "figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "misclassification_boxplot.png")

METHOD_COLS = [  # replace with your actual method columns
    "Dynamic graphlets + LR_agg_misclass",
    "Dynamic graphlets + regular deep learning (2,3)_agg_misclass",
    "Dynamic graphlets + regular deep learning (3,1,ReLu)_agg_misclass",
    "Dynamic graphlets + regular deep learning (3,1,leakyReLu)_agg_misclass",
    "Dynamic graphlets + regular deep learning (3,3)_agg_misclass",
    "Default features + DGCN_agg_misclass",
    "Dynamic graphlets + DGCN_agg_misclass",
    "Dynamic graphlets + SGCN_agg_misclass"
]

METHOD_LABELS = [
    "Dynamic graphlets + LR",
    "Dynamic graphlets + regular deep learning (2,3)",
    "Dynamic graphlets + regular deep learning (3,1,ReLu)",
    "Dynamic graphlets + regular deep learning (3,1,leakyReLu)",
    "Dynamic graphlets + regular deep learning (3,3)",
    "Default features + DGCN",
    "Dynamic graphlets + DGCN",
    "Dynamic graphlets + SGCN"
]

COLORS = ["red", "blue", "grey", "black", "lightgrey", "lightblue", "pink", "green"]
LABEL_WRAP = 25  # wrap after 25 characters

# -------------------------------
# Create output directory if not exists
# -------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Load data
# -------------------------------
df = pd.read_excel(EXCEL_FILE)
misclass_df = df[METHOD_COLS]

# -------------------------------
# Wrap x-axis labels
# -------------------------------
def wrap_label(label, width=25):
    return "\n".join(textwrap.wrap(label, width=width))

wrapped_labels = [wrap_label(label, LABEL_WRAP) for label in METHOD_LABELS]

# -------------------------------
# Plot boxplot
# -------------------------------
plt.figure(figsize=(12, 6))  # wider to fit wrapped labels

bp = plt.boxplot(
    misclass_df,
    patch_artist=True,
    labels=wrapped_labels,
    showfliers=True
)

# Color boxes
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)

plt.ylabel("Misclassification rate")
plt.xlabel("PSN type")
plt.title("Misclassification rates per method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# -------------------------------
# Save figure to file
# -------------------------------
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print(f"Boxplot saved to: {OUTPUT_FILE}")
