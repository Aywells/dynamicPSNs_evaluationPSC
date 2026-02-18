import re
import glob
import csv
from pathlib import Path

# --------- CONFIG ----------
PATTERN = "CVMODELS/logs/*_model*.log"   # change to "*.txt" or "/path/to/files/*.txt"
OUT_CSV = "parsed_metrics.csv"
# --------------------------

# Regex patterns (tolerant to spacing/case/colon)
PATTERNS = {
    #"aggregate_misclassification_rate": re.compile(
    #    r"Aggregate\s+misclassification\s+rate\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
    #),
    "aggregate_misclassification_rate": re.compile(
        r"Aggregate\s+misclassification\s+rate(?:\s*\([^)]*\))?\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
    ),


    "num_samples": re.compile(
        r"Number\s+of\s+samples\s*[:=]\s*(\d+)", re.IGNORECASE
    ),
    "num_classes": re.compile(
        r"Number\s+of\s+classes\s*[:=]\s*(\d+)", re.IGNORECASE
    ),
}
# Matches: rcv_model.log, rcv_model6.log, rcv_model_6.log, rcv_model-6.log, rcv_model123.log


def extract_metrics(text: str) -> dict:
    out = {k: None for k in PATTERNS.keys()}
    for key, rx in PATTERNS.items():
        m = rx.search(text)
        if m:
            val = m.group(1)
            out[key] = float(val) if "." in val else int(val)
    return out

rows = []

for fp in sorted(glob.glob(PATTERN)):
    p = Path(fp)
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        rows.append({"file": str(p), "error": str(e)})
        continue

    metrics = extract_metrics(txt)
    row = {"file": p.name, **metrics}
    rows.append(row)

# Write CSV
fieldnames = ["file",
              "aggregate_misclassification_rate",
              "num_samples",
              "num_classes"]

fieldnames = ["file",
              "amr",
              "fta",
              "ns",
              "nc"]

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fieldnames})

# Print a quick summary
print(f"Wrote: {OUT_CSV}")
for r in rows:
    print(r)
