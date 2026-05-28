#!/usr/bin/env python3

import re
from pathlib import Path

# ---------- filename handling ----------

LOG_GLOBS = (
    "rcv_model*.log",
    "lrcv_model*.log",
    "paper_model*.log",
    "deep_model*.log",
)

# ---------- regex patterns for content ----------

PATTERNS = {
    "aggregate_misclassification_rate_cv": re.compile(
        r"Aggregate\s+misclassification\s+rate(?:\s*\(CV\))?\s*[:=]\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    ),
    "num_samples": re.compile(
        r"Number\s+of\s+samples\s*[:=]\s*(\d+)",
        re.IGNORECASE,
    ),
    "num_classes": re.compile(
        r"Number\s+of\s+classes\s*[:=]\s*(\d+)",
        re.IGNORECASE,
    ),
}

# ---------- helpers ----------

def extract_metrics(text: str) -> dict:
    """Extract metrics from log file text."""
    results = {}
    for key, pattern in PATTERNS.items():
        m = pattern.search(text)
        results[key] = float(m.group(1)) if m else None
    return results


def iter_log_files(folder: Path):
    """Yield all matching log files."""
    for glob in LOG_GLOBS:
        yield from folder.glob(glob)


# ---------- main ----------

def main(log_dir: str = "."):
    log_dir = Path(log_dir)

    print(
        f"{'file':30s}  "
        f"{'agg_miscls(CV)':>14s}  "
        f"{'samples':>8s}  "
        f"{'classes':>7s}"
    )
    print("-" * 70)

    for log_file in sorted(iter_log_files(log_dir)):
        text = log_file.read_text(errors="ignore")
        metrics = extract_metrics(text)

        print(
            f"{log_file.name:30s}  "
            f"{metrics['aggregate_misclassification_rate_cv']!s:>14s}  "
            f"{metrics['num_samples']!s:>8s}  "
            f"{metrics['num_classes']!s:>7s}"
        )


if __name__ == "__main__":
    main()
