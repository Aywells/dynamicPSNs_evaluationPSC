#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

AGG_HEADER_RE = re.compile(r"^===\s*Aggregate\s+confusion\s+matrix\s*\(sum\s+over\s+folds\)\s*===", re.I)

def parse_aggregate_confusion_matrix(text: str) -> Optional[List[str]]:
    """
    Returns the matrix lines (e.g., ["[[30 10]", " [ 2 78]]"])
    or None if not found.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if AGG_HEADER_RE.match(line.strip()):
            # Collect subsequent non-empty lines that look like matrix rows.
            out = []
            for j in range(i + 1, len(lines)):
                s = lines[j].rstrip()
                if not s.strip():
                    if out:
                        break
                    continue

                # matrix rows usually start with '[' or ' ['
                if s.lstrip().startswith("["):
                    out.append(s)
                else:
                    # stop once we leave the matrix block
                    if out:
                        break
            return out if out else None
    return None

def read_file(path: Path) -> str:
    return path.read_text(errors="ignore")

def expand_inputs(args: List[str]) -> List[Path]:
    """
    Accepts:
      - explicit filenames
      - globs like "*.log" or "logs/*.log"
    """
    paths: List[Path] = []
    for a in args:
        # expand glob patterns
        matches = sorted(Path().glob(a)) if any(ch in a for ch in "*?[]") else [Path(a)]
        for m in matches:
            if m.is_file():
                paths.append(m)
    # dedupe preserving order
    seen = set()
    uniq = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  extract_agg_cm.py file1.log file2.log ...")
        print('  extract_agg_cm.py "rcv_model*.log" "lrcv_model*.log" "deep_model*.log" "paper_model*.log" ')
        sys.exit(2)

    files = expand_inputs(sys.argv[1:])
    if not files:
        raise SystemExit("No input files found (check your paths/globs).")

    out_path = Path("aggregate_confusion_matrices.txt")

    with out_path.open("w") as out:
        for fp in files:
            txt = read_file(fp)
            cm_lines = parse_aggregate_confusion_matrix(txt)

            out.write(f"FILE: {fp}\n")
            if cm_lines:
                out.write("AGGREGATE CONFUSION MATRIX:\n")
                for line in cm_lines:
                    out.write(line + "\n")
            else:
                out.write("AGGREGATE CONFUSION MATRIX: NOT FOUND\n")
            out.write("\n" + ("-" * 60) + "\n\n")

            # also print to stdout (as requested)
            print(f"{fp.name}: {'FOUND' if cm_lines else 'NOT FOUND'}")

    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    main()
