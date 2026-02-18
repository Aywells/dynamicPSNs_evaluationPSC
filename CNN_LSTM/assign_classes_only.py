#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build labels.npy from a label folder where:
  - Each file in LABEL_DIR corresponds to one class.
  - Each class file contains a list of filenames (from dcgdvFiles)
    that belong to that class, one per line.

Result:
  - labels.npy saved in DATA_DIR
  - y[i] is the class ID for files_sorted[i] in dcgdvFiles
"""

from pathlib import Path
from collections import Counter
import numpy as np

# ======== CONFIG ========
DATA_DIR   = Path("/users/fgatsi/dataset")
MATRIX_DIR = DATA_DIR/"final_data"   # where your dynamic graphlet matrices live
LABEL_DIR  = DATA_DIR/"labels"       # or where the class files are
FILE_PATTERN = "*.txt"               # the file type
OUT_PATH   = DATA_DIR/"labels.npy"
# ========================

def main():
    # 1. List all matrix files (this order will define y[i])
    matrix_files = sorted(MATRIX_DIR.glob(FILE_PATTERN))
    if not matrix_files:
        raise RuntimeError(f"No matrix files found in {MATRIX_DIR} matching {FILE_PATTERN}")

    print(f"Found {len(matrix_files)} matrix files in {MATRIX_DIR}")

    # 2. List all class files
    class_files = sorted(LABEL_DIR.glob("*"))
    if not class_files:
        raise RuntimeError(f"No class files found in {LABEL_DIR}")

    # Class name = stem of the class file, e.g. "alpha_beta.txt" -> "alpha_beta"
    class_names = [cf.stem for cf in class_files]
    class_to_id = {name: i for i, name in enumerate(class_names)}

    print("Classes and IDs:")
    for name, cid in class_to_id.items():
        print(f"  {cid}: {name}")

    # 3. Build filename -> class_id mapping
    file_to_class = {}

    for cf in class_files:

        class_name = cf.stem
        cid = class_to_id[class_name]

        with cf.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Sometimes the lines might contain a path; keep only the basename
                fname = Path(line).name

                if fname in file_to_class:
                    # Already has a class → keep the original, just warn and skip
                    print(
                        f"WARNING: {fname} already assigned to class "
                        f"{file_to_class[fname]}, ignoring additional membership in {class_name}"
                    )
                    continue  # <-- DO NOT overwrite

                # First time we see this file → assign its class
                file_to_class[fname] = cid

    print("\n=== file_to_class contents ===")
    for fname, cid in sorted(file_to_class.items()):
        print(f"{fname}\t{cid}")

    id_to_class = {v: k for k, v in class_to_id.items()}

    print("\n=== file_to_class contents ===")
    for fname, cid in sorted(file_to_class.items()):
        cname = id_to_class[cid]
        print(f"{fname}\t{cid}\t{cname}")

    out_path = Path("file_to_class_map.txt")
    with out_path.open("w") as out:
        for fname, cid in sorted(file_to_class.items()):
            out.write(f"{fname}\t{cid}\n")

    print(f"Wrote mapping for {len(file_to_class)} files to {out_path}")

# ---- paths ----
MAP_PATH   = DATA_DIR/"file_to_class_map.txt"
FINAL_DIR  = DATA_DIR/"final_data"
OUT_LABELS = DATA_DIR/"labels_final.npy"
OUT_FILES  = DATA_DIR/"final_files.txt"

def load_mapping(map_path: Path):
    """
    Read file_to_class_map.txt and build a dict that can be
    looked up by several key forms:
      - full name (with extension if present)
      - stem
      - stub before '_dcgdv' (if present)
    """
    mapping = {}

    with map_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            raw_name = parts[0]          # e.g. '8fab_B_122_222' or '8fab_B_122_222_dcgdv_6_4_1.txt'
            cid      = int(parts[1])

            p = Path(raw_name)

            # Different key variants
            name  = p.name                      # '8fab_B_122_222_dcgdv_6_4_1.txt'
            stem  = p.stem                      # '8fab_B_122_222_dcgdv_6_4_1'
            stub  = stem.split("_dcgdv")[0]     # '8fab_B_122_222'  (if '_dcgdv' present)

            for key in {name, stem, stub}:
                if key and key not in mapping:
                    mapping[key] = cid

    return mapping


    mapping = load_mapping(MAP_PATH)

    matrix_files = sorted(FINAL_DIR.glob("*.txt"))  # adjust pattern if needed

    labels = []
    used_files = []
    unlabeled = []

    for f in matrix_files:
        name = f.name                       # '8fab_B_122_222_dcgdv_6_4_1.txt'
        stem = f.stem                       # '8fab_B_122_222_dcgdv_6_4_1'
        stub = stem.split("_dcgdv")[0]      # '8fab_B_122_222'

        cid = None
        for key in (name, stem, stub):
            if key in mapping:
                cid = mapping[key]
                break

        if cid is None:
            unlabeled.append(name)
            continue

        labels.append(cid)
        used_files.append(name)

    if unlabeled:
        print("WARNING: the following files in final_data have no label:")
        for u in unlabeled:
            print("  ", u)
        print(f"Only labeled files will be saved ({len(used_files)}).")

    labels = np.array(labels, dtype=np.int64)
    np.save(OUT_LABELS, labels)

    with OUT_FILES.open("w") as f:
        for name in used_files:
            f.write(name + "\n")

    print(f"Saved {len(labels)} labels to {OUT_LABELS}")
    print(f"Saved file order to {OUT_FILES}")
    print("Example:")
    if used_files:
        print("  ", used_files[0], "-> class", labels[0])

   
if __name__ == "__main__":
    main()
