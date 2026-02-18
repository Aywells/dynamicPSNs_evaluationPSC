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



    # 4. Build labels-array aligned with sorted matrix_files
    labeled = []
    unlabeled = []

    for f in matrix_files:
        fname = f.name
        stub_name = Path(fname).stem
        parts = stub_name.split("_dcgdv_")
        f_name = parts[0]
        if f_name in file_to_class:
            labeled.append(file_to_class[fname])
        else:
            unlabeled.append(fname)

#    if unlabeled:
#        print("\nWARNING: The following files have NO class label:")
#        for u in unlabeled:
#            print("  ", u)
#
#        raise RuntimeError("Some matrix files are unlabeled. "
#                           "Add them to a class file or remove them before training.")

    y = np.array(labeled, dtype=int)

    # 5. Show a small summary and save
    counts = Counter(y)
    print("\nLabel counts per class_id:")
    for cid, cnt in sorted(counts.items()):
        cname = class_names[cid]
        print(f"  class_id {cid:2d} ({cname:>15}): {cnt} samples")
    np.savetxt(DATA_DIR/"labels.txt", labeled, fmt="%d")
    np.save(OUT_PATH, y)
    print(f"\nSaved labels to {OUT_PATH}")
    print(f"labels shape: {y.shape}, num_classes: {len(class_names)}")


if __name__ == "__main__":
    main()
