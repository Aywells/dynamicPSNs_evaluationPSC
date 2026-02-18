#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline to build labels for dynamic graphlet matrices.

Assumptions
----------
- DATA_DIR contains:
    - labels/        : each file = one class, each line = filename (or path)
    - final_data/    : dynamic graphlet matrices, e.g. 8fab_B_122_222_dcgdv_6_4_1.txt

Steps
-----
1. Read LABEL_DIR:
     * Each file in LABEL_DIR corresponds to one class.
     * Each line in a class file is a filename (or full path) belonging to that class.
   -> Build a mapping: file_to_class_map.txt with lines
        <fname>\t<class_id>

   First assignment wins: if a file appears in multiple class files,
   the first class is kept; subsequent ones only trigger a WARNING.

2. Use file_to_class_map.txt to label files in final_data/:
     * For each matrix file:
         - try to match by:
             - full filename           (e.g. 8fab_B_122_222_dcgdv_6_4_1.txt)
             - stem                    (e.g. 8fab_B_122_222_dcgdv_6_4_1)
             - stub before "_dcgdv"    (e.g. 8fab_B_122_222)
     * Save:
         - labels_final.npy : NumPy array of class IDs
         - final_files.txt  : list of filenames in the same order
"""

from pathlib import Path
import numpy as np

# ======== CONFIG ========
DATA_DIR    = Path("/users/fgatsi/dataset")
LABEL_DIR   = DATA_DIR / "labels"       # class definition files
MATRIX_DIR  = DATA_DIR / "final_data"   # where your dynamic graphlet matrices live
FILE_PATTERN = "*.txt"

MAP_PATH    = DATA_DIR / "file_to_class_map.txt"
OUT_LABELS  = DATA_DIR / "labels_final.npy"
OUT_FILES   = DATA_DIR / "final_files.txt"
# ========================


# --------------------------------------------------------------------
# 1) Build file_to_class_map.txt from LABEL_DIR
# --------------------------------------------------------------------
def build_file_to_class_map():
    """
    Read class files from LABEL_DIR and build a filename -> class_id mapping.

    Returns
    -------
    file_to_class : dict[str, int]
        Maps (basename) filenames to integer class IDs.
    class_to_id : dict[str, int]
        Maps class name (stem of class file) to integer class ID.
    """
    # List all class files
    class_files = sorted(LABEL_DIR.glob("*"))
    if not class_files:
        raise RuntimeError(f"No class files found in {LABEL_DIR}")

    # Class name = stem of the class file, e.g. "alpha_beta.txt" -> "alpha_beta"
    class_names = [cf.stem for cf in class_files]
    class_to_id = {name: i for i, name in enumerate(class_names)}

    print("Classes and IDs:")
    for name, cid in class_to_id.items():
        print(f"  {cid}: {name}")

    file_to_class: dict[str, int] = {}

    # Build mapping, keeping the first assignment and warning on duplicates
    for cf in class_files:
        class_name = cf.stem
        cid = class_to_id[class_name]

        with cf.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Keep only the basename (in case the line is a path)
                fname = Path(line).name

                if fname in file_to_class:
                    # Already has a class → keep the original, just warn and skip
                    print(
                        f"WARNING: {fname} already assigned to class "
                        f"{file_to_class[fname]}, ignoring additional membership in {class_name}"
                    )
                    continue  # do NOT overwrite

                # First time we see this file → assign its class
                file_to_class[fname] = cid

    # Optionally show mapping with class names
    id_to_class = {v: k for k, v in class_to_id.items()}

    print("\n=== file_to_class contents ===")
    for fname, cid in sorted(file_to_class.items()):
        cname = id_to_class[cid]
        print(f"{fname}\t{cid}\t{cname}")

    # Write to file_to_class_map.txt in DATA_DIR
    with MAP_PATH.open("w") as out:
        for fname, cid in sorted(file_to_class.items()):
            out.write(f"{fname}\t{cid}\n")

    print(f"\nWrote mapping for {len(file_to_class)} files to {MAP_PATH}")

    return file_to_class, class_to_id


# --------------------------------------------------------------------
# 2) Load mapping and label final_data
# --------------------------------------------------------------------
def load_mapping(map_path: Path):
    """
    Read file_to_class_map.txt and build a dict that can be
    looked up by several key forms:
      - full name (with extension if present)
      - stem
      - stub before '_dcgdv' (if present)

    Supports lines like:
      name  cid
    or:
      class_name  name  cid
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

            if len(parts) == 2:
                # format: <name> <cid>
                raw_name, cid_str = parts
            else:
                # format: <class_name> <name> <cid>
                raw_name, cid_str = parts[-2], parts[-1]

            try:
                cid = int(cid_str)
            except ValueError:
                print("Skipping bad line (cannot parse class ID):", line)
                continue

            p = Path(raw_name)

            name = p.name                      # e.g. '4u83_A_1_226'
            stem = p.stem                      # same here
            stub = stem.split("_dcgdv")[0]     # if you ever have '_dcgdv' in names

            for key in {name, stem, stub}:
                if key and key not in mapping:
                    mapping[key] = cid

    return mapping



def build_labels_for_final_data(mapping: dict[str, int]):
    """
    Use a name->class_id mapping to label matrices in MATRIX_DIR.
    Saves:
      - OUT_LABELS (np.ndarray of class IDs)
      - OUT_FILES  (text file with filenames in the same order)
    """
    matrix_files = sorted(MATRIX_DIR.glob(FILE_PATTERN))
    if not matrix_files:
        raise RuntimeError(f"No matrix files found in {MATRIX_DIR} matching {FILE_PATTERN}")

    print(f"\nFound {len(matrix_files)} matrix files in {MATRIX_DIR}")

    labels = []
    used_files = []
    unlabeled = []

    for f in matrix_files:
        name = f.name                  # '8fab_B_122_222_dcgdv_6_4_1.txt'
        stem = f.stem                  # '8fab_B_122_222_dcgdv_6_4_1'
        stub = stem.split("_dcgdv")[0] # '8fab_B_122_222'

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
        print("\nWARNING: the following files in final_data have no label:")
        for u in unlabeled:
            print("  ", u)
        print(f"Only labeled files will be saved ({len(used_files)} / {len(matrix_files)}).")

    labels_arr = np.array(labels, dtype=np.int64)
    np.save(OUT_LABELS, labels_arr)

    with OUT_FILES.open("w") as f:
        for name in used_files:
            f.write(name + "\n")

    print(f"\nSaved {len(labels_arr)} labels to {OUT_LABELS}")
    print(f"Saved file order to {OUT_FILES}")
    if used_files:
        print("Example mapping:")
        print("  ", used_files[0], "-> class", labels_arr[0])


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------
def main():
    # Step 1: build file_to_class_map.txt from LABEL_DIR
    build_file_to_class_map()

    # Step 2: load mapping and label final_data
    mapping = load_mapping(MAP_PATH)
    build_labels_for_final_data(mapping)


if __name__ == "__main__":
    main()
