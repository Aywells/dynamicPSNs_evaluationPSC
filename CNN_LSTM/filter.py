from pathlib import Path
import numpy as np

# ====== CONFIG ======
data_folder = Path("dcgdvFiles/")        # folder with your matrix files
indices_file = data_folder/"zerocolumns.txt"    # file containing column indices
output_folder = data_folder/"dcgdv_filtered"  # output folder

output_folder.mkdir(exist_ok=True)

# ====== 1. Read column indices into a list ======
#drop_indices = []

#with indices_file.open() as f:
#    for line in f:
#        line = line.strip()
#        if not line or line.startswith("#"):
#            continue
#        # allow space- or comma-separated indices on each line
#        parts = line.replace(",", " ").split()
#        drop_indices.extend(int(p) for p in parts)

# If indices in the file are 1-based and you need 0-based, uncomment this:
# drop_indices = [i - 1 for i in drop_indices]


import re

# ====== 1. Read column indices into a list ======
drop_indices = []

with indices_file.open() as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # split on ',' or ']' characters
        parts = re.split(r'[,\]]', line)

        for p in parts:
            p = p.strip()
            if not p:
                continue
            # only try to convert tokens that are actually integers
            try:
                drop_indices.append(int(p))
            except ValueError:
                # Optional: uncomment to debug weird tokens
                # print("Skipping non-integer token:", repr(p))
                continue



# Remove duplicates and sort
drop_indices = sorted(set(drop_indices))
print("Columns to drop (0-based):", drop_indices)

# ====== 2 & 3. For each file, drop those columns and save to new folder ======
for file in sorted(data_folder.glob("*.txt")):
    # Skip the indices file itself and anything already in the output folder
    if file == indices_file:
        continue

    print(f"Processing {file.name}...")

    # Read matrix from file (whitespace-separated)
    A = np.loadtxt(file)

    if A.ndim == 1:
        # Single row case: reshape to (1, n_cols)
        A = A.reshape(1, -1)

    n_rows, n_cols = A.shape

    # Safety check: warn if any index is out of range
    bad = [i for i in drop_indices if i >= n_cols]
    if bad:
        print(f"  WARNING: indices {bad} >= number of columns ({n_cols}) in {file.name}")

    # Build a boolean mask of columns to keep
    keep_mask = np.ones(n_cols, dtype=bool)
    for i in drop_indices:
        if i < n_cols:
            keep_mask[i] = False

    A_filtered = A[:, keep_mask]

    # Save filtered matrix with the same filename into dcgdv_filtered/
    out_path = output_folder / file.name
    np.savetxt(out_path, A_filtered, fmt="%.0f")  # or fmt="%d" if all ints

    print(f"  Original shape: {A.shape}, filtered shape: {A_filtered.shape}")
