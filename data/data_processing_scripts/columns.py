import os
import numpy as np
import hashlib
from tqdm import tqdm

INPUT_ROOT = "feature-matrix"     # folder containing the directories with txt files
OUTPUT_ROOT = "feature-matrix_no0" # folder where new directories will be created

def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

all_txt_paths = []

for root, dirs, files in os.walk(INPUT_ROOT):
    for f in files:
        if f.endswith(".txt"):
            all_txt_paths.append(os.path.join(root, f))

if not all_txt_paths:
    raise ValueError("No .txt files found.")

print("Total files found:", len(all_txt_paths))

print("Detecting duplicate files...")

unique_files = {}
for path in tqdm(all_txt_paths):

    h = file_hash(path)

    if h not in unique_files:
        unique_files[h] = path

unique_paths = list(unique_files.values())

print("Unique files:", len(unique_paths))

print("Scanning files for zero columns...")

first = True

for path in tqdm(unique_paths):

    data = np.genfromtxt(path, dtype=float, filling_values=0)

    # remove first column
    data = data[:, 1:]

    data = np.nan_to_num(data)

    if first:
        n_cols = data.shape[1]
        global_zero = np.ones(n_cols, dtype=bool)
        first = False

    col_zero = np.all(data == 0, axis=0)

    global_zero &= col_zero


keep_columns = np.where(~global_zero)[0]

print("Columns removed:", np.sum(global_zero))

print("Processing files...")

for root, dirs, files in os.walk(INPUT_ROOT):

    rel_dir = os.path.relpath(root, INPUT_ROOT)
    output_dir = os.path.join(OUTPUT_ROOT, rel_dir)

    os.makedirs(output_dir, exist_ok=True)

    txt_files = [f for f in files if f.endswith(".txt")]

    for file in tqdm(txt_files, leave=False):

        input_file = os.path.join(root, file)
        output_file = os.path.join(output_dir, file)

        data = np.genfromtxt(input_file, dtype=float, filling_values=0)

        data = data[:, 1:]
        data = np.nan_to_num(data)

        filtered = data[:, keep_columns]

        np.savetxt(output_file, filtered, delimiter="\t", fmt="%g")


print("Done.")