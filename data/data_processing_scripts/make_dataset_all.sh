#!/usr/bin/env bash

# This Shell script takes dataset class file as input and uses it to extract 
# the corresponding dataset files from the final_data folder for all 72 datasets

set -euo pipefail

srcdir="final_data"

for x in $(seq 0 71); do
    infile="all_datasets/dataset-${x}.txt"
    outdir="all_datasets/dataset-${x}"

    # skip if infile doesn't exist
    [ -f "$infile" ] || { echo "Skipping (missing): $infile"; continue; }

    mkdir -p "$outdir"

    while read -r col1 _; do
        [ -z "$col1" ] && continue

        # exact name match (no extension)
        if [ -e "$srcdir/$col1" ]; then
            cp "$srcdir/$col1" "$outdir"/
        fi

        # If your source files may have extensions, use this instead:
        # for f in "$srcdir/$col1"*; do
        #   [ -e "$f" ] && cp "$f" "$outdir"/
        # done
    done < "$infile"

    echo "Done: $infile -> $outdir"
done
