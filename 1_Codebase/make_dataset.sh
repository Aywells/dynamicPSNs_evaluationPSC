#!/usr/bin/env bash

# This Shell script takes dataset class file as input and uses it 
# to extract the corresponding files from the final_data folder.

infile="dataset-xx.txt"
outdir="dataset-xx"
srcdir="final_data"

# 1. create the output folder
mkdir -p "$outdir"

# 2. step through file1
while read -r col1 _; do
    # skip empty lines
    [ -z "$col1" ] && continue

    # find matching files in folder1 (exact name match)
    # if your files have extensions, use "$srcdir/$col1"* instead
    if [ -e "$srcdir/$col1" ]; then
        cp "$srcdir/$col1" "$outdir"/
    fi
done < "$infile"
