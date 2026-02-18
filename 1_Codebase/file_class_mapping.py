#!/usr/bin/env python3

infile = "labels/cath-1.20.5.txt"
outfile = "dataset-3.txt"

with open(infile, "r") as fin, open(outfile, "w") as fout:
    for line in fin:
        # skip empty lines unchanged
        if not line.strip():
            fout.write(line)
            continue

        cols = line.rstrip("\n").split("\t")
        # append suffix to column 2
        cols[1] = cols[1] + "_dcgdv_6_4_1.txt"

        # swap col1 and col2 when writing
        new_cols = [cols[1], cols[0]]
        fout.write("\t".join(new_cols) + "\n")
