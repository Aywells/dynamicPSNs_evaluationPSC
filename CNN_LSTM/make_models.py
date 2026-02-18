#!/usr/bin/env python3
from pathlib import Path
import re

# This Python script takes the model template as input and generates all 72 model files
TEMPLATE_PATH = Path("leaky_model_template_cv.py")  # <-- your template file
OUT_DIR = Path(".")                                # or Path("generated_models")
N_MAX = 71

def replace_assignment(src: str, var: str, new_value: str) -> str:
    """
    Replace a line like: var = "something"
    Keeps formatting and replaces only the RHS string.
    """
    pattern = rf'^(\s*{re.escape(var)}\s*=\s*)(".*?"|\'.*?\')\s*$'
    repl = rf'\1"{new_value}"'
    out, n = re.subn(pattern, repl, src, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f'Expected exactly 1 assignment for {var}, found {n}')
    return out

def main():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(N_MAX + 1):
        folder = f"dataset-{i}"
        dfile = f"dataset-{i}.txt"

        content = template
        content = replace_assignment(content, "dataset_folder", folder)
        content = replace_assignment(content, "dataset_file", dfile)

        out_path = OUT_DIR / f"leaky_model{i}_cv.py"
        out_path.write_text(content, encoding="utf-8")
        print("Wrote", out_path)

if __name__ == "__main__":
    main()
