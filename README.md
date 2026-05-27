# *Traditional* machine learning vs. *deep* learning from dynamic graph representations of proteins’ 3D folds in the task of protein structure classification CODE
---
This repository describes the 72 datasets and contains the code used to run the regular deep learning (CNN+LSTM) and graph-based deep learning (GCN) method variants from our paper, "Traditional machine learning vs. deep learning from dynamic graph representations of proteins’ 3D folds in the task of protein structure classification", A. Wells, F. A. Gatsi, A. Striegel, and T. Milenković (2026), under review."


## Table of contents

- [1. Directories in the repository](Directories-in-the-repository)
- [2. Overall workflow](#2-overall-workflow)
- [3. Step I: Prepare the dataset annotation files](#3-step-I-prepare-the-dataset-annotation-files)
- [4. Step II: Download the required CIF files](#4-step-II-download-the-required-cif-files)
- [5. Step III: Generate dynamic PSNs](#5-step-III-generate-dynamic-psns)
- [6. Step IV: Generate dGDVMs](#6-step-IV-generate-dgdvms)
- [7. Step V: Organize the generated data for this repository](#7-step-V-organize-the-generated-data-for-this-repository)
- [8. CNN+LSTM method](#8-cnnlstm-methods)
- [9. SGCN and DGCN method](#9-sgcn-and-dgcn-methods)
- [10. Inputs and outputs by method](#10-inputs-and-outputs-by-method)
- [11. Results processing](#11-results-processing)

---

## 1. Directories in the repository
```text
repository_root/
|
|-- CNN_LSTM/
|-- datasets/
|-- examples/
|-- GCN/
|-- scripts/
```

### `CNN_LSTM/`

This directory contains the code for training the CNN+LSTM variants on dynamic graphlet degree vector matrices (dGDVMs). 

The CNN+LSTM code supports four method variants:

| Varient name as in the paper | Varient name as in `CNN_LSTM/` | Input | Main idea |
|---|---|---|---|
| Dynamic graphlets + (2 CNN, 3 LSTM) | `paper_model_template_cv.py` | non-zero dGDVM matrix for each protein | The "default" variant inspired by [H. Guo, et al. (2019)](https://arxiv.org/abs/1910.02594); uses 2 CNN layers followed by 3 LSTM layers with ReLU activation. |
| Dynamic graphlets +  (3 CNN, 3 LSTM) | `deep_model_template_cv.py` | non-zero dGDVM matrix for each protein | The variant that uses 3 CNN layers followed by 3 LSTM layers with ReLU activation. |
| Dynamic graphlets +  (3 CNN, 1 LSTM) | `relu_model_template_cv.py` | non-zero dGDVM matrix for each protein | The variant that uses 3 CNN layers followed by 1 LSTM layer with ReLU activation. |
| Dynamic graphlets +  (3 CNN, 1 LSTM) under LeakyReLU | `leaky_model_template_cv.py` | non-zero dGDVM matrix for each protein | The variant that uses 3 CNN layers followed by 1 LSTM layer with LeakyReLU activation. |


### `datasets/`

This directory contains the dataset annotation files used to define the PSC datasets.

For our paper, the datasets correspond to the 72 PSC datasets. Each dataset is represented by a `.txt` annotation file. The file name identifies the dataset. For example:

```text
datasets/
|
|-- cath-1.txt
|-- cath-1.10.txt
|-- cath-1.10.10.txt
|-- ...
|-- scop-*.txt
|-- astral-40.txt
|-- scop-25.txt
```

### `examples/`

This directory is intended for small example inputs, toy datasets, example annotation files, or demonstration files that can be used to test whether the data-generation and model-training steps are working.

Use the datasets in this directory when you want to run a small test before running all 72 datasets.

### `GCN/`

This directory contains the code for running graph convolutional network methods.

The GCN code supports two method variants:

| Method | Input | Main idea |
|---|---|---|
| `SGCN` | Final PSN snapshot only + (default or dGDVMs) | Static graph method; ignores temporal evolution. |
| `DGCN` | Dynamic PSN  + (default or dGDVMs) | Dynamic graph method; uses all snapshots and models temporal information. |

Note that the GCN workflow can use either default/random node features or dGDVM node features.

### `scripts/`

This directory contains general utility scripts for data generation, organization, and conversion.

---

## 2. Overall workflow

### Repository overview

At a high level, the repository supports the following pipeline:

```text
protein-domain annotation files
                        |
                        v
download corresponding CIF structure files
                        |
                        v
generate dynamic protein structure networks (dynamic PSNs)
                        |
                        v
compute dynamic graphlet degree vector matrices (dGDVMs)
                        |
        +-------------------------------+
        |                               |
        v                               v
CNN+LSTM models                  SGCN/DGCN models
trained on dGDVMs                trained on PSNs
        |                        (default or dGDVM features)
        |                               |
        v                               v
cross-validation outputs         cross-validation outputs
        |                               |
        v                               v
result files                     result files
```

The repository should be used in the following order:

1. **Prepare dataset annotation files.**
   - Use the provided 72 dataset files in `datasets/`, or create your own annotation file.

2. **Download the corresponding CIF files.**
   - Each sample in the annotation file refers to a protein structure/domain.
   - The corresponding `.cif` files must be downloaded before PSNs can be generated.

3. **Generate dynamic PSNs.**
   - Dynamic PSNs are generated from the CIF files.

4. **Generate dGDVMs.**
   - dGDVMs are computed from the dynamic PSNs.
   - These are required for CNN+LSTM models and optional as initialized features for GCN models.

5. **Organize the generated files.**
   - CNN+LSTM models need dataset-specific folders of dGDVM files and optional partition files.
   - GCN models need dataset-specific folders containing PSN snapshots, optional dGDVM files, and optional partition files.

6. **Run the models.**
   - Run CNN+LSTM models from `CNN_LSTM/`.
   - Run SGCN/DGCN models from `GCN/`.

7. **Process results.**
   - Collect aggregate misclassification rates, fold-level metrics, confusion matrices, and other outputs.

<strong style="color: blue;">For all of these steps, we go into more detail below...</strong>

---

## 3. Step I: Prepare the dataset annotation files

The dataset annotation file defines which protein domains are included in a PSC dataset and what class label each domain has.

### 3.1 Using the provided 72 datasets

The repository provides a `datasets/` directory containing annotation files for the 72 PSC benchmark datasets used in the study.

Each `.txt` file corresponds to one PSC dataset. The file name is used as the dataset name throughout the pipeline.

For example:

```text
datasets/cath-1.txt
datasets/cath-1.10.txt
datasets/astral-40.txt
```
In our paper, we use the same 72 datasets from [K. Newaz, et al. (2022)](https://doi.org/10.1002/prot.26349) ([Github link](https://github.com/KhaliqueN/DynamicPSN))

### 3.2 Creating a new dataset annotation file

A custom dataset annotation file should be a tab-separated `.txt` file with two columns:

```text
<class_label>    <protein_domain_name>
```

The first column is the structural class label. The second column is the protein-domain identifier.

The expected protein-domain identifier format is:

```text
CIFID_chainID_startID1_endID1+startID2_endID2+...+startIDn_endIDn
```

where:

- `CIFID` is the PDB/CIF identifier without the `.cif` extension.
- `chainID` is the protein chain identifier.
- `startID` and `endID` define the residue range for a domain segment.
- Multiple discontinuous segments are separated using `+`.

Example with one continuous segment:

```text
alpha    1aip_C_2_54
```

Example with two discontinuous segments:

```text
alpha    1fnn_A_1_17+192_275
```

If the protein domain spans the full chain, use the first and last residue numbers of the chain as the start and end positions.

For more details see [K. Newaz, et al. (2022)](https://doi.org/10.1002/prot.26349) ([Github link](https://github.com/KhaliqueN/DynamicPSN))

---

## 4. Step II: Download the required CIF files

After preparing the dataset annotation files, download every `.cif` file referenced by the protein-domain identifiers.

For example, the domain:

```text
1fnn_A_1_17+192_275
```

requires the file:

```text
1fnn.cif
```

A recommended location for downloaded CIF files is:

```text
repository_root/cif/
```

Before generating PSNs, confirm that all CIF IDs listed in the dataset annotation file have corresponding `.cif` files in the CIF directory.

For batch downloading of `.cif` files, see the following batch downloader from the Protein Data Bank (PDB) ([click here](https://www.rcsb.org/downloads))

---

## 5. Step III: Generate dynamic PSNs

In our paper, we follow the same instructions for generating dynamic PSNs from [K. Newaz, et al. (2022)](https://doi.org/10.1002/prot.26349) ([Github link](https://github.com/KhaliqueN/DynamicPSN)). Here, we summarize the steps for generating dynamic PSN:

The software from [K. Newaz, et al. (2022)](https://doi.org/10.1002/prot.26349) can be run in different modes:

| Mode | Output |
|---|---|
| `Mode 1` | Generate dynamic PSNs only. |
| `Mode 2` | Generate dynamic PSNs and dynamic graphlets. |
| `Mode 3` | Generate dynamic PSNs, dynamic graphlets, and run the original LR-based PSC workflow. |

For our study, the most important outputs are:

1. **Dynamic PSN snapshot files**, used by SGCN/DGCN.
2. **dGDVMs**, used by CNN+LSTM and optionally by SGCN/DGCN.

Therefore, for our purposes, for each dataset annotation file, we run the following sub-steps:

### 5.1 Main PSN-generation parameters

The dynamic PSN software requires several important parameters.

| Parameter | Meaning | Common value in the original workflow |
|---|---|---|
| Distance cutoff | Maximum 3D distance between amino acids for adding an edge in the PSN. | `6`Å |
| Number of amino acids | Number of residues added at each sequential PSN snapshot. | `5` |
| Annotation file | Dataset file listing class labels and protein-domain IDs. | File from `datasets/` |
| CIF directory | Folder containing downloaded `.cif` files. | `cif/` |

### 5.2 Example command for generating PSNs only

```bash
Rscript scripts/cmdrun.r 6 5 "Mode 1" datasets/cath-1.txt cif/ 0
```

This command generates dynamic PSNs for the protein domains listed in `datasets/cath-1.txt`.

### 5.3 Expected dynamic PSN output format

For each protein/domain, the dynamic PSN output should be a folder containing ordered snapshot files:

```text
psns/
|
|-- protein_name_1/
|   |-- 1.txt
|   |-- 2.txt
|   |-- 3.txt
|   |-- ...
|
|-- protein_name_2/
|   |-- 1.txt
|   |-- 2.txt
|   |-- 3.txt
|   |-- ...
```

---

## 6. Step IV: Generate dGDVMs

Dynamic graphlets summarize the topology of dynamic PSNs. In this repository, these features are stored as **dGDVMs**.

A dGDVM is a matrix for one protein/domain sample:

```text
number of rows    = number of amino acids / PSN nodes
number of columns = number of dynamic graphlets
```

In our paper, dGDVMs are the primary input to the CNN+LSTM models. They can also be used as initialized features for SGCN/DGCN.

### 6.1 Example command for generating PSNs and dGDVMs

```bash
Rscript scripts/cmdrun.r 6 5 "Mode 2" datasets/cath-1.txt cif/ 0
```

This command generates both dynamic PSNs and dynamic graphlets.

### 6.2 Expected dGDVM output format

A dGDVM file should contain one matrix per sample. A recommended structure is:

```text
dgdvms/
|
|-- protein_name_1.txt
|-- protein_name_2.txt
|-- ...
```

Each file should contain a numeric matrix:

```text
0 1 2 0 ...
1 0 3 1 ...
2 1 0 4 ...
...
```

Rows correspond to nodes/amino acids. Columns correspond to dynamic graphlet counts.

The file name should match, or be easily normalized to match, the sample name in the dataset annotation file.

*NOTE: generated dGDVMs using the code provided by [K. Newaz, et al. (2022)](https://doi.org/10.1002/prot.26349) will contain zero columns. In our study, we remove the union of all zero columns in all the datasets' dGDVMs, from all dGDVMs. Code to generate the non-zero dGDVMs can be found in `scripts/columns`.*

---

## 7. Step V: Organize the generated data for this repository

After generating PSNs and dGDVMs, organize the data for the method you want to run; the CNN+LSTM and GCN workflows use different input structures.

---

### 7.1 Organizing data for CNN+LSTM

The CNN+LSTM workflow expects dataset-specific folders of dGDVM files.

The dataset scripts in `CNN_LSTM/` create subfolders for each dataset from a larger collection of dGDVM files. This makes it easier to train CNN+LSTM models separately on each PSC datasets.

Relevant scripts:

```text
scripts/make_dataset_all.sh
scripts/make_dataset.sh
```

The intended use is:

- `make_dataset_all.sh`: generate folders for all datasets.
- `make_dataset.sh`: generate a folder for one dataset.

A recommended CNN+LSTM file structure is:

```text
CNN_LSTM/
|
|-- data/
|   |-- dgdvms/
|   |   |-- sample_1.txt
|   |   |-- sample_2.txt
|   |   |-- ...
|   |
|   |-- datasets/
|       |-- cath-1/
|       |   |-- sample_1.txt
|       |   |-- sample_2.txt
|       |   |-- ...
|       |
|       |-- cath-1.10/
|       |-- ...
```

The exact folder names may depend on the script settings. The important point is that each dataset-specific CNN+LSTM training script should see only the dGDVM files belonging to that dataset.

---

### 7.2 Organizing data for SGCN/DGCN

The GCN workflow expects each dataset to have its own folder under a `datasets/` directory inside or relative to the GCN run location.

Required structure:

```text
GCN/
|
|-- datasets/
|   |-- <DATASET_NAME>/
|       |-- <DATASET_NAME>.txt
|       |-- psns/
|       |-- dgdvms/
|       |-- partitions/
```

The `dgdvms/` folder is required only when using:

```bash
--feature_mode dgdvms
```

The `partitions/` folder is required only when using:

```bash
--partitions_mode user
```

If you use:

```bash
--partitions_mode auto
```

then the script automatically generates 5 stratified folds and saves them under the dataset's output folder.

---

## 8. CNN+LSTM method

After the previous steps of setting up the data and file organization, the following sub-steps can be used to operate the CNN+LSTM method:

### 8.1 CNN+LSTM input

Recall that each protein/domain is represented by one dGDVM file:

```text
protein_name_1.txt
```

The matrix inside the file should have the following interpretation:

```text
rows    = amino acids / protein nodes
columns = dynamic graphlets
```

The input to the CNN+LSTM model is therefore a set of matrices, one matrix per protein/domain sample, along with class labels from the corresponding dataset annotation file.

### 8.2 CNN+LSTM model architectures

Four CNN+BiLSTM architectures are implemented.

| Model name | Architecture | Activation | Description |
|---|---|---|---|
| `Relu` | 3 CNN layers + 1 BiLSTM layer | ReLU | A CNN model with one bidirectional LSTM layer. |
| `Leaky` | 3 CNN layers + 1 BiLSTM layer | LeakyReLU | Same depth as `Relu`, but uses LeakyReLU activation. |
| `Paper` | 2 CNN layers + 3 BiLSTM layers | ReLU | Reproduces the original CNN+LSTM architecture used in the paper. |
| `Deep` | 3 CNN layers + 3 BiLSTM layers | ReLU | Deeper CNN and BiLSTM model. |

Each model type has a corresponding template file in `/CNN_LSTM`:

```text
relu_model_template_cv.py
leaky_model_template_cv.py
paper_model_template_cv.py
deep_model_template_cv.py
```

### 8.3 Generating dataset-specific CNN+LSTM model scripts

Training is divided into one model script per dataset. This makes it easier to submit jobs independently on a cluster.

The script:

```text
make_models.py
```

takes a model-template file as input and generates dataset-specific model files.

Example:

```bash
python make_models.py relu_model_template_cv.py
```

Repeat this command for each CNN+LSTM architecture you want to run:

```bash
python make_models.py leaky_model_template_cv.py
python make_models.py paper_model_template_cv.py
python make_models.py deep_model_template_cv.py
```

### 8.4 Running CNN+LSTM training

After dataset-specific model scripts are generated, run each script directly or submit it as a batch job on a cluster environment.

Example:

```bash
python generated_model_for_cath_1.py
```

### 8.5 CNN+LSTM output

After cross-validation training, the CNN+LSTM workflow produces outputs such as:

- per-fold misclassification rate,
- aggregate misclassification rate,
- aggregate confusion matrix,
- logs or intermediate output files depending on the model script.

Only aggregate misclassification results are typically processed downstream by `process_results.py`, while confusion matrices can be extracted using `extract_cm.py`.

---

## 9. SGCN and DGCN method

The GCN workflow runs PSC directly on PSN snapshots.

The `GCN/GCN.py` supports the following methods:

| Method | Input | Main idea |
|---|---|---|
| `SGCN` | Final PSN snapshot only + (default or dGDVMs) | Static graph method; ignores temporal evolution. |
| `DGCN` | Dynamic PSN  + (default or dGDVMs) | Dynamic graph method; uses all snapshots and models temporal information. |

Below, we briefly discribe the method frameworks:

### 9.1 SGCN

Input: one PSN snapshot per protein/domain. Specifically, in our study, SGCN uses only the final PSN snapshot generated using the code provided by [K. Newaz, et al. (2022)](https://doi.org/10.1002/prot.26349).

SGCN method framework:

```text
final PSN snapshot
        |
        v
GCN layers
        |
        v
attention pooling
        |
        v
classifier
        |
        v
predicted PSC class
```

### 9.2 DGCN

DGCN is the dynamic GCN model.

Input: full ordered sequence of PSN snapshots per protein/domain. Specifically, in our study, DGCN uses the entire dynamic PSN generated using the code provided by [K. Newaz, et al. (2022)](https://doi.org/10.1002/prot.26349).

DGCN method framework:

```text
PSN snapshot 1      PSN snapshot 2      ...      PSN snapshot S
      |                   |                           |
      v                   v                           v
   GCN + pool          GCN + pool                  GCN + pool
      |                   |                           |
      +-------------------+---------------------------+
                          |
                          v
                 temporal sequence model
                          |
                          v
                     classifier
                          |
                          v
                  predicted PSC class
```

### 9.3 GCN feature modes

The `GCN.py` script supports two feature initializations for both SGCN and DGCN.

| Feature mode | Description | Required folder |
|---|---|---|
| `default` | Uses randomly initialized/default node features. Feature size is controlled by `--default_feature_dim`. | No `dgdvms/` folder required. |
| `dgdvms` | Uses dGDVM matrices as node features. | Requires `dgdvms/`. |

### 9.4 GCN partition modes

The GCN script supports two partition modes for both SGCN and DGCN.

| Partition mode | Description | Required folder |
|---|---|---|
| `user` | Uses existing 5-fold partitions. | Requires `partitions/1.txt` through `partitions/5.txt`. |
| `auto` | Automatically generates 5 stratified folds. | No user partitions required. |

If using `user` mode, the partition folder must contain:

```text
partitions/
|-- 1.txt
|-- 2.txt
|-- 3.txt
|-- 4.txt
|-- 5.txt
```

Each file lists the test samples for that fold, one protein/domain per line.

### 9.5 GCN command-line arguments

Required arguments:

```text
--dataset_name       Dataset folder name.
--model_type         sgcn or dgcn.
--feature_mode       default or dgdvms.
--partitions_mode    user or auto.
```

Optional arguments:

```text
--root_dir                 Root directory. Default: .
--batch_size               Batch size. Default: 1.
--epochs                   Maximum number of training epochs. Default: 100.
--patience                 Early-stopping patience. Default: 5.
--save_models              Save trained model files.
--quiet_epochs             Reduce epoch-level printed output.
```

### 9.6 GCN output

All GCN outputs are written to:

```text
GCN/datasets/<DATASET_NAME>/output/
```

Output files include:

```text
run.log
fold_metrics.csv
raw_classifications.csv
optimal_hyperparameters.json
runtime_seconds.txt
generated_partitions/
```

Output meanings:

| Output | Meaning |
|---|---|
| `run.log` | Full log of the run. |
| `fold_metrics.csv` | Fold-level performance values. |
| `raw_classifications.csv` | Per-sample predictions and true labels. |
| `optimal_hyperparameters.json` | Best hyperparameters selected during inner CV. |
| `runtime_seconds.txt` | Total runtime. |
| `generated_partitions/` | Auto-generated partitions, if `--partitions_mode auto` was used. |

*Important: if an invalid sample is detected, the GCN script records the invalid sample in a `skipped_samples.txt` and in `run.log`, and then continues.*

---

## 10. Inputs and outputs for each method

### 10.1 CNN+LSTM

### 10.2 SGCN

### 10.3 DGCN

---

## 11. Results processing

### 11.1 CNN+LSTM results

Use:

```bash
python process_results.py
```

This script compiles aggregate misclassification rates across datasets.

Use:

```bash
python extract_cm.py
```

This script extracts aggregate confusion matrices for class-level performance analysis.

### 11.2 GCN results

GCN results are already written into the dataset-specific output folder:

```text
GCN/datasets/<DATASET_NAME>/output/
```

```text
fold_metrics.csv
raw_classifications.csv
runtime_seconds.txt
```

from each dataset output folder.

---
