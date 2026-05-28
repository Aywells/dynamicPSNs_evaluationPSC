# *Traditional* machine learning vs. *deep* learning from dynamic graph representations of proteins’ 3D folds in the task of protein structure classification - CODE
---
This repository describes the 72 datasets and contains the code used to run the regular deep learning (CNN+LSTM) and graph-based deep learning (GCN) method variants from our paper, "Traditional machine learning vs. deep learning from dynamic graph representations of proteins’ 3D folds in the task of protein structure classification", A. Wells, F. A. Gatsi, A. Striegel, and T. Milenković (2026), under review, 2026."

# Directories in the repository

*The following are the directories in the repository:*

## &emsp;`data/`

#### `data/72_datasets/`

   This directory contains the information about the 72 considered datasets. This information includes the list of all domains in a given dataset, and the domains' CATH or SCOPe structural class annotations. We discuss the 72 dataset files in more detail in [I.a. More details on the 72 dataset files](#a-more-details-on-the-72-dataset-files)

#### `data/data_examples/`

   This directory contains detailed data (`dynamic_PSNs`,`non-zero_dGDVMs`, and `5_training_testing_folds` for cross validation) for three smaller of all 72 considered datasets, intended to allow for running our code easily. Note that we cannot provide such detailed data for all 72 datasets due to large space requirements. Nonetheless, in [I.b. Download CIF files](#b-download-cif-files) and [I.c. Construct dynamic PSNs and extract (full) dGDVMs](#c-construct-dynamic-psns-and-extract-full-dgdvms), we explain a set-by-set procedure on how, from the information in `data/72_datasets/`, one can produce such detailed data, including how to download a `.cif` file for a given domain in a given dataset, how to contruct the dynamic PSN for that domain, and how to extract the (full) dGDVM for that dynamic PSN.

#### `data/data_processing_scripts/`

   This directory contains one utility script for converting the (full) dGDVM into its non-zero counterpart. It also contains two utility scripts for preparing the data for use in the considered deep learning variants.

#### `data/results_from_the_study`
   This directory contains detailed data produced and descriped in the paper. Specifically, it contains two excel files, `results_variants` and `results_baselines`, where the former provides per-dataset misclassification rates and runtimes (in minutes) for the variants presented in the paper, and the latter provides per-dataset misclassification rates for the baseline methods.


## &emsp;`code/`

#### `code/CNN_LSTM/`

   This directory contains the code for the four considered CNN+LSTM variants. We discuss the CNN+LSTM code in more detail in [III.a. CNN+LSTM variants](#a-cnnlstm-variants),.

#### `code/GCN/`

   This directory contains the code for the three considered GCN variants. We discuss the GCN code in more detail in [III.b. GCN variants](#b-gcn-variants),.

---

# The pipeline to use the data and the code

At a high level, the repository supports the following steps (i.e. the same steps used in the paper):

- [I. Prepare the data](#i-prepare-the-data)
   - [a. More details on the 72 dataset files](#a-more-details-on-the-72-dataset-files)
   - [b. Download CIF files](#b-download-cif-files)
   - [c. Contruct dynamic PSNs and extract (full) dGDVMs](#c-contruct-dynamic-psns-and-extract-full-dgdvms)

- [II. Process and clean the data](#ii-process-and-clean-the-data)
   - [a. Convert (full) dGDVM into non-zero dGDVM](#a-convert-full-dgdvm-into-non-zero-dgdvm)
   - [b. Prepare non-zero dGDVMs for use in method variants](#b-prepare-non-zero-dgdvms-for-use-in-method-variants)

- [III. Run the method variants](#iii-run-the-method-variants)
   - [a. CNN+LSTM variants](#a-cnnlstm-variants)
   - [b. GCN variants](#b-gcn-variants)

- [IV. Our results: misclassification rates and runtimes for all method variants](#v-our-results-misclassification-rates-and-runtimes-for-all-method-variants)
   - [a. Results for the main method variants](#a-results-for-the-main-method-variants)
   - [b. Results for the baseline methods](#b-results-for-the-baseline-methods)

Below, we go into each step in more detail:

---

## &emsp;I. Prepare/generate the data

In the paper, we use the same 72 datasets introduced by [Newaz et al. (2022)](https://doi.org/10.1002/prot.26349) ([GitHub link](https://github.com/KhaliqueN/DynamicPSN)). In addition, the method to aquire `.cif` files, Dynamic PSNs and (full) dGDVMs is the same as what is instructed by [Newaz et al. (2022)](https://doi.org/10.1002/prot.26349). For completeness, we include the following instructions for preparing and generating the data.


### &emsp;&emsp;&emsp;a. More details on the 72 dataset files

The dataset files are provided in `data/72_datasets/`, where each `.txt` file in this directory corresponds to one of the 72 datasets in the paper.

In a dataset file, each line corresponds to one protein domain and contains two tab-separated columns:

```text
<class_label>    <protein_domain_name>
```

| Column | Description |
|---|---|
| <class_label> | The structural class label of the protein domain (from either CATH or SCOPe). This is the prediction label used for PSC. |
| <protein_domain_name> | The protein domain identifier, which specifies the CIF structure, chain, and residue range(s) that define the domain. |

The two columns are separated by a tab.

Note that the <protein_domain_name> in in the following format:

```text
CIFID_chainID_startID1_endID1+startID2_endID2+...+startIDn_endIDn
```
where:
- `CIFID` is the CIF identifier without the `.cif` extension.
- `chainID` is the protein chain identifier.
- `startID` and `endID` define the residue range for a domain segment.
- Multiple discontinuous segments are separated using `+`.

---

### &emsp;&emsp;&emsp;b. Download CIF files

For every row in every dataset files, download its corresponding `.cif` file referenced by the `CIFID` in the <protein_domain_name> identifier.

For example, the domain:

```text
1fnn_A_1_17+192_275
```

requires the file:

```text
1fnn.cif
```

We recommend the user to make a directory to store the downloaded CIF files. Note that across all 72 datasets used in the paper, there exist some identical protein domains.

For batch downloading of `.cif` files, see the following batch downloader from the Protein Data Bank (PDB) ([click here](https://www.rcsb.org/downloads))

---

### &emsp;&emsp;&emsp;c. Construct dynamic PSNs and extract (full) dGDVMs

For a dataset, we generate dynamic PSNs for all protein domains following the instructions in, and using the software from, [K. Newaz, et al. (2022)](https://doi.org/10.1002/prot.26349) ([GitHub link](https://github.com/KhaliqueN/DynamicPSN)). The user should download this software. This software can be run in different modes:

| Mode | Output |
|---|---|
| `Mode 1` | Generate dynamic PSNs only. |
| `Mode 2` | Generate dynamic PSNs and dynamic graphlets. |
| `Mode 3` | Generate dynamic PSNs, dynamic graphlets, and run the original LR-based PSC. |

For our study, the most important outputs are:

1. **Dynamic PSNs**, which are later used as input for the graph-based deep learning variants ([III.b. GCN variants](#b-gcn-variants)).
2. **dGDVMs**, which are later used as input for the regular deep learning variants, and optionally for the graph-based deep learning variants ([III.a. CNN+LSTM variants](#a-cnnlstm-variants) and [III.b. GCN variants](#b-gcn-variants)).

Therefore, for our purposes, for each dataset file, we run consider the following when running the software:

#### Main PSN-generation parameters

The dynamic PSN software requires several important parameters:

| Parameter | Meaning | Value used in the paper when running the software |
|---|---|---|
| Distance_cutoff | Maximum 3D distance between amino acids for adding an edge in the PSN. | `6`Å |
| Number_of_amino_acids | Number of residues added at each sequential PSN snapshot. | `5` |
| Dataset_file | Dataset `.txt` file | Path to a dataset file in `datasets/` |
| CIF_directory | Folder containing downloaded `.cif` files. |  Path to the folder containing downloaded `.cif` files |

#### Command for generating PSNs and (full) dGDVMs

Next, we run the following command to generate the dynamic PSNs and (full) dGDVMs:

```bash
Rscript scripts/cmdrun.r Distance_cutoff Number_of_amino_acids "Mode 2" datasets/Dataset_file.txt CIF_directory/ 0
```
This command generates dynamic PSNs and (full) dGDVMs for the protein domains listed in `datasets/Dataset_file.txt`.

#### Expected dynamic PSN output and (full) dGDVM file format

After completion of the above command, a dataset's generated dynamic PSNs should be located in a newly created directory called `output/dynamic-networks`.

For a protein domain in a dataset, its generated dynamic PSN output should be in a folder containing ordered snapshot files. For example:

```text
dynamic-networks/
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

Additionally, a dataset's generated (full) dGDVMs should be located in a newly created directory called `output/feature-matrix`.

For a protein domain in a dataset, its generated (full) dGDVM file should contain a numeric matrix. For example:
```

```text
0 1 2 0 ...
1 0 3 1 ...
2 1 0 4 ...
...
```

In a (full) dGDVM file, rows correspond to nodes/amino acids and columns correspond to dynamic graphlet counts. In addition, the first column in the file corresponds to the node/amino acid index.

## &emsp;II. Data processing and cleaning

After aquiring the dynamic PSNs and their corresponding (full) dGDVMs, we process and clean the data in preparation for input/initialization into our regular deep learning and graph-based deep learning variants. The user is to run the following commands/programs in order:

- Because generated (full) dGDVMs -- using the code provided by [K. Newaz, et al. (2022)](https://doi.org/10.1002/prot.26349) -- will contain zero columns (i.e. zero graphlet counts for all nodes), in our study, we remove the union of all zero columns in all the datasets' (full) dGDVMs, from all (full) dGDVMs. to do so, we run the following command:

```bash
python columns.py
```
with the directory containing the (full) dGDVMs (i.e. `output/feature-matrix`) placed in the same folder of the code used to generate the non-zero dGDVMs (i.e. `scripts/columns.py`).

- Because the regular deep learning variants expect a particular organization of datasets and non-zero dGDVM files, we run the following programs:

```bash
scripts/make_dataset_all.sh
scripts/make_dataset.sh
```

with the directory containing the non-zero dGDVMs and `72_datasets` placed in the same folder of the scripts `make_dataset_all.sh` and `make_dataset.sh`.

*NOTE: The exact folder names may depend on the script settings. The important point is that each script should see only the dGDVM files belonging to a dataset or all.*

---

## &emsp;III. Run the method variants

After preparing the non-zero dGDVMs and dynamic PSNs, we run the considered deep learning method variants. In the paper, we consider two main paradigms of deep learning approaches:

1. **Regular deep learning variants**, which use the non-zero dGDVMs as input.
2. **Graph-based deep learning variants**, which use dynamic PSNs, and, optionally non-zero dGDVMs, as input.

The code for these method variants is provided in `code/CNN_LSTM` and `/conde/GCN`, respectively

For each method variant, the user should run the corresponding scripts dataset-by-dataset. Each run produces output files containing the performance of the method on a given PSC dataset. 

Below, we go into more detail on both the regular deep learning and graph-based deep learning variants and their corresponding outputs in [III.a. CNN+LSTM variants](#a-cnnlstm-variants) and [III.b. GCN variants](#b-gcn-variants), respectively.

---

### &emsp;&emsp;&emsp;a. CNN+LSTM variants

The user can find the code for all CNN+LSTM variants in `code/CNN_LSTM/`.

#### Method variant code

In the paper, we consider the following four CNN+LSTM variants:

| Varient name as in the paper | Varient name as in `code/CNN_LSTM/` | Input | Main idea |
|---|---|---|---|
| Dynamic graphlets + (2 CNN, 3 LSTM) | `paper_model_template_cv.py` | non-zero dGDVM matrix for each protein | The "default" variant inspired by [H. Guo, et al. (2019)](https://arxiv.org/abs/1910.02594); uses 2 CNN layers followed by 3 LSTM layers with ReLU activation. |
| Dynamic graphlets +  (3 CNN, 3 LSTM) | `deep_model_template_cv.py` | non-zero dGDVM matrix for each protein | The variant that uses 3 CNN layers followed by 3 LSTM layers with ReLU activation. |
| Dynamic graphlets +  (3 CNN, 1 LSTM) | `relu_model_template_cv.py` | non-zero dGDVM matrix for each protein | The variant that uses 3 CNN layers followed by 1 LSTM layer with ReLU activation. |
| Dynamic graphlets +  (3 CNN, 1 LSTM) under LeakyReLU | `leaky_model_template_cv.py` | non-zero dGDVM matrix for each protein | The variant that uses 3 CNN layers followed by 1 LSTM layer with LeakyReLU activation. |

#### Input

The general input for each varient is a dataset's:

```text
non-zero dGDVM files
class labels
training/testing folds (if provided)
```

For a given dataset, the CNN+LSTM code should be run using these files. The expected input organization is the one produced in [II. Process and clean the data](#ii-process-and-clean-the-data).

Training is divided into one variant script per dataset. This makes it easier to submit jobs independently and in parallel on a cluster. To generate dataset-specific variants, the following script is run:

```text
make_models.py
```

which takes a varient-template file as input and generates dataset-specific variant files. For example:

```bash
python make_models.py relu_model_template_cv.py
```

Repeat this command for each CNN+LSTM variant the user wants to run:

```bash
python make_models.py leaky_model_template_cv.py
python make_models.py paper_model_template_cv.py
python make_models.py deep_model_template_cv.py
```

After dataset-specific model scripts are generated, run each script directly or submit it as a batch job on a cluster environment.

Example:

```bash
python generated_model_for_dataset.py
```

*Note that optional command-line arguments and method parameters are fixed in the varient-template files. If a user wishes to change the method parameters/arguments for a CNN+LSTM variant, one would need to change these parameters in the `.py` file, and then re-generate dataset-specific variant files using `make_models.py` (described above)*

#### Output and interpretation

For each dataset, the method is evaluated under five-fold cross-validation. That is, the model is trained and tested five times, once per fold. The output should therefore allow the user to determine the method's performance on each fold, as well as its aggregate misclassification performance across folds. The CNN+LSTM workflow produces log files that contain the following outputs:

- per-fold misclassification rate,
- aggregate misclassification rate,
- aggregate confusion matrix,
- variant run times

In our workflow, two additional scripts are used to extract and summarize the outputs from the CNN+LSTM method runs:

```text
process_results.py
extract_cm.py
```
These scripts are used for different purposes:

The script `process_results.py` is used to collect and summarize the misclassification-rate results produced by the CNN+LSTM method runs.

In particular, this script is used to extract the aggregate misclassification rate for each dataset and method variant. The typical input to `process_results.py` is the directory or set of output files produced by a CNN+LSTM method run. The script reads these output files, extracts the relevant misclassification-rate values, and organizes them into a cleaner result file that can be used for downstream comparison across datasets and methods.

The script `extract_cm.py` is used to extract confusion matrices from the CNN+LSTM method outputs.

While the misclassification rate tells us how often the method is wrong overall, the confusion matrix helps identify which structural classes are being confused with one another. The typical input to `extract_cm.py` is the raw CNN+LSTM output containing prediction results or logged confusion-matrix information. The script extracts the confusion matrix and writes it into a separate, easier-to-read format.

---

### &emsp;&emsp;&emsp;b. GCN variants

The user can find the code for all GCMN variants in `code/GCN/`.


#### Method variant code

In the paper, we consider the following three GCN variants:

| Method variant             | Description                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `Default features + DGCN`  | Uses dynamic PSNs as graph inputs with default node feature initialization.                       |
| `Dynamic graphlets + DGCN` | Uses dynamic PSNs as graph inputs with non-zero dGDVM feature initialization.        |
| `Dynamic graphlets + SGCN` | Uses the final/static PSN representation with non-zero dGDVM feature initialization. |

Note that the GCN code supports both **static GCN (SGCN)** and **dynamic GCN (DGCN)** methods. Below we go into more detail on running both the SGCN and DGCN methods

#### Input

The general inputs to the GCN methods are:

```text
dynamic PSNs
non-zero dGDVM files (when dGDVMs are used)
dataset class labels
training/testing folds (if provided)
```

For the dynamic GCN variants, the input for each protein domain is a full dynamic PSN. These ordered snapshots are the same dynamic PSNs generated in [I.c. Construct dynamic PSNs and extract (full) dGDVMs](#c-construct-dynamic-psns-and-extract-full-dgdvms).

For the static GCN variant, only the final PSN snapshot of the dynamic PSN is used.

#### Main GCN parameters

The `GCN.py` script supports two feature initialization modes for both SGCN and DGCN:

| Feature mode | Description                                                                                             | Required folder               |
| ------------ | ------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `default`    | Uses randomly initialized/default node features. Feature size is controlled by `--default_feature_dim`. | No `dgdvms/` folder required. |
| `dgdvms`     | Uses non-zero dGDVM matrices as node features.                                                          | Requires `dgdvms/`.           |

The `dgdvms/` folder is required only when using:

```bash
--feature_mode dgdvms
```

If using:

```bash
--feature_mode default
```

then no `dgdvms/` folder is required.

Inaddition, the `GCN.py` script supports two partition modes for both SGCN and DGCN:

| Partition mode | Description                                    | Required folder                                         |
| -------------- | ---------------------------------------------- | ------------------------------------------------------- |
| `user`         | Uses existing five-fold partitions.            | Requires `partitions/1.txt` through `partitions/5.txt`. |
| `auto`         | Automatically generates five stratified folds. | No user-provided partitions are required.               |

The `partitions/` folder is required only when using:

```bash
--partitions_mode user
```

If using:

```bash
--partitions_mode auto
```

then the script automatically generates five stratified folds and saves them under the dataset's output folder.

If the user wishes to match a specific set of five-fold partitions, then `--partitions_mode user` should be used. In this case, the partition folder must contain:

```text
partitions/
|-- 1.txt
|-- 2.txt
|-- 3.txt
|-- 4.txt
|-- 5.txt
```

Each file lists the test samples for that fold, one protein/domain per line.

#### GCN command-line arguments

The required command-line arguments are:

```text
--dataset_name       Dataset folder name.
--model_type         sgcn or dgcn.
--feature_mode       default or dgdvms.
--partitions_mode    user or auto.
```

The optional command-line arguments include:

```text
--root_dir                 Root directory. Default: .
--batch_size               Batch size. Default: 1.
--epochs                   Maximum number of training epochs. Default: 100.
--patience                 Early-stopping patience. Default: 5.
--save_models              Save trained model files.
--quiet_epochs             Reduce epoch-level printed output.
```

For example, to run DGCN with non-zero dGDVM node features and user-provided five-fold partitions, the command would have the following general form:

```bash
python GCN.py \
    --dataset_name DATASET_NAME \
    --model_type dgcn \
    --feature_mode dgdvms \
    --partitions_mode user
```

To run SGCN with non-zero dGDVM node features and automatically generated partitions, the command would have the following general form:

```bash
python GCN.py \
    --dataset_name DATASET_NAME \
    --model_type sgcn \
    --feature_mode dgdvms \
    --partitions_mode auto
```

To run DGCN with default node features and automatically generated partitions, the command would have the following general form:

```bash
python GCN.py \
    --dataset_name DATASET_NAME \
    --model_type dgcn \
    --feature_mode default \
    --partitions_mode auto
```

#### Output and interpretation

All GCN outputs are written to:

```text
GCN/<DATASET_NAME>/output/
```

The output files include:


- run.log
- fold_metrics.csv
- raw_classifications.csv
- optimal_hyperparameters.json
- runtime_seconds.txt
- generated_partitions/


The output files have the following meanings:

| Output                         | Meaning                                                          |
| ------------------------------ | ---------------------------------------------------------------- |
| `run.log`                      | Full log of the run.                                             |
| `fold_metrics.csv`             | Fold-level performance values.                                   |
| `raw_classifications.csv`      | Per-sample predictions and true labels.                          |
| `optimal_hyperparameters.json` | Best hyperparameters selected during inner cross-validation.     |
| `runtime_seconds.txt`          | Total runtime of the run.                                        |
| `generated_partitions/`        | Auto-generated partitions, if `--partitions_mode auto` was used. |

For each dataset, the GCN variant is evaluated under five-fold cross-validation. The outputs should therefore allow the user to determine the method's fold-level performance, aggregate misclassification performance across folds, per-sample predictions, selected hyperparameters, and runtime.

*NOTE: if an invalid sample is detected, the GCN script records the invalid sample in:*

```text
skipped_samples.txt
run.log
```
and then continues running. Therefore, the user should check these files after each run to determine whether any samples were skipped.

##  &emsp;IV. Our results: misclassification rates and runtimes for all method variants

The results reported in the paper are provided in:

```text
data/results_from_the_study/
```

This directory contains two Excel files:

```text
results_variants.xlsx
results_baselines.xlsx
```

These files contain the per-dataset results used to compare the considered protein structure classification methods across the 72 datasets.

---

### &emsp;&emsp;&emsp;a. Results for the main method variants

The file:

```text
results_variants.xlsx
```

contains the per-dataset misclassification rates and runtimes for the main method variants considered in the paper.

Each row corresponds to one PSC dataset. The columns include the dataset name and method-specific result columns.

| Column suffix   | Meaning                                                     |
| --------------- | ----------------------------------------------------------- |
| `_agg_misclass` | Aggregate misclassification rate for a method on a dataset. |
| `_run_time`     | Runtime for a method on a dataset, reported in minutes.     |

The methods included in this file are:

| Method type       | Methods                                                                                                                                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LR method         | `Dynamic graphlets + LR`                                                                                                                                                                                                  |
| CNN+LSTM variants | `Dynamic graphlets + regular deep learning (2,3)`; `Dynamic graphlets + regular deep learning (3,1,ReLu)`; `Dynamic graphlets + regular deep learning (3,1,leakyReLu)`; `Dynamic graphlets + regular deep learning (3,3)` |
| GCN variants      | `Default features + DGCN`; `Dynamic graphlets + DGCN`; `Dynamic graphlets + SGCN`                                                                                                                                         |

Lower misclassification rates indicate better classification performance. Lower runtimes indicate faster execution.

---

### &emsp;&emsp;&emsp;b. Results for the baseline methods

The file:

```text
results_baselines.xlsx
```

contains the per-dataset misclassification rates for the baseline methods.

Each row corresponds to one PSC dataset. The columns are:

| Column                      | Meaning                                                                   |
| --------------------------- | ------------------------------------------------------------------------- |
| `Dataset`                   | Name of the PSC dataset.                                                  |
| `Majority class baseline`   | Misclassification rate obtained by always predicting the majority class.  |
| `Static graphlets baseline` | Misclassification rate obtained using the static graphlet-based baseline. |

These baseline results provide reference points for interpreting the main method variants. The majority-class baseline shows whether a method learns information beyond simply predicting the largest class, while the static graphlet baseline helps evaluate whether dynamic or deep learning-based methods improve over a static graphlet representation.
