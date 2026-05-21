This code runs protein structure classification (PSC) experiments using either:
- a Static Graph Convolutional Network (SGCN), or
- a Dynamic Graph Convolutional Network (DGCN)

The script supports:
- two feature modes:
  (i) default: randomly initialized node features
  (ii) dgdvms: node features initialized using dGDVM feature matrices
- two partition modes:
  (i) user: use existing 5-fold partitions stored in the dataset folder
  (ii) auto: automatically generate 5 stratified folds
- nested 5-fold cross-validation:
  outer CV for evaluation and inner CV for hyperparameter tuning
- logging to both terminal and file
- strict dataset validation (program stops if any sample is invalid)

For a given dataset, the script:
1. loads all protein samples
2. loads PSN snapshots
3. optionally loads dGDVM features
4. validates all samples
5. loads or generates partitions
6. performs nested cross-validation
7. tunes hyperparameters
8. trains the model
9. evaluates performance
10. writes outputs to the dataset output folder

---------------------------------------------------------------------

MODEL TYPES

SGCN:
- uses only the final PSN snapshot
- no temporal modeling

DGCN:
- uses all PSN snapshots
- applies GCN per snapshot
- aggregates via attention pooling
- models temporal dependencies using a bidirectional LSTM

---------------------------------------------------------------------

FEATURE MODES

default:
- random node features
- controlled by default_feature_dim

dgdvms:
- loads features from dgdvms/ folder
- matches files to samples using normalized filenames

---------------------------------------------------------------------

PARTITION MODES

user:
- reads 5 partition files from:
  datasets/<dataset_name>/partitions/
- files must be named:
  1.txt, 2.txt, 3.txt, 4.txt, 5.txt

auto:
- generates 5 stratified folds
- saves them to:
  datasets/<dataset_name>/output/generated_partitions/

---------------------------------------------------------------------

REQUIRED FOLDER STRUCTURE

datasets/
    <DATASET_NAME>/
        <DATASET_NAME>.txt
        dgdvms/
        psns/
        partitions/

---------------------------------------------------------------------

DATASET LABEL FILE

File:
datasets/<DATASET_NAME>/<DATASET_NAME>.txt

Format:
<class_label> <sample_name>

Example:
0 sampleA
1 sampleB
2 sampleC

---------------------------------------------------------------------

DGDVM FOLDER

Location:
datasets/<DATASET_NAME>/dgdvms/

Contains one .txt per sample.

Format:
Rows = nodes
Columns = features

Example:
0 1 2
1 0 3
2 1 0

File names must roughly match sample names.

---------------------------------------------------------------------

PSN FOLDER

Location:
datasets/<DATASET_NAME>/psns/<sample_name>/

Contains:
1.txt, 2.txt, 3.txt, ...

Each file is an edge list:
u v

Example:
1 2
2 3
3 4

Supports both 0-based and 1-based indexing.

---------------------------------------------------------------------

PARTITIONS FOLDER

Location:
datasets/<DATASET_NAME>/partitions/

Files:
1.txt to 5.txt

Each file lists test samples for that fold.

---------------------------------------------------------------------

OUTPUT FOLDER

All outputs go to:
datasets/<DATASET_NAME>/output/

Files may include:

run.log
summary_metrics.json
fold_metrics.csv
raw_classifications.csv
optimal_hyperparameters.json
runtime_seconds.txt
generated_partitions/ (if auto mode)
skipped_samples.txt (if errors occur)

---------------------------------------------------------------------

IMPORTANT: INVALID SAMPLES

If any sample is invalid:
- it is recorded
- written to skipped_samples.txt
- program stops immediately

---------------------------------------------------------------------

MODEL DETAILS

Graph convolution:

SGCN:
- 2 GCN layers (64 hidden units)
- attention pooling
- classifier: 64 → 32 → classes

DGCN:
- 2 GCN layers per snapshot
- attention pooling per snapshot
- BiLSTM (hidden size 64 per direction)
- classifier: 128 → 32 → classes

---------------------------------------------------------------------

CROSS-VALIDATION

Outer CV:
- 5 folds for testing

Inner CV:
- 5 folds for hyperparameter tuning

---------------------------------------------------------------------

HYPERPARAMETERS

Grid search over:
- learning rate
- dropout
- weight decay

Example:
--lr_grid 1e-3,5e-4
--dropout_grid 0.3,0.4
--weight_decay_grid 0.0,1e-5

---------------------------------------------------------------------

COMMAND-LINE ARGUMENTS

Required:
--dataset_name
--model_type (sgcn or dgcn)
--feature_mode (default or dgdvms)
--partitions_mode (user or auto)

Optional:
--root_dir (default: .)
--default_feature_dim (default: 1072)
--batch_size (default: 5)
--epochs (default: 100)
--patience (default: 10)
--scheduler_factor (default: 0.5)
--scheduler_patience (default: 5)
--outer_folds (default: 5)
--inner_folds (default: 5)
--seed (default: 42)
--device (default: auto)
--num_workers (default: 0)
--lr_grid
--dropout_grid
--weight_decay_grid
--save_models
--quiet_epochs

---------------------------------------------------------------------

EXAMPLE COMMANDS

DGCN with dgdvms:
python run_psn_gcn_cv.py --dataset_name Cath-1 --model_type dgcn --feature_mode dgdvms --partitions_mode user

SGCN with default features:
python run_psn_gcn_cv.py --dataset_name Cath-1 --model_type sgcn --feature_mode default --partitions_mode auto

---------------------------------------------------------------------

DGCN can be memory intensive due to padding.

If you get "Killed":
- reduce batch size:
  --batch_size 1
- reduce feature dimension:
  --default_feature_dim 64