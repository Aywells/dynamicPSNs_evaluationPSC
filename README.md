# CNN+LSTM Deep Learning Pipeline

This repository contains scripts for:

- Generating 72 structured datasets from dGDV files  
- Creating dataset-specific deep learning model scripts  
- Training CNN–BiLSTM architectures across all datasets  
- Processing aggregate evaluation results  

The codebase is designed for scalable execution (e.g., CRC batch systems) and structured experimentation across multiple model architectures.

---

## Overview

The experimental pipeline consists of three stages:

1. Dataset generation  
2. Model generation and training  
3. Results processing  

Each stage is modular and can be executed independently.

---

# 1. Dataset Generation

The dataset scripts create subfolders for each dataset from a larger collection of dGDV files.  

This results in **72 independent datasets**, enabling:

- Per-dataset model training  
- Clean experiment organization  
- Efficient cluster-based batch execution  

## Scripts

### `make_dataset_all.sh`

Generates all 72 datasets using predefined class files:

dataset-xx.txt


Each `dataset-xx.txt` file defines the class structure for one dataset.

Run:

bash
bash make_dataset_all.sh
make_dataset.sh
Generates a single dataset corresponding to a specified class file.

Run:

bash make_dataset.sh dataset-01.txt
Use this when regenerating or debugging a specific dataset.

# 2. Model Generation
To streamline large-scale training, the model training is divided into 72 sub-models, one per dataset.

This design allows:

Clean mapping between dataset and model

Simple batch submission to CRC

Organized experiment tracking

Model Generator
make_models.py
This script:

Takes a model template file as input

Generates 72 dataset-specific model files

Produces one model configuration per dataset

Example:

python make_models.py relu_model_template_cv.py
Model Architectures
Four CNN–BiLSTM architectures are implemented.

1. Relu
3 CNN layers

1 BiLSTM layer

ReLU activation

2. Leaky
3 CNN layers

1 BiLSTM layer

LeakyReLU activation

3. Paper
2 CNN layers

3 BiLSTM layers

Reproduces the original architecture described in the paper

ReLU activation

4. Deep
3 CNN layers

3 BiLSTM layers

Deep CNN + deep BiLSTM architecture

Model Templates
Each architecture has a corresponding template file:

relu_model_template_cv.py

leaky_model_template_cv.py

paper_model_template_cv.py

deep_model_template_cv.py

These templates are used by make_models.py to generate dataset-specific model scripts.

# 3. Training Outputs
After cross-validation training, the following outputs are produced:

Per-fold misclassification rate

Aggregate misclassification rate

Aggregate confusion matrix

Note: Only aggregate misclassification results are processed downstream.

# 4. Results Processing
process_results.py
Processes and compiles:

Aggregate misclassification rates across datasets

Run:

python process_results.py
extract_cm.py
Extracts:

Aggregate confusion matrices

Useful for detailed class-level performance analysis.

Run:

python extract_cm.py
Typical Workflow
Step 1 — Generate Datasets
bash make_dataset_all.sh
Step 2 — Generate Dataset-Specific Models
python make_models.py relu_model_template_cv.py
Repeat for other templates as needed.

Step 3 — Submit Training Jobs
Submit the generated model scripts to CRC (or other cluster environment).

Step 4 — Process Results
python process_results.py
python extract_cm.py
