# Dynamic Graph Convolutional Network (DGCN)

A PyTorch implementation of a Dynamic Graph Convolutional Network with multi-level attention mechanisms for temporal graph classification.

## Overview

The dGCN framework combines:
- **Spatial Processing**: Multi-layer Graph Convolutional Network (GCN) with residual connections
- **Graph-level Pooling**: Attention-based aggregation of node features to graph embeddings
- **Temporal Processing**: Multi-head self-attention over temporal sequences
- **Classification Head**: MLP classifier for graph-level predictions

This framework is designed to classify dynamic graphs (sequences of snapshots) with optional support for graphlet-based features.

## Features

- ✅ Multi-layer GCN with layer normalization and residual connections
- ✅ Attention-based graph-level pooling
- ✅ Temporal attention over graph sequences
- ✅ Support for custom node features (default or graphlet-based)
- ✅ 5-fold cross-validation training loop
- ✅ Automatic feature dimension detection
- ✅ String label support with automatic numeric class mapping
- ✅ Flexible directory-based data organization

## Directory Structure

Organize your data as follows:

```
.
├── datasets/              # Dataset definition files
│   └── dataset_name.txt   # Format: <label> <sample_name> per line
├── dynamic-networks/      # Network snapshots for each sample
│   ├── sample1/
│   │   ├── snapshot_1.npy (or .txt, .npz)
│   │   ├── snapshot_2.npy
│   │   └── ...
│   ├── sample2/
│   │   ├── snapshot_1.npy
│   │   └── ...
│   └── ...
├── dynamic-graphlets/     # Graphlet features (optional)
│   ├── sample1_dcgdv.npy
│   ├── sample2_dcgdv.npy
│   └── ...
├── partitions/            # 5-fold partition definitions
│   └── dataset_name/
│       ├── 1  # Format: one sample name per line
│       ├── 2
│       ├── 3
│       ├── 4
│       └── 5
├── DGCN.py
└── README.md
```

## Dataset File Format

**datasets/your_dataset.txt**:
```
class_A sample1
class_B sample2
class_A sample3
class_C sample4
```

Labels can be:
- Numeric: `0 sample1`, `1 sample2`
- Strings: `disease sample1`, `healthy sample2`
- Any non-numeric identifiers

Labels are automatically mapped to integer classes (0, 1, 2, ...).

## Partition Files Format

**partitions/dataset_name/1** (and 2, 3, 4, 5):
```
sample1
sample3
sample7
```

Each fold file contains one sample name per line.

## Adjacency Matrix Formats

Snapshot files can be in:
- **NumPy arrays**: `.npy` or `.npz` (2D square matrices for adjacency)
- **Text files**: `.txt` with space/comma-separated values
- **Edge lists**: 2 columns (u, v) pairs; converted to adjacency matrix

## Setup

### Requirements
```
torch>=1.9.0
numpy>=1.19.0
```

### Installation
```bash
pip install torch numpy
```

## Usage

### 1. Using Real Data

Edit `DGCN.py`:

```python
DATASET_TXT = 'scop-g.txt'           # Your dataset filename
FEATURE_MODE = 'default'              # 'default' or 'graphlet'
```

Ensure directory structure is set up correctly, then run:
```bash
python DGCN.py
```

### 2. Using Synthetic Data (Demo)

Leave `DATASET_TXT = ''` to use synthetic data:

```python
DATASET_TXT = ''  # Falls back to synthetic
```

### Feature Modes

#### Default Mode
Uses ones as node features (shape: [num_nodes, 1]).
```python
FEATURE_MODE = 'default'
```

#### Graphlet Mode
Loads graphlet-based features from corresponding files (e.g., `sample1_dcgdv.npy`).
```python
FEATURE_MODE = 'graphlet'
```

Expected graphlet feature files:
- Naming pattern: `*{sample_name}*dcgdv*` (wildcards allowed)
- Format: NumPy array of shape [feature_dim, num_nodes] or [num_nodes, feature_dim]
- Automatically detected and transposed if needed

### 3. Hyperparameters

Customize in `DGCN.py`, within `main()`:

```python
FEATURE_DIM = 8              # Auto-detected from real data if available
HIDDEN_DIM = 64              # GCN hidden dimension
NUM_CLASSES = 4              # Auto-detected from labels if using real data
GCN_LAYERS = 2               # Number of GCN layers
NUM_HEADS = 4                # Attention heads
DROPOUT = 0.3                # Dropout rate
BATCH_SIZE = 8               # Batch size
EPOCHS = 50                  # Epochs per fold
LEARNING_RATE = 0.001        # Optimizer learning rate
```

## Workflow

### Training Pipeline

1. **Load Dataset**: Reads `.txt` file and extracts unique labels
2. **Label Mapping**: Maps string/numeric labels → integers (0, 1, ...)
3. **Infer Dimensions**: Detects feature dimension from loaded graphs
4. **Partition Data**: Organizes samples into 5 folds
5. **5-Fold Cross-Validation**:
   - For each fold:
     - Fold `i` → test set
     - Fold `(i+1) % 5` → validation set
     - Remaining 3 folds → training set
   - Train model, evaluate on val/test
   - Save best model per fold

### Output Files

- `best_model_fold1.pt`, `best_model_fold2.pt`, ..., `best_model_fold5.pt`: Trained weights
- Console logs: Accuracy per epoch, final test accuracy per fold

## Model Architecture

### GCN Layer
```
Input: x [N, in_features], adj [N, N]
1. Add self-loops to adj
2. Normalize: D^(-1/2) A D^(-1/2)
3. GCN: output = D^(-1/2) A D^(-1/2) * Linear(x)
Output: [N, out_features]
```

### StackedGCN
```
Multiple GCN layers + layer norm + ReLU + dropout
Residual connections between layers (after first layer)
Output: [N, hidden_dim]
```

### AttentionPooling
```
Compute attention scores per node: Linear → Tanh → Linear
Apply softmax → weighted sum over nodes
Output: [hidden_dim] (graph-level embedding)
```

### TemporalAttention
```
Multi-head self-attention over temporal sequence
Key padding mask applied to padded timesteps
Residual connection + layer norm
Output: [T, batch, hidden_dim]
```

### DynamicGCN (Full Model)
```
For each temporal snapshot:
  1. Apply StackedGCN (spatial processing)
  2. Apply AttentionPooling (graph-level aggregation)
Stack temporal embeddings → TemporalAttention
Take last valid timestep + temporal MLP
Classify: Linear → ReLU → Dropout → Linear → logits
```

## GCN Framework Components

### Core Classes

#### `GraphSnapshot`
Represents a single graph at a timestep.
```python
snapshot = GraphSnapshot(
    adjacency=torch.tensor([[0, 1], [1, 0]], dtype=torch.float32),
    node_features=torch.randn(2, 8)  # Optional; defaults to ones
)
```

#### `DynamicGraph`
Represents a temporal sequence of snapshots with a label.
```python
dg = DynamicGraph(
    snapshots=[snapshot1, snapshot2, snapshot3],
    label=0
)
```

#### `GCNLayer`
Single graph convolutional layer.
```python
gcn = GCNLayer(in_features=8, out_features=64)
output = gcn(node_features, adjacency_matrix)
```

#### `StackedGCN`
Multiple GCN layers with layer norm and residuals.
```python
stacked_gcn = StackedGCN(
    input_dim=8,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3
)
output = stacked_gcn(node_features, adjacency_matrix)
```

#### `AttentionPooling`
Graph-level aggregation via attention.
```python
pooler = AttentionPooling(hidden_dim=64)
graph_embedding = pooler(node_features, node_mask)
```

#### `TemporalAttention`
Temporal self-attention over sequence.
```python
temporal_attn = TemporalAttention(hidden_dim=64, num_heads=4)
attended, weights = temporal_attn(sequence, temporal_mask)
```

#### `DynamicGCN`
Complete model combining all components.
```python
model = DynamicGCN(
    input_dim=8,
    hidden_dim=64,
    num_classes=4,
    gcn_layers=2,
    num_attention_heads=4,
    dropout=0.3
)
logits, attention_weights = model(padded_snapshots, temporal_mask)
```

## Training & Evaluation

### train_epoch
Trains for one epoch over a dataloader.
```python
loss, accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
```

### evaluate
Evaluates on a dataloader.
```python
loss, accuracy = evaluate(model, val_loader, criterion, device)
```

### collate_dynamic_graphs
Batches dynamic graphs with padding.
```python
padded_snapshots, temporal_mask, labels = collate_dynamic_graphs(batch)
```

## Example: Custom Usage

```python
import torch
from DGCN import (
    GraphSnapshot, DynamicGraph, DynamicGCN,
    DynamicGraphDataset, collate_dynamic_graphs
)
from torch.utils.data import DataLoader

# Create snapshots
snap1 = GraphSnapshot(
    adjacency=torch.tensor([[0., 1.], [1., 0.]]),
    node_features=torch.randn(2, 8)
)
snap2 = GraphSnapshot(
    adjacency=torch.tensor([[0., 0.], [0., 0.]]),
    node_features=torch.randn(2, 8)
)

# Create dynamic graph
dg = DynamicGraph([snap1, snap2], label=0)

# Create dataset and dataloader
dataset = DynamicGraphDataset([dg])
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_dynamic_graphs)

# Create model
model = DynamicGCN(
    input_dim=8, hidden_dim=64, num_classes=2,
    gcn_layers=2, num_attention_heads=4
)

# Forward pass
for padded_snapshots, temporal_mask, labels in dataloader:
    logits, attn_weights = model(padded_snapshots, temporal_mask)
    print(f"Predictions: {logits}")
    print(f"Attention shape: {attn_weights.shape}")
```
