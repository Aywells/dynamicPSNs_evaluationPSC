import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import math
import os
import glob


# ============================================================================
# 1. GRAPH DATA STRUCTURES
# ============================================================================

class GraphSnapshot:
    """Represents a single graph snapshot at a specific time point."""
    def __init__(self, adjacency: torch.Tensor, node_features: Optional[torch.Tensor] = None):
        """
        Args:
            adjacency: [N, N] adjacency matrix
            node_features: [N, F] node feature matrix (optional)
        """
        self.adjacency = adjacency
        self.num_nodes = adjacency.shape[0]
        
        # If no features provided, use ones
        if node_features is None:
            self.node_features = torch.ones(self.num_nodes, 1)
        else:
            self.node_features = node_features
        
        self.feature_dim = self.node_features.shape[1]


class DynamicGraph:
    """Represents a temporal sequence of graph snapshots."""
    def __init__(self, snapshots: List[GraphSnapshot], label: int):
        """
        Args:
            snapshots: List of GraphSnapshot objects
            label: Classification label for this dynamic graph
        """
        self.snapshots = snapshots
        self.num_snapshots = len(snapshots)
        self.label = label
        
        # Track max nodes for padding
        self.max_nodes = max(s.num_nodes for s in snapshots)
        self.feature_dim = snapshots[0].feature_dim


# ============================================================================
# 2. GCN LAYERS
# ============================================================================

class GCNLayer(nn.Module):
    """Standard Graph Convolutional Layer."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_features] node features
            adj: [N, N] adjacency matrix
        Returns:
            [N, out_features] updated node features
        """
        # Add self-loops
        adj = adj + torch.eye(adj.shape[0], device=adj.device)
        
        # Normalize adjacency matrix: D^(-1/2) A D^(-1/2)
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        # Apply GCN operation
        support = self.linear(x)
        output = adj_normalized @ support
        
        return output


class StackedGCN(nn.Module):
    """Multiple GCN layers with residual connections and layer normalization."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First layer
        self.gcn_layers = nn.ModuleList([GCNLayer(input_dim, hidden_dim)])
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] node features
            adj: [N, N] adjacency matrix
        Returns:
            [N, hidden_dim] updated node features
        """
        h = x
        
        for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.layer_norms)):
            h_new = gcn(h, adj)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Residual connection (except first layer)
            if i > 0 and h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new
        
        return h


# ============================================================================
# 3. ATTENTION MECHANISMS
# ============================================================================

class AttentionPooling(nn.Module):
    """Attention-based graph-level pooling."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_features: [N, hidden_dim] node features
            mask: [N] binary mask (1 for real nodes, 0 for padded)
        Returns:
            [hidden_dim] graph-level embedding
        """
        # Compute attention scores
        scores = self.attention_mlp(node_features)  # [N, 1]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=0)  # [N, 1]
        
        # Weighted sum
        graph_embedding = (attention_weights * node_features).sum(dim=0)  # [hidden_dim]
        
        return graph_embedding


class TemporalAttention(nn.Module):
    """Multi-head self-attention over temporal sequence."""
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # Expects [T, batch, hidden_dim]
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence: [T, batch, hidden_dim] temporal sequence
            mask: [batch, T] padding mask (True for real timesteps, False for padded)
        Returns:
            attended_seq: [T, batch, hidden_dim]
            attention_weights: [batch, T, T]
        """
        # Convert mask: MultiheadAttention expects True for positions to IGNORE
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True = ignore
        else:
            key_padding_mask = None
        
        # Self-attention
        attn_output, attn_weights = self.multihead_attn(
            query=sequence,
            key=sequence,
            value=sequence,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(sequence + self.dropout(attn_output))
        
        return output, attn_weights


# ============================================================================
# 4. MAIN MODEL
# ============================================================================

class DynamicGCN(nn.Module):
    """Complete Dynamic Graph Convolutional Network with multi-level attention."""
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        gcn_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Spatial processing
        self.spatial_gcn = StackedGCN(input_dim, hidden_dim, gcn_layers, dropout)
        
        # Graph-level pooling with attention
        self.graph_pooling = AttentionPooling(hidden_dim)
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim, num_attention_heads, dropout)
        
        # Additional temporal processing layers
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self, 
        padded_snapshots: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        temporal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            padded_snapshots: List of T tuples, each containing:
                - node_features: [batch, max_nodes, feature_dim]
                - adjacency: [batch, max_nodes, max_nodes]
                - node_mask: [batch, max_nodes]
            temporal_mask: [batch, T] - True for real snapshots, False for padded
        Returns:
            logits: [batch, num_classes]
            attention_weights: [batch, T, T]
        """
        batch_size = padded_snapshots[0][0].shape[0]
        num_snapshots = len(padded_snapshots)
        
        # Process each snapshot
        graph_embeddings = []
        
        for t in range(num_snapshots):
            node_features, adjacency, node_mask = padded_snapshots[t]
            # Process each graph in the batch
            batch_embeddings = []
            
            for b in range(batch_size):
                # Extract single graph
                x = node_features[b]  # [max_nodes, feature_dim]
                adj = adjacency[b]    # [max_nodes, max_nodes]
                mask = node_mask[b]   # [max_nodes]
                
                # Apply GCN
                h = self.spatial_gcn(x, adj)  # [max_nodes, hidden_dim]
                
                # Pool to graph-level with attention
                g = self.graph_pooling(h, mask)  # [hidden_dim]
                
                batch_embeddings.append(g)
            
            # Stack batch
            batch_emb = torch.stack(batch_embeddings, dim=0)  # [batch, hidden_dim]
            graph_embeddings.append(batch_emb)
        
        # Create temporal sequence: [T, batch, hidden_dim]
        temporal_sequence = torch.stack(graph_embeddings, dim=0)
        
        # Apply temporal attention
        attended_sequence, attn_weights = self.temporal_attention(
            temporal_sequence, 
            temporal_mask
        )  # [T, batch, hidden_dim], [batch, T, T]
        
        # Aggregate temporal information
        # Get last valid timestep for each sample
        final_representations = []
        for b in range(batch_size):
            # Find last valid timestep
            valid_indices = torch.where(temporal_mask[b])[0]
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1]
                final_repr = attended_sequence[last_valid_idx, b]
            else:
                # Fallback: use first timestep
                final_repr = attended_sequence[0, b]
            final_representations.append(final_repr)
        
        final_repr = torch.stack(final_representations, dim=0)  # [batch, hidden_dim]
        
        # Additional temporal processing
        final_repr = final_repr + self.temporal_mlp(final_repr)
        
        # Classification
        logits = self.classifier(final_repr)  # [batch, num_classes]
        
        return logits, attn_weights


# ============================================================================
# 5. DATASET AND COLLATION
# ============================================================================

class DynamicGraphDataset(Dataset):
    """Dataset for dynamic graphs."""
    def __init__(self, dynamic_graphs: List[DynamicGraph]):
        self.dynamic_graphs = dynamic_graphs
    
    def __len__(self):
        return len(self.dynamic_graphs)
    
    def __getitem__(self, idx):
        return self.dynamic_graphs[idx]


def collate_dynamic_graphs(batch: List[DynamicGraph]) -> Tuple:
    """
    Collate function to batch dynamic graphs with padding.
    
    Args:
        batch: List of DynamicGraph objects
    Returns:
        padded_snapshots: List of T tuples with batched data
        temporal_mask: [batch, max_T]
        labels: [batch]
    """
    batch_size = len(batch)
    
    # Find maximum dimensions
    max_snapshots = max(dg.num_snapshots for dg in batch)
    max_nodes = max(dg.max_nodes for dg in batch)
    feature_dim = batch[0].feature_dim
    
    # Initialize padded structures
    padded_snapshots = []
    
    for t in range(max_snapshots):
        # Batch tensors for this timestep
        batch_node_features = torch.zeros(batch_size, max_nodes, feature_dim)
        batch_adjacency = torch.zeros(batch_size, max_nodes, max_nodes)
        batch_node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        
        for b, dg in enumerate(batch):
            if t < dg.num_snapshots:
                snapshot = dg.snapshots[t]
                n = snapshot.num_nodes
                
                # Copy node features
                batch_node_features[b, :n, :] = snapshot.node_features
                
                # Copy adjacency
                batch_adjacency[b, :n, :n] = snapshot.adjacency
                
                # Set mask
                batch_node_mask[b, :n] = True
        
        padded_snapshots.append((batch_node_features, batch_adjacency, batch_node_mask))
    
    # Create temporal mask
    temporal_mask = torch.zeros(batch_size, max_snapshots, dtype=torch.bool)
    for b, dg in enumerate(batch):
        temporal_mask[b, :dg.num_snapshots] = True
    
    # Extract labels
    labels = torch.tensor([dg.label for dg in batch], dtype=torch.long)
    
    return padded_snapshots, temporal_mask, labels


# ============================================================================
# 6. TRAINING UTILITIES
# ============================================================================

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for padded_snapshots, temporal_mask, labels in dataloader:
        # Move to device
        padded_snapshots = [
            (nf.to(device), adj.to(device), mask.to(device))
            for nf, adj, mask in padded_snapshots
        ]
        temporal_mask = temporal_mask.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(padded_snapshots, temporal_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for padded_snapshots, temporal_mask, labels in dataloader:
            # Move to device
            padded_snapshots = [
                (nf.to(device), adj.to(device), mask.to(device))
                for nf, adj, mask in padded_snapshots
            ]
            temporal_mask = temporal_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, _ = model(padded_snapshots, temporal_mask)
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


# ============================================================================
# 7. SYNTHETIC DATA GENERATION (FOR DEMONSTRATION)
# ============================================================================

def generate_synthetic_dynamic_graph(
    num_snapshots: int,
    num_nodes_range: Tuple[int, int],
    feature_dim: int,
    label: int,
    edge_prob: float = 0.3
) -> DynamicGraph:
    """Generate a synthetic dynamic graph for testing."""
    snapshots = []
    
    for t in range(num_snapshots):
        # Random number of nodes
        num_nodes = np.random.randint(num_nodes_range[0], num_nodes_range[1] + 1)
        
        # Random adjacency matrix
        adj = (torch.rand(num_nodes, num_nodes) < edge_prob).float()
        adj = (adj + adj.T) / 2  # Make symmetric
        adj.fill_diagonal_(0)    # Remove self-loops
        
        # Random node features
        node_features = torch.randn(num_nodes, feature_dim)
        
        snapshot = GraphSnapshot(adj, node_features)
        snapshots.append(snapshot)
    
    return DynamicGraph(snapshots, label)


def create_synthetic_dataset(
    num_samples: int,
    num_classes: int,
    snapshot_range: Tuple[int, int],
    node_range: Tuple[int, int],
    feature_dim: int
) -> List[DynamicGraph]:
    """Create a synthetic dataset."""
    dataset = []
    
    for _ in range(num_samples):
        num_snapshots = np.random.randint(snapshot_range[0], snapshot_range[1] + 1)
        label = np.random.randint(0, num_classes)
        
        dg = generate_synthetic_dynamic_graph(
            num_snapshots=num_snapshots,
            num_nodes_range=node_range,
            feature_dim=feature_dim,
            label=label
        )
        dataset.append(dg)
    
    return dataset


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    FEATURE_DIM = 8
    HIDDEN_DIM = 64
    NUM_CLASSES = 4
    GCN_LAYERS = 2
    NUM_HEADS = 4
    DROPOUT = 0.3
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # -----------------------------
    # User dataset / feature options
    # -----------------------------
    # Set the dataset txt filename located in the `datasets` directory.
    # Example: DATASET_TXT = 'mydataset.txt'
    DATASET_TXT = 'scop-g.txt'  # <-- set to dataset file name (or leave empty for synthetic)

    # Feature mode: 'default' to use GraphSnapshot default features, 'graphlet' to load graphlet features
    FEATURE_MODE = 'default'  # options: 'default' | 'graphlet'

    # Directories (relative to current working directory)
    DATASETS_DIR = 'datasets'
    NETWORKS_DIR = 'dynamic-networks'
    GRAPHLETS_DIR = 'dynamic-graphlets'
    PARTITIONS_DIR = 'partitions'

    def read_dataset_txt(path: str) -> List[Tuple[str, str]]:
        """Read dataset txt where each line has: <label> <sample_name>
        Labels are returned as raw strings (not converted to ints).
        """
        entries = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    label = parts[0]
                    sample_name = parts[1]
                    entries.append((label, sample_name))
                else:
                    # try comma separated
                    parts = line.split(',')
                    if len(parts) >= 2:
                        label = parts[0]
                        sample_name = parts[1]
                        entries.append((label, sample_name))
        return entries

    def load_adjacency_from_file(fp: str):
        name = fp.lower()
        try:
            if name.endswith('.npy'):
                arr = np.load(fp)
                return arr
            if name.endswith('.npz'):
                data = np.load(fp)
                # take first array inside
                for v in data.files:
                    return data[v]
            # try loadtxt
            arr = np.loadtxt(fp)
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                return arr
            # If two columns, treat as edge list
            if arr.ndim == 2 and arr.shape[1] == 2:
                # edge list
                edges = arr.astype(int)
                nodes = np.unique(edges)
                idx = {n: i for i, n in enumerate(nodes)}
                N = len(nodes)
                A = np.zeros((N, N), dtype=float)
                for u, v in edges:
                    i, j = idx[u], idx[v]
                    A[i, j] = 1
                    A[j, i] = 1
                return A
        except Exception:
            return None

    def load_graphlet_features(sample_name: str):
        # find files containing sample_name and 'dcgdv'
        pattern = os.path.join(GRAPHLETS_DIR, f"*{sample_name}*dcgdv*")
        files = glob.glob(pattern)
        if not files:
            return None
        fp = files[0]
        try:
            if fp.lower().endswith('.npy'):
                return np.load(fp)
            if fp.lower().endswith('.npz'):
                data = np.load(fp)
                for v in data.files:
                    return data[v]
            return np.loadtxt(fp)
        except Exception:
            return None

    def load_dynamic_graph(sample_name: str, label: int):
        sample_dir = os.path.join(NETWORKS_DIR, sample_name)
        if not os.path.isdir(sample_dir):
            return None

        # collect snapshot files sorted
        files = sorted(glob.glob(os.path.join(sample_dir, '*')))
        snapshots = []

        for fp in files:
            A = load_adjacency_from_file(fp)
            if A is None:
                continue
            # ensure square
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                continue
            N = A.shape[0]
            # choose node features
            if FEATURE_MODE == 'graphlet':
                feat = load_graphlet_features(sample_name)
                if feat is not None:
                    # if feature rows match nodes
                    if feat.ndim == 1:
                        feat = feat.reshape(-1, 1)
                    if feat.shape[0] == N:
                        node_features = torch.tensor(feat, dtype=torch.float32)
                    elif feat.shape[1] == N:
                        node_features = torch.tensor(feat.T, dtype=torch.float32)
                    else:
                        node_features = None
                else:
                    node_features = None
            else:
                node_features = None

            adj_t = torch.tensor(A, dtype=torch.float32)
            snapshot = GraphSnapshot(adj_t, node_features)
            snapshots.append(snapshot)

        if not snapshots:
            return None

        return DynamicGraph(snapshots, label)

    def load_partitions(dataset_basename: str):
        part_dir = os.path.join(PARTITIONS_DIR, dataset_basename)
        folds = []
        if not os.path.isdir(part_dir):
            return None
        for i in range(1, 6):
            fp = os.path.join(part_dir, str(i)+".txt")
            if not os.path.isfile(fp):
                folds.append([])
                continue
            with open(fp, 'r') as f:
                names = [line.strip().split()[0] for line in f if line.strip()]
            folds.append(names)
        return folds

    def build_dataset_from_sample_names(sample_names: List[str], label_map: dict):
        dlist = []
        for s in sample_names:
            if s not in label_map:
                print(f"Warning: sample {s} not found in dataset txt; skipping")
                continue
            dg = load_dynamic_graph(s, label_map[s])
            if dg is None:
                print(f"Skipping sample {s}: network or snapshots not found or invalid")
                continue
            dlist.append(dg)
        return dlist
    
    # If user provided a dataset txt, try to load real data and partitions
    if DATASET_TXT:
        ds_path = os.path.join(DATASETS_DIR, DATASET_TXT)
        if not os.path.isfile(ds_path):
            print(f"Dataset file not found: {ds_path}. Falling back to synthetic data.")
            DATASET_TXT = ''
        else:
            entries = read_dataset_txt(ds_path)
            # Build string label to sample name map
            label_to_samples = {}
            for label_str, sample_name in entries:
                if label_str not in label_to_samples:
                    label_to_samples[label_str] = []
                label_to_samples[label_str].append(sample_name)
            
            # Create unique label to int mapping
            unique_labels = sorted(label_to_samples.keys())
            str_label_to_int = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            NUM_CLASSES = len(unique_labels)
            print(f"Found {NUM_CLASSES} unique classes: {unique_labels}")
            
            # Create sample name to int label map
            label_map = {}
            for label_str, sample_name in entries:
                label_map[sample_name] = str_label_to_int[label_str]
            
            dataset_basename = os.path.splitext(DATASET_TXT)[0]
            folds = load_partitions(dataset_basename)

            if folds is None:
                print(f"Partitions directory not found for dataset {dataset_basename}. Falling back to synthetic data.")
                DATASET_TXT = ''
            else:
                # Build lists per fold
                fold_datasets = []
                for i, names in enumerate(folds):
                    dlist = build_dataset_from_sample_names(names, label_map)
                    fold_datasets.append(dlist)
                    print(f"Fold {i+1}: prepared {len(dlist)} samples")

                # Infer feature dimension from loaded data
                inferred_feature_dim = None
                for fold_list in fold_datasets:
                    if fold_list:
                        dg = fold_list[0]
                        inferred_feature_dim = dg.feature_dim
                        break
                
                if inferred_feature_dim is not None and inferred_feature_dim != FEATURE_DIM:
                    print(f"Updating FEATURE_DIM from {FEATURE_DIM} to {inferred_feature_dim} (detected from real data)")
                    FEATURE_DIM = inferred_feature_dim
                else:
                    print(f"Using FEATURE_DIM = {FEATURE_DIM}")

                # 5-fold cross validation
                print("\nStarting 5-fold cross-validation...")
                for fold_idx in range(5):
                    test_idx = fold_idx
                    val_idx = (fold_idx + 1) % 5
                    train_idxs = [i for i in range(5) if i not in (test_idx, val_idx)]

                    train_list = []
                    for ti in train_idxs:
                        train_list.extend(fold_datasets[ti])
                    val_list = fold_datasets[val_idx]
                    test_list = fold_datasets[test_idx]

                    print(f"Fold {fold_idx+1}: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

                    if not train_list or not val_list:
                        print("Not enough data for this fold; skipping")
                        continue

                    train_loader = DataLoader(
                        DynamicGraphDataset(train_list),
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        collate_fn=collate_dynamic_graphs
                    )

                    val_loader = DataLoader(
                        DynamicGraphDataset(val_list),
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        collate_fn=collate_dynamic_graphs
                    )

                    test_loader = DataLoader(
                        DynamicGraphDataset(test_list),
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        collate_fn=collate_dynamic_graphs
                    )

                    # initialize model per-fold
                    model = DynamicGCN(
                        input_dim=FEATURE_DIM,
                        hidden_dim=HIDDEN_DIM,
                        num_classes=NUM_CLASSES,
                        gcn_layers=GCN_LAYERS,
                        num_attention_heads=NUM_HEADS,
                        dropout=DROPOUT
                    ).to(device)

                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=5
                    )

                    best_val_acc = 0.0
                    for epoch in range(EPOCHS):
                        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                        scheduler.step(val_loss)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            torch.save(model.state_dict(), f'best_model_fold{fold_idx+1}.pt')

                        if (epoch + 1) % 5 == 0:
                            print(f"Fold {fold_idx+1} Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")

                    # Evaluate on test set using best model
                    model.load_state_dict(torch.load(f'best_model_fold{fold_idx+1}.pt'))
                    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                    print(f"Fold {fold_idx+1} Test Acc: {test_acc:.2f}%")

                print("\n5-fold cross-validation complete")

    # If DATASET_TXT was empty or fell back, use synthetic dataset as before
    if not DATASET_TXT:
        print("\nGenerating synthetic dataset...")
        train_data = create_synthetic_dataset(
            num_samples=200,
            num_classes=NUM_CLASSES,
            snapshot_range=(3, 10),
            node_range=(10, 30),
            feature_dim=FEATURE_DIM
        )
        
        val_data = create_synthetic_dataset(
            num_samples=50,
            num_classes=NUM_CLASSES,
            snapshot_range=(3, 10),
            node_range=(10, 30),
            feature_dim=FEATURE_DIM
        )
        
        print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Create dataloaders
    train_dataset = DynamicGraphDataset(train_data)
    val_dataset = DynamicGraphDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_dynamic_graphs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_dynamic_graphs
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = DynamicGCN(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        gcn_layers=GCN_LAYERS,
        num_attention_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model and demonstrate inference
    print("\nDemonstrating inference with attention visualization...")
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    
    # Get one sample
    sample_dg = val_data[0]
    print(f"\nSample dynamic graph:")
    print(f"  Number of snapshots: {sample_dg.num_snapshots}")
    print(f"  Nodes per snapshot: {[s.num_nodes for s in sample_dg.snapshots]}")
    print(f"  True label: {sample_dg.label}")
    
    # Prepare single sample batch
    padded_snapshots, temporal_mask, labels = collate_dynamic_graphs([sample_dg])
    padded_snapshots = [(nf.to(device), adj.to(device), mask.to(device)) 
                        for nf, adj, mask in padded_snapshots]
    temporal_mask = temporal_mask.to(device)
    
    # Inference
    with torch.no_grad():
        logits, attn_weights = model(padded_snapshots, temporal_mask)
        probs = F.softmax(logits, dim=1)
        predicted = logits.argmax(1).item()
    
    print(f"  Predicted label: {predicted}")
    print(f"  Class probabilities: {probs[0].cpu().numpy()}")
    print(f"\nTemporal attention weights shape: {attn_weights.shape}")
    print(f"Attention weights (first 5x5):\n{attn_weights[0, :5, :5].cpu().numpy()}")


if __name__ == "__main__":
    main()