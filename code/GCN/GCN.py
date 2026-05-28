#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

LOG_FILE_HANDLE = None
SKIPPED_SAMPLES: List[Dict[str, str]] = []

def init_logger(log_path: Path) -> None:
    global LOG_FILE_HANDLE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE_HANDLE = open(log_path, "w", encoding="utf-8")

def close_logger() -> None:
    global LOG_FILE_HANDLE
    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.close()
        LOG_FILE_HANDLE = None

def _write_log(prefix: str, msg: str) -> None:
    text = f"{prefix} {msg}"
    print(text, flush=True)
    global LOG_FILE_HANDLE
    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.write(text + "\n")
        LOG_FILE_HANDLE.flush()

def log(msg: str) -> None:
    _write_log("[INFO]", msg)

def warn(msg: str) -> None:
    _write_log("[WARN]", msg)

def step(msg: str) -> None:
    _write_log("\n==========", f"{msg} ==========")

def record_skipped(sample_name: str, reason: str) -> None:
    global SKIPPED_SAMPLES
    SKIPPED_SAMPLES.append({"sample_name": sample_name, "reason": reason})
    warn(f"Invalid sample '{sample_name}': {reason}")

def write_skipped_samples_file(cfg: "Config") -> None:
    out_fp = output_dir(cfg) / "skipped_samples.txt"
    with open(out_fp, "w", encoding="utf-8") as f:
        f.write("sample_name\treason\n")
        for row in SKIPPED_SAMPLES:
            f.write(f"{row['sample_name']}\t{row['reason']}\n")
    log(f"Wrote skipped_samples.txt with {len(SKIPPED_SAMPLES)} problematic samples.")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class Config:
    root_dir: str
    dataset_name: str
    model_type: str # "sgcn" or "dgcn"
    feature_mode: str # "default" or "dgdvms"
    partitions_mode: str # "user" or "auto"

    max_nodes: Optional[int] = None
    default_feature_dim: Optional[int] = None
    batch_size: int = 1
    epochs: int = 100
    patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    outer_folds: int = 5
    inner_folds: int = 5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

    lr_grid: Tuple[float, ...] = (1e-3,)
    dropout_grid: Tuple[float, ...] = (0.3,)
    weight_decay_grid: Tuple[float, ...] = (0.0,)

    save_models: bool = False
    verbose_epochs: bool = True

def dataset_dir(cfg: Config) -> Path:
    return Path(cfg.root_dir) / "datasets" / cfg.dataset_name

def output_dir(cfg: Config) -> Path:
    return dataset_dir(cfg) / "output"

def normalize_name(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]", "", stem)

def build_dgdvm_index(dgdvm_dir: Path) -> Dict[str, List[Path]]:
    log(f"Indexing dGDVM files in: {dgdvm_dir}")
    index: Dict[str, List[Path]] = {}
    files = sorted(dgdvm_dir.glob("*.txt"))
    log(f"Found {len(files)} dGDVM txt files.")
    for fp in files:
        key = normalize_name(fp.name)
        index.setdefault(key, []).append(fp)
    return index

def match_dgdvm_file(sample_name: str, dgdvm_index: Dict[str, List[Path]]) -> Path:
    target = normalize_name(sample_name)

    if target in dgdvm_index and len(dgdvm_index[target]) == 1:
        return dgdvm_index[target][0]

    candidates = []
    for key, files in dgdvm_index.items():
        if target in key or key in target:
            candidates.extend(files)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Could not match dGDVM file for sample '{sample_name}'."
        )

    raise ValueError(
        f"Ambiguous dGDVM match for sample '{sample_name}'. Candidates: {[c.name for c in candidates]}"
    )

def read_label_file(label_fp: Path) -> Tuple[List[str], List[str]]:
    log(f"Reading dataset label file: {label_fp}")
    if not label_fp.exists():
        raise FileNotFoundError(f"Dataset label file not found: {label_fp}")

    class_labels: List[str] = []
    sample_names: List[str] = []

    with open(label_fp, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Expected at least 2 columns in {label_fp} at line {line_num}."
                )
            class_labels.append(parts[0])
            sample_names.append(parts[1])

    log(f"Loaded {len(sample_names)} samples from label file.")
    return class_labels, sample_names

def make_class_mapping(class_labels: List[str]) -> Tuple[List[int], Dict[str, int]]:
    unique = sorted(set(class_labels))
    mapping = {lab: i for i, lab in enumerate(unique)}
    y = [mapping[lab] for lab in class_labels]
    log(f"Detected {len(unique)} unique classes.")
    return y, mapping

def read_edge_list(snapshot_fp: Path) -> torch.Tensor:
    edges: List[Tuple[int, int]] = []
    nodes = set()

    with open(snapshot_fp, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid edge line in {snapshot_fp} at line {line_num}: '{line}'"
                )
            u = int(float(parts[0]))
            v = int(float(parts[1]))
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)

    if len(nodes) == 0:
        raise ValueError(f"No edges/nodes found in snapshot file: {snapshot_fp}")

    min_node = min(nodes)
    max_node = max(nodes)

    if min_node == 1:
        edges = [(u - 1, v - 1) for (u, v) in edges]
        max_node -= 1
    elif min_node != 0:
        sorted_nodes = sorted(nodes)
        remap = {old: new for new, old in enumerate(sorted_nodes)}
        edges = [(remap[u], remap[v]) for (u, v) in edges]
        max_node = len(sorted_nodes) - 1

    n = max_node + 1
    A = torch.zeros(n, n, dtype=torch.float32)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A

def read_psn_snapshots(sample_psn_dir: Path) -> torch.Tensor:
    if not sample_psn_dir.exists():
        raise FileNotFoundError(f"PSN sample directory not found: {sample_psn_dir}")

    snapshot_files = sorted(
        sample_psn_dir.glob("*.txt"),
        key=lambda p: int(re.sub(r"[^\d]", "", p.stem)) if re.search(r"\d", p.stem) else p.stem
    )

    if len(snapshot_files) == 0:
        raise ValueError(f"No snapshot files found in {sample_psn_dir}")

    mats = [read_edge_list(fp) for fp in snapshot_files]
    max_n = max(m.shape[0] for m in mats)

    padded = []
    for A in mats:
        n = A.shape[0]
        if n == max_n:
            padded.append(A)
        else:
            P = torch.zeros(max_n, max_n, dtype=torch.float32)
            P[:n, :n] = A
            padded.append(P)

    return torch.stack(padded, dim=0)

def read_dgdvm_txt(dgdvm_fp: Path) -> torch.Tensor:
    rows = []
    with open(dgdvm_fp, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            rows.append(vals)

    if len(rows) == 0:
        raise ValueError(f"Empty dGDVM file: {dgdvm_fp}")

    return torch.tensor(rows, dtype=torch.float32)

def build_samples_from_disk(cfg: Config) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    step("LOADING DATASET FROM DISK")

    ddir = dataset_dir(cfg)
    log(f"Dataset directory: {ddir}")
    if not ddir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {ddir}")

    label_fp = ddir / f"{cfg.dataset_name}.txt"
    class_labels_raw, sample_names = read_label_file(label_fp)
    y, class_mapping = make_class_mapping(class_labels_raw)

    psn_root = ddir / "psns"
    dgdvm_root = ddir / "dgdvms"

    if not psn_root.exists():
        raise FileNotFoundError(f"Missing PSNs folder: {psn_root}")

    dgdvm_index = build_dgdvm_index(dgdvm_root) if dgdvm_root.exists() else {}

    samples: List[Dict[str, Any]] = []
    loaded_count = 0

    for i, (sample_name, label) in enumerate(zip(sample_names, y), start=1):
        if i % 25 == 0 or i == len(sample_names):
            log(f"Processing sample {i}/{len(sample_names)}: {sample_name}")

        try:
            sample_psn_dir = psn_root / sample_name
            adjs = read_psn_snapshots(sample_psn_dir)
            T, n, _ = adjs.shape

            sample: Dict[str, Any] = {
                "sample_name": sample_name,
                "label": int(label),
                "adjs": adjs,
            }

            if cfg.feature_mode == "dgdvms":
                if not dgdvm_root.exists():
                    raise FileNotFoundError(f"dgdvms folder not found: {dgdvm_root}")

                dgdvm_fp = match_dgdvm_file(sample_name, dgdvm_index)
                feat = read_dgdvm_txt(dgdvm_fp)

                if feat.shape[0] != n:
                    if feat.shape[0] > n:
                        feat = feat[:n, :]
                    else:
                        pad = torch.zeros(n - feat.shape[0], feat.shape[1], dtype=torch.float32)
                        feat = torch.cat([feat, pad], dim=0)

                feats = feat.unsqueeze(0).repeat(T, 1, 1)
                sample["features"] = feats

            samples.append(sample)
            loaded_count += 1

        except Exception as e:
            record_skipped(sample_name, str(e))

    log(f"Finished loading dataset '{cfg.dataset_name}'.")
    log(f"Loaded samples: {loaded_count}")
    log(f"Problematic samples: {len(SKIPPED_SAMPLES)}")

    if len(SKIPPED_SAMPLES) > 0:
        raise RuntimeError(
            f"Encountered {len(SKIPPED_SAMPLES)} problematic samples while loading the dataset. "
            f"See skipped_samples.txt and run.log for details."
        )

    if loaded_count == 0:
        raise RuntimeError("No valid samples could be loaded.")

    return samples, class_mapping

def parse_partition_file(fp: Path, sample_names: List[str]) -> List[int]:
    if not fp.exists():
        raise FileNotFoundError(f"Partition file not found: {fp}")

    name_to_idx = {name: i for i, name in enumerate(sample_names)}
    lines = []

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)

    if len(lines) == 0:
        raise ValueError(f"Empty partition file: {fp}")

    if all(line in name_to_idx for line in lines):
        return [name_to_idx[line] for line in lines]

    vals = [int(x) for x in lines]
    if min(vals) == 1:
        vals = [v - 1 for v in vals]

    max_idx = len(sample_names) - 1
    if not all(0 <= v <= max_idx for v in vals):
        raise ValueError(
            f"Partition file {fp} contains indices outside valid range 0..{max_idx}"
        )
    return vals

def load_user_outer_folds(cfg: Config, sample_names: List[str]) -> List[Dict[str, List[int]]]:
    step("LOADING USER PARTITIONS")

    part_dir = dataset_dir(cfg) / "partitions"
    if not part_dir.exists():
        raise FileNotFoundError(f"Missing partitions folder: {part_dir}")

    fold_test_sets = []
    for k in range(1, 6):
        txt = part_dir / f"{k}.txt"
        if not txt.exists():
            txt = part_dir / str(k)
        log(f"Reading partition file: {txt}")
        fold_test_sets.append(parse_partition_file(txt, sample_names))

    all_idx = set(range(len(sample_names)))
    outer_folds = []
    for i, test_idx in enumerate(fold_test_sets, start=1):
        test_set = sorted(set(test_idx))
        train_set = sorted(all_idx - set(test_set))
        log(f"Partition fold {i}: train={len(train_set)}, test={len(test_set)}")
        outer_folds.append({"train": train_set, "test": test_set})

    return outer_folds

def make_auto_outer_folds(labels: np.ndarray, n_splits: int, seed: int) -> List[Dict[str, List[int]]]:
    step("GENERATING OUTER CV PARTITIONS")
    log(f"Generating {n_splits} stratified outer folds.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_idx = np.arange(len(labels))
    folds = []
    for i, (tr, te) in enumerate(skf.split(all_idx, labels), start=1):
        log(f"Generated outer fold {i}: train={len(tr)}, test={len(te)}")
        folds.append({"train": tr.tolist(), "test": te.tolist()})
    return folds

def make_auto_inner_folds(
    train_indices: Sequence[int],
    labels: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Dict[str, List[int]]]:
    train_indices = np.array(train_indices, dtype=np.int64)
    train_labels = labels[train_indices]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner = []
    for tr_pos, va_pos in skf.split(train_indices, train_labels):
        inner.append({
            "train": train_indices[tr_pos].tolist(),
            "val": train_indices[va_pos].tolist(),
        })
    return inner

def save_generated_partitions(
    cfg: Config,
    sample_names: List[str],
    outer_folds: List[Dict[str, List[int]]],
) -> None:
    step("SAVING GENERATED PARTITIONS")
    out_dir = output_dir(cfg) / "generated_partitions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, fold in enumerate(outer_folds, start=1):
        fp = out_dir / f"{i}.txt"
        with open(fp, "w", encoding="utf-8") as f:
            for idx in fold["test"]:
                f.write(sample_names[idx] + "\n")
        log(f"Saved generated fold {i} to {fp}")

class PSNDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], indices: Sequence[int], cfg: Config) -> None:
        self.samples = samples
        self.indices = list(indices)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.indices)

    def _make_default_features(self, T: int, n: int, global_index: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(self.cfg.seed + int(global_index))
        return torch.randn(T, n, self.cfg.default_feature_dim, generator=g, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        gidx = self.indices[idx]
        s = self.samples[gidx]

        adjs = s["adjs"].float()
        T, n, _ = adjs.shape

        if n > self.cfg.max_nodes:
            raise ValueError(
                f"Sample '{s['sample_name']}' has {n} nodes, exceeding max_nodes={self.cfg.max_nodes}."
            )

        if self.cfg.feature_mode == "dgdvms":
            feats = s["features"].float()
        else:
            feats = self._make_default_features(T=T, n=n, global_index=gidx)

        if self.cfg.model_type == "sgcn":
            return {
                "sample_name": s["sample_name"],
                "adj": adjs[-1],
                "feat": feats[-1],
                "num_nodes": n,
                "label": int(s["label"]),
            }

        return {
            "sample_name": s["sample_name"],
            "adjs": adjs,
            "feats": feats,
            "num_nodes": n,
            "num_snapshots": T,
            "label": int(s["label"]),
        }


def collate_sgcn(batch: List[Dict[str, Any]], max_nodes: int) -> Dict[str, Any]:
    B = len(batch)
    feat_dim = batch[0]["feat"].shape[-1]

    A = torch.zeros(B, max_nodes, max_nodes, dtype=torch.float32)
    X = torch.zeros(B, max_nodes, feat_dim, dtype=torch.float32)
    node_mask = torch.zeros(B, max_nodes, dtype=torch.bool)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    names = [b["sample_name"] for b in batch]

    for i, b in enumerate(batch):
        n = b["num_nodes"]
        A[i, :n, :n] = b["adj"]
        X[i, :n, :] = b["feat"]
        node_mask[i, :n] = True

    return {"A": A, "X": X, "node_mask": node_mask, "labels": labels, "sample_names": names}


def collate_dgcn(batch: List[Dict[str, Any]], max_nodes: int) -> Dict[str, Any]:
    B = len(batch)
    max_T = max(b["num_snapshots"] for b in batch)
    feat_dim = batch[0]["feats"].shape[-1]

    A_seq = torch.zeros(B, max_T, max_nodes, max_nodes, dtype=torch.float32)
    X_seq = torch.zeros(B, max_T, max_nodes, feat_dim, dtype=torch.float32)
    node_mask = torch.zeros(B, max_T, max_nodes, dtype=torch.bool)
    time_mask = torch.zeros(B, max_T, dtype=torch.bool)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    names = [b["sample_name"] for b in batch]

    for i, b in enumerate(batch):
        T = b["num_snapshots"]
        n = b["num_nodes"]
        A_seq[i, :T, :n, :n] = b["adjs"]
        X_seq[i, :T, :n, :] = b["feats"]
        node_mask[i, :T, :n] = True
        time_mask[i, :T] = True

    return {
        "A_seq": A_seq,
        "X_seq": X_seq,
        "node_mask": node_mask,
        "time_mask": time_mask,
        "labels": labels,
        "sample_names": names,
    }


def make_dataloader(samples: List[Dict[str, Any]], indices: Sequence[int], cfg: Config, shuffle: bool) -> DataLoader:
    ds = PSNDataset(samples=samples, indices=indices, cfg=cfg)
    collate_fn = (
        (lambda batch: collate_sgcn(batch, cfg.max_nodes))
        if cfg.model_type == "sgcn"
        else (lambda batch: collate_dgcn(batch, cfg.max_nodes))
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

def prepare_adjacency(A: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, N, _ = A.shape
    I = torch.eye(N, dtype=A.dtype, device=A.device).unsqueeze(0).expand(B, N, N)
    A_tilde = A + I
    if node_mask is not None:
        valid = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        A_tilde = A_tilde * valid.to(A.dtype)
    return A_tilde


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, A: torch.Tensor, H: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        A_tilde = prepare_adjacency(A, node_mask=node_mask)
        H = A_tilde @ H
        H = self.linear(H)
        H = self.norm(H)
        H = F.relu(H)
        H = self.dropout(H)
        if node_mask is not None:
            H = H * node_mask.unsqueeze(-1).to(H.dtype)
        return H


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, H: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        e = self.fc2(torch.tanh(self.fc1(H)))
        e = e.masked_fill(~node_mask.unsqueeze(-1), -1e9)
        alpha = torch.softmax(e, dim=1)
        return torch.sum(alpha * H, dim=1)


class DGCN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, 64, dropout=dropout)
        self.gcn2 = GCNLayer(64, 64, dropout=dropout)
        self.pool = AttentionPooling(hidden_dim=64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=False,
        )
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, A_seq, X_seq, node_mask, time_mask):
        B, T, _, _ = A_seq.shape
        pooled = []

        for t in range(T):
            H = self.gcn1(A_seq[:, t], X_seq[:, t], node_mask[:, t])
            H = self.gcn2(A_seq[:, t], H, node_mask[:, t])
            pooled.append(self.pool(H, node_mask[:, t]))

        G = torch.stack(pooled, dim=1).transpose(0, 1).contiguous()
        lstm_out, _ = self.lstm(G)
        lstm_out = lstm_out.transpose(0, 1)

        lengths = torch.clamp(time_mask.long().sum(dim=1) - 1, min=0)
        final = torch.stack([lstm_out[i, lengths[i], :] for i in range(B)], dim=0)

        x = self.fc1(final)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class SGCN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, 64, dropout=dropout)
        self.gcn2 = GCNLayer(64, 64, dropout=dropout)
        self.pool = AttentionPooling(hidden_dim=64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, A, X, node_mask):
        H = self.gcn1(A, X, node_mask)
        H = self.gcn2(A, H, node_mask)
        g = self.pool(H, node_mask)
        x = self.fc1(g)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


def build_model(cfg: Config, input_dim: int, num_classes: int, dropout: float) -> nn.Module:
    log(
        f"Building model: {cfg.model_type.upper()} with input_dim={input_dim}, "
        f"num_classes={num_classes}, dropout={dropout}"
    )
    if cfg.model_type == "sgcn":
        return SGCN(input_dim=input_dim, num_classes=num_classes, dropout=dropout).to(cfg.device)
    return DGCN(input_dim=input_dim, num_classes=num_classes, dropout=dropout).to(cfg.device)

def move_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: Config,
    optimizer: Optional[torch.optim.Optimizer],
) -> Tuple[float, float, List[Dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    total_wrong = 0
    raw_rows: List[Dict[str, Any]] = []

    for batch in dataloader:
        batch = move_batch(batch, cfg.device)
        labels = batch["labels"]

        if training:
            optimizer.zero_grad()

        if cfg.model_type == "sgcn":
            logits = model(A=batch["A"], X=batch["X"], node_mask=batch["node_mask"])
        else:
            logits = model(
                A_seq=batch["A_seq"],
                X_seq=batch["X_seq"],
                node_mask=batch["node_mask"],
                time_mask=batch["time_mask"],
            )

        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        wrong = (preds != labels).sum().item()

        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        total_wrong += wrong

        for i, name in enumerate(batch["sample_names"]):
            raw_rows.append({
                "sample_name": name,
                "true_label": int(labels[i].item()),
                "pred_label": int(preds[i].item()),
                "correct": int(preds[i].item() == labels[i].item()),
                "probabilities": probs[i].detach().cpu().tolist(),
            })

    avg_loss = total_loss / max(total_count, 1)
    misclass = total_wrong / max(total_count, 1)
    return avg_loss, misclass, raw_rows

def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    lr: float,
    weight_decay: float,
) -> Tuple[nn.Module, Dict[str, Any]]:
    log(f"Training with lr={lr}, weight_decay={weight_decay}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_misclass = math.inf
    best_val_loss = math.inf
    best_epoch = -1
    patience_counter = 0

    for epoch in range(cfg.epochs):
        train_loss, train_mis, _ = run_one_epoch(model, train_loader, cfg, optimizer)
        val_loss, val_mis, _ = run_one_epoch(model, val_loader, cfg, optimizer=None)
        scheduler.step(val_loss)

        if cfg.verbose_epochs:
            log(
                f"Epoch {epoch + 1}/{cfg.epochs} | "
                f"train_loss={train_loss:.6f}, train_mis={train_mis:.6f} | "
                f"val_loss={val_loss:.6f}, val_mis={val_mis:.6f}"
            )

        if val_mis < best_val_misclass:
            best_val_misclass = val_mis
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            log(f"New best validation misclassification: {best_val_misclass:.6f} at epoch {epoch + 1}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            log(f"Early stopping triggered at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    return model, {
        "best_val_misclass": best_val_misclass,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }

def tune_hyperparameters(
    cfg: Config,
    samples: List[Dict[str, Any]],
    labels: np.ndarray,
    outer_train_idx: Sequence[int],
    input_dim: int,
    num_classes: int,
) -> Dict[str, Any]:
    step("HYPERPARAMETER TUNING")

    inner_folds = make_auto_inner_folds(
        train_indices=outer_train_idx,
        labels=labels,
        n_splits=cfg.inner_folds,
        seed=cfg.seed,
    )
    log(f"Generated {len(inner_folds)} inner folds for hyperparameter tuning.")

    grid = [(lr, dr, wd) for lr in cfg.lr_grid for dr in cfg.dropout_grid for wd in cfg.weight_decay_grid]
    log(f"Total hyperparameter combinations: {len(grid)}")

    best_params = None
    best_score = math.inf
    all_scores = []

    for combo_id, (lr, dropout, wd) in enumerate(grid, start=1):
        log(f"Testing hyperparameter combo {combo_id}/{len(grid)}: lr={lr}, dropout={dropout}, weight_decay={wd}")
        fold_scores = []

        for inner_id, inner in enumerate(inner_folds, start=1):
            log(f"  Inner fold {inner_id}/{len(inner_folds)}")
            tr_loader = make_dataloader(samples, inner["train"], cfg, shuffle=True)
            va_loader = make_dataloader(samples, inner["val"], cfg, shuffle=False)

            model = build_model(cfg, input_dim=input_dim, num_classes=num_classes, dropout=dropout)
            _, hist = train_with_early_stopping(
                model=model,
                train_loader=tr_loader,
                val_loader=va_loader,
                cfg=cfg,
                lr=lr,
                weight_decay=wd,
            )
            fold_scores.append(hist["best_val_misclass"])

        mean_score = float(np.mean(fold_scores))
        log(f"  Mean inner validation misclassification: {mean_score:.6f}")

        all_scores.append({
            "lr": lr,
            "dropout": dropout,
            "weight_decay": wd,
            "mean_inner_val_misclass": mean_score,
            "fold_scores": fold_scores,
        })

        if mean_score < best_score:
            best_score = mean_score
            best_params = {
                "lr": lr,
                "dropout": dropout,
                "weight_decay": wd,
                "mean_inner_val_misclass": mean_score,
            }
            log("  -> New best hyperparameter setting found.")

    return {"best": best_params, "all_trials": all_scores}

def run_experiment(cfg: Config) -> None:
    global SKIPPED_SAMPLES
    SKIPPED_SAMPLES = []

    out_dir = output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    init_logger(out_dir / "run.log")

    try:
        step("STARTING EXPERIMENT")
        log(f"Dataset: {cfg.dataset_name}")
        log(f"Model type: {cfg.model_type}")
        log(f"Feature mode: {cfg.feature_mode}")
        log(f"Partitions mode: {cfg.partitions_mode}")
        log(f"Device: {cfg.device}")
        log(f"Output directory: {out_dir}")

        set_seed(cfg.seed)

        samples, class_mapping = build_samples_from_disk(cfg)

        largest_psn_size = infer_largest_psn_size(samples)

        if cfg.max_nodes is None:
            cfg.max_nodes = largest_psn_size
            log(f"Inferred max_nodes={cfg.max_nodes} from largest PSN in dataset.")
        else:
            log(f"Using user-provided max_nodes={cfg.max_nodes}.")

        if cfg.default_feature_dim is None:
            cfg.default_feature_dim = largest_psn_size
            log(
                f"Inferred default_feature_dim={cfg.default_feature_dim} "
                "from largest PSN in dataset."
            )
        else:
            log(f"Using user-provided default_feature_dim={cfg.default_feature_dim}.")

        sample_names = [s["sample_name"] for s in samples]
        labels = np.array([int(s["label"]) for s in samples], dtype=np.int64)
        num_classes = len(class_mapping)

        if cfg.feature_mode == "dgdvms":
            input_dim = int(samples[0]["features"].shape[-1])
        else:
            input_dim = cfg.default_feature_dim

        log(f"Input feature dimension: {input_dim}")

        if cfg.partitions_mode == "user":
            outer_folds = load_user_outer_folds(cfg, sample_names)
        else:
            outer_folds = make_auto_outer_folds(labels, cfg.outer_folds, cfg.seed)
            save_generated_partitions(cfg, sample_names, outer_folds)

        t0 = time.perf_counter()

        all_fold_metrics = []
        all_raw_rows = []
        chosen_hparams_per_fold = []

        for fold_id, outer in enumerate(outer_folds, start=1):
            step(f"OUTER FOLD {fold_id}/{len(outer_folds)}")
            log(f"Outer fold train size: {len(outer['train'])}")
            log(f"Outer fold test size: {len(outer['test'])}")

            tune_info = tune_hyperparameters(
                cfg=cfg,
                samples=samples,
                labels=labels,
                outer_train_idx=outer["train"],
                input_dim=input_dim,
                num_classes=num_classes,
            )
            best_hp = tune_info["best"]
            log(f"Selected hyperparameters for fold {fold_id}: {best_hp}")
            chosen_hparams_per_fold.append({"fold": fold_id, **best_hp})

            inner_for_final = make_auto_inner_folds(
                train_indices=outer["train"],
                labels=labels,
                n_splits=cfg.inner_folds,
                seed=cfg.seed,
            )[0]

            train_loader = make_dataloader(samples, inner_for_final["train"], cfg, shuffle=True)
            val_loader = make_dataloader(samples, inner_for_final["val"], cfg, shuffle=False)
            test_loader = make_dataloader(samples, outer["test"], cfg, shuffle=False)

            log("Building final model for this outer fold.")
            model = build_model(cfg, input_dim=input_dim, num_classes=num_classes, dropout=best_hp["dropout"])

            log("Training final model on outer-train split.")
            model, hist = train_with_early_stopping(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
                lr=best_hp["lr"],
                weight_decay=best_hp["weight_decay"],
            )

            log("Evaluating on outer-test split.")
            test_loss, test_misclass, raw_rows = run_one_epoch(
                model=model,
                dataloader=test_loader,
                cfg=cfg,
                optimizer=None,
            )

            log(f"Fold {fold_id} test loss: {test_loss:.6f}")
            log(f"Fold {fold_id} test misclassification: {test_misclass:.6f}")

            for row in raw_rows:
                row["fold"] = fold_id
                all_raw_rows.append(row)

            all_fold_metrics.append({
                "fold": fold_id,
                "test_loss": test_loss,
                "test_misclassification": test_misclass,
                "test_accuracy": 1.0 - test_misclass,
                "best_val_misclassification": hist["best_val_misclass"],
                "best_epoch": hist["best_epoch"],
                "lr": best_hp["lr"],
                "dropout": best_hp["dropout"],
                "weight_decay": best_hp["weight_decay"],
            })

            if cfg.save_models:
                model_fp = out_dir / f"model_fold_{fold_id}.pt"
                torch.save(model.state_dict(), model_fp)
                log(f"Saved model for fold {fold_id} to {model_fp}")

        runtime_seconds = time.perf_counter() - t0
        step("WRITING OUTPUT FILES")

        fold_mis = [m["test_misclassification"] for m in all_fold_metrics]
        fold_acc = [m["test_accuracy"] for m in all_fold_metrics]

        average_misclassification = float(np.mean(fold_mis))
        average_accuracy = float(np.mean(fold_acc))

        total_correct = sum(r["correct"] for r in all_raw_rows)
        total_count = len(all_raw_rows)
        aggregate_accuracy = total_correct / max(total_count, 1)
        aggregate_misclassification = 1.0 - aggregate_accuracy

        hp_counts: Dict[Tuple[float, float, float], List[float]] = {}
        for item in chosen_hparams_per_fold:
            key = (item["lr"], item["dropout"], item["weight_decay"])
            hp_counts.setdefault(key, []).append(item["mean_inner_val_misclass"])

        hp_ranked = sorted(hp_counts.items(), key=lambda kv: (-len(kv[1]), np.mean(kv[1])))
        best_key, best_scores = hp_ranked[0]
        overall_best_hp = {
            "lr": best_key[0],
            "dropout": best_key[1],
            "weight_decay": best_key[2],
            "selection_count": len(best_scores),
            "mean_inner_val_misclassification": float(np.mean(best_scores)),
        }

        summary = {
            "dataset_name": cfg.dataset_name,
            "model_type": cfg.model_type,
            "feature_mode": cfg.feature_mode,
            "partitions_mode": cfg.partitions_mode,
            "num_samples_loaded": len(samples),
            "num_samples_skipped": len(SKIPPED_SAMPLES),
            "num_classes": num_classes,
            "class_mapping": class_mapping,
            "aggregate_accuracy": aggregate_accuracy,
            "aggregate_misclassification": aggregate_misclassification,
            "average_accuracy": average_accuracy,
            "average_misclassification": average_misclassification,
            "classification_runtime_seconds": runtime_seconds,
        }

        with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log("Wrote summary_metrics.json")

        with open(out_dir / "optimal_hyperparameters.json", "w", encoding="utf-8") as f:
            json.dump({"per_fold": chosen_hparams_per_fold, "overall_best": overall_best_hp}, f, indent=2)
        log("Wrote optimal_hyperparameters.json")

        with open(out_dir / "runtime_seconds.txt", "w", encoding="utf-8") as f:
            f.write(f"{runtime_seconds:.6f}\n")
        log("Wrote runtime_seconds.txt")

        with open(out_dir / "fold_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fold",
                    "test_loss",
                    "test_misclassification",
                    "test_accuracy",
                    "best_val_misclassification",
                    "best_epoch",
                    "lr",
                    "dropout",
                    "weight_decay",
                ],
            )
            writer.writeheader()
            writer.writerows(all_fold_metrics)
        log("Wrote fold_metrics.csv")

        with open(out_dir / "raw_classifications.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "fold",
                "sample_name",
                "true_label",
                "pred_label",
                "correct",
                "probabilities_json",
            ])
            for row in all_raw_rows:
                writer.writerow([
                    row["fold"],
                    row["sample_name"],
                    row["true_label"],
                    row["pred_label"],
                    row["correct"],
                    json.dumps(row["probabilities"]),
                ])
        log("Wrote raw_classifications.csv")

        step("DONE")
        log(f"Aggregate accuracy: {aggregate_accuracy:.6f}")
        log(f"Aggregate misclassification: {aggregate_misclassification:.6f}")
        log(f"Average accuracy: {average_accuracy:.6f}")
        log(f"Average misclassification: {average_misclassification:.6f}")
        log(f"Runtime (seconds): {runtime_seconds:.6f}")

    except Exception as e:
        if len(SKIPPED_SAMPLES) > 0:
            write_skipped_samples_file(cfg)
        warn(f"Program terminated: {e}")
        raise

    finally:
        close_logger()

def infer_largest_psn_size(samples: List[Dict[str, Any]]) -> int:
    if len(samples) == 0:
        raise RuntimeError("Cannot infer largest PSN size because no samples were loaded.")
    return max(int(s["adjs"].shape[1]) for s in samples)

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run nested 5-fold CV for SGCN or DGCN on dataset folder structure."
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=None,
        help="Maximum number of nodes used for padding. If omitted, inferred from the largest PSN in the dataset.",
    )

    parser.add_argument(
        "--default_feature_dim",
        type=int,
        default=None,
        help="Default node-feature dimension when --feature_mode default is used. If omitted, inferred from the largest PSN in the dataset.",
    )
    parser.add_argument("--model_type", type=str, choices=["sgcn", "dgcn"], required=True)
    parser.add_argument("--feature_mode", type=str, choices=["default", "dgdvms"], required=True)
    parser.add_argument("--partitions_mode", type=str, choices=["user", "auto"], required=True)
    parser.add_argument("--max_nodes", type=int, default=1072)
    parser.add_argument("--default_feature_dim", type=int, default=1072)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr_grid", type=str, default="1e-3")
    parser.add_argument("--dropout_grid", type=str, default="0.3")
    parser.add_argument("--weight_decay_grid", type=str, default="0.0")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--quiet_epochs", action="store_true")

    args = parser.parse_args()

    return Config(
        root_dir=args.root_dir,
        dataset_name=args.dataset_name,
        model_type=args.model_type.lower(),
        feature_mode=args.feature_mode.lower(),
        partitions_mode=args.partitions_mode.lower(),
        max_nodes=args.max_nodes,
        default_feature_dim=args.default_feature_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        lr_grid=tuple(float(x) for x in args.lr_grid.split(",")),
        dropout_grid=tuple(float(x) for x in args.dropout_grid.split(",")),
        weight_decay_grid=tuple(float(x) for x in args.weight_decay_grid.split(",")),
        save_models=args.save_models,
        verbose_epochs=not args.quiet_epochs,
    )

if __name__ == "__main__":
    cfg = parse_args()
    run_experiment(cfg)