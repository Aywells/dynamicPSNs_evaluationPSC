from __future__ import annotations
# -*- coding: utf-8 -*-

"""
CNN + BiLSTM model for protein structure classification.

- Loads matrices from final_data (one file per protein)
        #  make training dataset infinite
        train_ds = train_ds.repeat()

        # how many batches per epoch
        steps_per_epoch = math.ceil(len(X_tr_fold) / BATCH_SIZE)

        model = build_cnn_bilstm_model(INPUT_DIM, n_classes)

        # fresh callbacks for each fold
        callbacks = get_callbacks()

        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=2,
        )

        # Save plots for this fold
        plot_history(
            history,
            title=f"Fold {fold}",
            out_path=str(DATA_DIR / f"cv_fold_{fold}")
        )

        # Evaluate on validation set at the end of training
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        fold_accuracies.append(val_acc)
        print(f"Fold {fold} validation accuracy: {val_acc:.4f}")

        # Detailed metrics on validation fold
        metrics = evaluate_metrics(model, val_ds, n_classes, average="macro", labels=np.arange(n_classes))
        C_agg += metrics["cm"]

    # Aggregate misclassification
    agg_total = C_agg.sum()
    agg_correct = np.trace(C_agg)
    agg_miscls = (agg_total - agg_correct) / agg_total

    # ===== Cross-validation summary =====
    print("\n===== Cross-validation results =====")
    print("Fold accuracies:", [float(a) for a in fold_accuracies])
    print(f"Mean val acc: {np.mean(fold_accuracies):.4f}  "
          f"+/- {np.std(fold_accuracies):.4f}")

    # ---- Final model on all train+val, evaluate on test ----
    train_ds_all = make_dataset(X_trainval, y_trainval,
                                batch_size=BATCH_SIZE, shuffle=True)
    test_ds	 = make_dataset(X_test,     y_test,
                                batch_size=BATCH_SIZE, shuffle=False)

    train_ds_all = train_ds_all.repeat()
    final_steps = math.ceil(len(X_trainval) / BATCH_SIZE)

    final_model = build_cnn_bilstm_model(INPUT_DIM, n_classes)
    final_callbacks = get_callbacks()

    print("\n===== Training final model on all train+val data =====")
    final_history = final_model.fit(
        train_ds_all,
        epochs=EPOCHS,
        steps_per_epoch=final_steps,
        callbacks=final_callbacks,
        verbose=2,
    )
- Uses final_files.txt to define the sample order
- Uses labels_final.npy as integer class labels
- Pads variable-length sequences (variable M, fixed C)
- Splits into (train+val) + test
- Runs 5-fold Stratified CV on the train+val part
- Trains a final model on all train+val and evaluates on test
"""

# =========================
# MODEL CONFIG PARAMETERS
# =========================
import time
start = time.perf_counter()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

import warnings
warnings.filterwarnings(
    "ignore",
    message="Your input ran out of data; interrupting training.*"
)

# Root dataset directory
DATA_DIR = Path("/users/fgatsi/dataset/dsets")

# Use your class file and folder of matrices
dataset_folder =  "dataset-0" 
dataset_file =  "dataset-0.txt" 

MATRIX_DIR  = DATA_DIR /dataset_folder       # folder with per-protein matrices
CLASS_FILE  = DATA_DIR /dataset_file         # "filename  class_label" per line

# =========================
# Hyperparameters
# =========================
INPUT_DIM        = 211            # number of feature columns (fixed)
NUM_CLASSES      = None           # None => infer as max(label)+1
CNN_FILTERS1     = 56
CNN_FILTERS2     = 96
CNN_FILTERS3     = 128
KERNEL_SIZE      = 5
CNN_DROPOUT      = 0.3
LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS      = 3              # currently *not* used; we use 1 BiLSTM for now
LSTM_DROPOUT     = 0.3

EPOCHS     = 100
BATCH_SIZE = 32
#TEST_SIZE  = 0.2   # for splitting data outside this file
N_SPLITS   = 5     # for CV outside this file

# =========================
# DATA LOADING
# =========================

def load_X_y():
    """
    Load:
      - X_list: list of matrices, one per file, shape (M_i, INPUT_DIM)
      - y:      np.array of integer class IDs, shape (N,)

    Uses mapping file CLASS_FILE with lines:
        <filename> <class_label>

    Each matrix file in MATRIX_DIR / <filename> has 211 columns:
      - column 0: to be ignored
      - columns 1..211: 211 feature columns (INPUT_DIM)
    """

    file_names = []
    label_names = []

    # 1) Read mapping file: filename + class_label
    with CLASS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad line in {CLASS_FILE}: {line}")
            fname, label = parts
            file_names.append(fname)
            label_names.append(label)

    print(f"Loaded {len(file_names)} entries from {CLASS_FILE}")

    # 2) Map class_label strings to integer IDs 0..K-1
    classes = sorted(set(label_names))
    class_to_index = {c: i for i, c in enumerate(classes)}
    print("Classes:", classes)
    print("Class to index mapping:", class_to_index)

    y = np.array([class_to_index[c] for c in label_names], dtype=np.int64)

    # 3) Load each matrix, drop first column
    X_list = []
    for fname in file_names:
        path = MATRIX_DIR / fname
        A = np.loadtxt(path)  # whitespace-separated matrix

        # Handle 1D edge case
        if A.ndim == 1:
            A = A.reshape(1, -1)

        # Drop the first column (index 0)
        A = A[:, 1:]

        if A.shape[1] != INPUT_DIM:
           raise ValueError(
                f"After dropping first column, matrix {path} has "
                f"{A.shape[1]} columns, expected {INPUT_DIM}"
            )

        X_list.append(A.astype("float32"))

    print(f"Loaded {len(X_list)} matrices from {MATRIX_DIR}")
    print("Example matrix shape (after dropping col 0):", X_list[0].shape)

    return X_list, y


def make_dataset(X_list, y, batch_size=32, shuffle=True):
    """
    Build a tf.data.Dataset from a list of (M_i, C) arrays and labels.
    Uses padding within each batch.
    """
    n_features = X_list[0].shape[1]

    def gen():
        for A, label in zip(X_list, y):
            # A: (M_i, C)
            yield A.astype("float32"), np.int64(label)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(X_list))

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=([None, n_features], []),
        padding_values=(
            tf.constant(0.0, dtype=tf.float32),
           tf.constant(0,   dtype=tf.int64),   # <-- match int64
        ),
    )
    return ds


# =========================
#  Model definition
# =========================
def build_cnn_bilstm_model(
    n_features=INPUT_DIM,
    n_classes=NUM_CLASSES,
    l2_reg=1e-4,
    lr=1e-3,
    ):
    """
    n_features: number of columns (e.g., 211 or 212)
    n_classes:  number of structure classes (if None, infer later from labels)
    """

    inputs = tf.keras.Input(shape=(None, n_features))   # (batch, T, C)

    # If you pad with all-zeros, you could do:
    # x = layers.Masking(mask_value=0.0)(inputs)
    x = inputs

    # ----- 3× CNN block: 56 -> 96 -> 128 filters -----
    # Conv 1: CNN_FILTERS1
    x = layers.Conv1D(
        filters=CNN_FILTERS1,
        kernel_size=KERNEL_SIZE,
        strides=1,
        padding="same",
        activation=None,
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dropout(CNN_DROPOUT)(x)

    # Conv 2: CNN_FILTERS2
    x = layers.Conv1D(
        filters=CNN_FILTERS2,
        kernel_size=KERNEL_SIZE,
        strides=1,
        padding="same",
        activation=None,
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dropout(CNN_DROPOUT)(x)

    # Conv 3: CNN_FILTERS3
    x = layers.Conv1D(
        filters=CNN_FILTERS3,
        kernel_size=KERNEL_SIZE,
        strides=1,
        padding="same",
        activation=None,
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dropout(CNN_DROPOUT)(x)

    # ----- BiLSTM block: LSTM_HIDDEN_SIZE units per direction -----
    # (We use 1 BiLSTM layer even though LSTM_LAYERS=3 is defined;
    #  stacking 3 was likely too heavy / hard to optimize.)
    x = layers.Bidirectional(
        layers.LSTM(
            LSTM_HIDDEN_SIZE,
            return_sequences=False,	  # final vector per protein
            dropout=0.0,         # input dropout inside LSTM
            unit_forget_bias=True,
        )
    )(x)  # shape: (batch, 2 * LSTM_HIDDEN_SIZE) = (batch, 256)

    # Dropout after sequence aggregation
    x = layers.Dropout(LSTM_DROPOUT)(x)

    # ----- Output layer -----
    outputs = layers.Dense(
        n_classes if n_classes is not None else 1,  # will fix if None in training helper
        activation="softmax",
        #kernel_regularizer=regularizers.l2(l2_reg),
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="cnn_bilstm_graphlet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
#  Callbacks for early stopping and learning rate optimization
# ============================================================
def get_callbacks():
    # More patient early stopping to avoid cutting training too early
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",     # you can switch to "val_accuracy" if smoother
        patience=20,            # allow many "bad" epochs
        min_delta=1e-4,         # minimal improvement
        restore_best_weights=True,
        verbose=1,
    )

    # Reduce learning rate (LR) when val_loss plateaus
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,             # halve learning rate
        patience=10,            # after 5 stagnant epochs
        min_lr=1e-6,
        verbose=1,
    )

    return [early_stop, lr_schedule]


# =========================
#  Training helper
# =========================

def plot_history(history, title="", out_path=None):
    """Plot training & validation loss/accuracy from a Keras History."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    # ---- Loss ----
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, hist["loss"], label="train loss")
    if "val_loss" in hist:
        plt.plot(epochs, hist["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss" if title else "Loss")
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(f"{out_path}_loss.png", dpi=200)
        plt.close()
    else:
        plt.show()

    # ---- Accuracy ----
    if "accuracy" in hist:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, hist["accuracy"], label="train acc")
        if "val_accuracy" in hist:
            plt.plot(epochs, hist["val_accuracy"], label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{title} - Accuracy" if title else "Accuracy")
        plt.legend()
        plt.tight_layout()
        if out_path:
            plt.savefig(f"{out_path}_acc.png", dpi=200)
            plt.close()
        else:
            plt.show()

def evaluate_metrics(model, dataset, n_classes, average="macro", labels=None, print_cm=True):
    """
    Compute Precision, Recall, F1-score, AUROC, Confusion Matrix,
    and Misclassification rate for a trained model on a tf.data.Dataset.

    Assumes:
      - labels are integer class IDs
      - dataset yields (X_batch, y_batch)
    """
    y_true = []
    y_pred = []
    y_prob = []

    # 1) Collect predictions
    for X_batch, y_batch in dataset:
        probs = model.predict(X_batch, verbose=0)      # (batch, n_classes)
        preds = np.argmax(probs, axis=1)

        y_true.append(y_batch.numpy())
        y_pred.append(preds)
        y_prob.append(probs)

    y_true = np.concatenate(y_true).astype(int)
    y_pred = np.concatenate(y_pred).astype(int)
    y_prob = np.concatenate(y_prob)

    # Labels for confusion_matrix (keeps consistent ordering across folds)
    if labels is None:
        labels = np.arange(n_classes)

    # 2) Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    print(f"\n{average.capitalize()}-averaged Precision: {precision:.4f}")
    print(f"{average.capitalize()}-averaged Recall:    {recall:.4f}")
    print(f"{average.capitalize()}-averaged F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # 3) Confusion matrix + misclassification
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    total = cm.sum()
    correct = np.trace(cm)
    miscls = (total - correct) / total if total > 0 else np.nan

    if print_cm:
        print("\nConfusion matrix:")
        print(cm)
        print(f"Misclassification rate: {miscls:.4f}")

    # 4) AUROC (multi-class, one-vs-rest)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(
                y_true, y_prob,
                multi_class="ovr",
                average=average
            )
        print(f"{average.capitalize()}-averaged AUROC:   {auc:.4f}")
    except ValueError as e:
        auc = None
        print("AUROC could not be computed:", e)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "miscls": miscls,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ==================================
# PRINTING THE CV FOLDS (PARTITIONS)
# ==================================
def print_stratified_folds_full_data(
    y: Union[np.ndarray, Sequence[int]],
    class_file: Union[str, Path],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    print_filenames: bool = True,
    return_partitions: bool = True,
) -> Optional[List[np.ndarray]]:
    """
    Create StratifiedKFold partitions over the ENTIRE dataset (no test split),
    print each fold's "val partition" indices (and optional filenames),
    and optionally return the list of partitions (each is a numpy array of indices).

    Assumes `class_file` has whitespace-separated columns:
        <filename> <class>
    and that its row order matches `y` order.

    Parameters
    ----------
    y : array-like
        Labels for all samples (length n_samples).
    class_file : str | Path
        Path to CLASS_FILE (filename + class per line).
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle before splitting.
    random_state : int
        Random seed (used when shuffle=True).
    print_filenames : bool
        If True, print filenames for each fold partition.
    return_partitions : bool
        If True, return list of fold partitions (val indices). Otherwise return None.

    Returns
    -------
    partitions : list[np.ndarray] | None
        List of length n_splits containing the val-partition indices for each fold.
        Across folds, these partitions cover the whole dataset exactly once.
    """
    y_arr = np.asarray(y)
    n_samples = len(y_arr)
    all_indices = np.arange(n_samples)

    df = pd.read_csv(class_file, sep=r"\s+", header=None, names=["filename", "cls"])
    df = df.iloc[:n_samples].reset_index(drop=True)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )

    partitions: List[np.ndarray] = []

    for fold, (_, val_idx) in enumerate(skf.split(all_indices, y_arr), 1):
        partitions.append(val_idx)

        print()
        print(f"Fold-{fold} (val partition indices): {val_idx.tolist()}")

        if print_filenames:
            print(f"Fold-{fold} (val partition filenames):")
            for fname in df.iloc[val_idx]["filename"].tolist():
                print(fname)

    # Sanity checks: cover all samples exactly once across the printed partitions
    union = np.unique(np.concatenate(partitions)) if partitions else np.array([], dtype=int)
    total = sum(len(p) for p in partitions)

    print()
    print(f"Coverage check: {len(union)}/{n_samples} unique indices covered")
    print("No-overlap check:", "OK" if total == n_samples else f"Not OK (sum sizes={total}, n={n_samples})")

    return partitions if return_partitions else None


# ================================
# TRAINING & CROSS-VALIDATION (CV)
# ================================
def run_stratified_kfold_cv(
    X_list,
    y,
    n_classes,
    input_dim,
    n_splits,
    batch_size,
    epochs,
    build_model_fn,     # e.g., build_cnn_bilstm_model
    make_dataset_fn,    # e.g., make_dataset
    get_callbacks_fn,   # e.g., get_callbacks
    evaluate_metrics_fn,# e.g., evaluate_metrics
    random_state=42,
    ):
    """
    Runs Stratified K-Fold CV on ALL data (no test set), scaling per fold using TRAIN ONLY.
    Prints fold results each cycle and computes aggregate confusion matrix + misclassification.
    Returns a dict with fold accuracies, aggregate CM, aggregate MR, and summary stats.
    """
    y = np.asarray(y, dtype=np.int64)
    indices = np.arange(len(X_list))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_accuracies = []
    C_agg = np.zeros((n_classes, n_classes), dtype=int)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(indices, y), start=1):

        # ---- Split raw fold data ----
        X_tr_raw  = [X_list[i] for i in tr_idx]
        X_val_raw = [X_list[i] for i in val_idx]
        y_tr      = y[tr_idx]
        y_val     = y[val_idx]

        # ---- Fit scaler on TRAIN fold only (avoid leakage) ----
        train_rows = np.concatenate(X_tr_raw, axis=0)
        scaler = StandardScaler().fit(train_rows)

        # ---- Transform train + val fold ----
        X_tr  = [scaler.transform(A) for A in X_tr_raw]
        X_val = [scaler.transform(A) for A in X_val_raw]

        # ---- Datasets ----
        train_ds = make_dataset_fn(X_tr, y_tr, batch_size=batch_size, shuffle=True).repeat()
        val_ds   = make_dataset_fn(X_val, y_val, batch_size=batch_size, shuffle=False)

        steps_per_epoch = math.ceil(len(X_tr) / batch_size)

        # ---- Build and train model ----
        model = build_model_fn(input_dim, n_classes)
        callbacks = get_callbacks_fn()

        print(f"\n===== Fold {fold}/{n_splits} =====")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=2,
        )

        # ---- Fold evaluation ----
        _, val_acc = model.evaluate(val_ds, verbose=0)
        fold_accuracies.append(val_acc)
        print(f"\nFold {fold} val accuracy: {val_acc:.4f}\n")

        metrics = evaluate_metrics_fn(
            model,
            val_ds,
            n_classes,
            average="macro",
            labels=np.arange(n_classes),
            print_cm=True,
        )
        C_agg += metrics["cm"]

    # ---- Aggregate results ----
    agg_total = C_agg.sum()
    agg_correct = np.trace(C_agg)
    agg_miscls = (agg_total - agg_correct) / agg_total if agg_total > 0 else np.nan

    summary = {
        "fold_accuracies": fold_accuracies,
        "mean_val_acc": float(np.mean(fold_accuracies)) if fold_accuracies else np.nan,
        "std_val_acc": float(np.std(fold_accuracies)) if fold_accuracies else np.nan,
        "C_agg": C_agg,
        "agg_miscls": float(agg_miscls),
        "n_samples": int(len(y)),
        "n_classes": int(n_classes),
    }
    return summary


def main():

    # ---- Load all data ----
    X_list, y = load_X_y()
    y = np.asarray(y, dtype=np.int64)
    n_samples = len(X_list)

    # Infer number of classes if needed
    if NUM_CLASSES is None:
        n_classes = int(y.max()) + 1
    else:
        n_classes = NUM_CLASSES

    print(f"Number of samples (before filtering): {n_samples}")
    print(f"Number of classes (before filtering): {n_classes}")

    # ====== Drop classes with too few samples ======
    MIN_SAMPLES_PER_CLASS = N_SPLITS  # e.g., 5 for 5-fold CV

    counts = np.bincount(y, minlength=n_classes)
    valid_classes = np.where(counts >= MIN_SAMPLES_PER_CLASS)[0]

    print("\nClass counts (before filtering):")
    for cid, c in enumerate(counts):
        print(f"  class {cid:3d}: {c} samples")

    print(f"\nKeeping classes with >= {MIN_SAMPLES_PER_CLASS} samples.")
    print("Valid class IDs:", valid_classes.tolist())

    # Build mask for samples belonging to valid classes
    mask = np.isin(y, valid_classes)

    X_list = [x for x, keep in zip(X_list, mask) if keep]
    y = y[mask]

    n_samples = len(X_list)
    n_classes = len(valid_classes)

    print(f"\nAfter filtering:")
    print(f"  Number of samples: {n_samples}")
    print(f"  Number of classes: {n_classes}")

    # Remap class IDs to 0..n_classes-1 for compact softmax
    old_to_new = {old: new for new, old in enumerate(valid_classes)}
    y = np.array([old_to_new[int(lbl)] for lbl in y], dtype=np.int64)

    print("Remapped class IDs so they are 0..", n_classes - 1)

    # Infer number of classes again (after remap)
    if NUM_CLASSES is None:
        n_classes = int(y.max()) + 1
    else:
        n_classes = NUM_CLASSES

    print(f"\nFinal stats (after filtering & remap):")
    print(f"  Number of samples: {n_samples}")
    print(f"  Number of classes: {n_classes}")


    partitions = print_stratified_folds_full_data(
        y=y,
        class_file=CLASS_FILE,
        n_splits=5,
        shuffle=True,
        random_state=42,
        print_filenames=True,
        return_partitions=True,
    )


    cv = run_stratified_kfold_cv(
        X_list=X_list,
        y=y,
        n_classes=n_classes,
        input_dim=INPUT_DIM,
        n_splits=N_SPLITS,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        build_model_fn=build_cnn_bilstm_model,
        make_dataset_fn=make_dataset,
        get_callbacks_fn=get_callbacks,
        evaluate_metrics_fn=evaluate_metrics,
    )

    print("\n===== Cross-validation results =====")
    print("Fold accuracies:", [float(a) for a in cv["fold_accuracies"]])
    print(f"Mean val acc: {cv['mean_val_acc']:.4f} +/- {cv['std_val_acc']:.4f}")

    print("\n=== Aggregate confusion matrix (sum over folds) ===")
    print(cv["C_agg"])
    print(f"Aggregate misclassification rate (CV): {cv['agg_miscls']:.6f}")
    print(f"Number of samples: {cv['n_samples']}")
    print(f"Number of classes: {cv['n_classes']}")

    #loaded_model = tf.keras.models.load_model(DATA_DIR / "cnn_bilstm_graphlet_final.keras")

    # Computing elapsed time
    elapsed_time = time.perf_counter() - start
    h = int(elapsed_time // 3600)
    m = int((elapsed_time % 3600) // 60)
    s = elapsed_time % 60
    print(f"EXECUTION_TIME_SECONDS={elapsed_time:.6f}")
    print(f"EXECUTION_TIME_HHMMSS={h:02d}:{m:02d}:{s:06.3f}")


if __name__ == "__main__":
    main()
