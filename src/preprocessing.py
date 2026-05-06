"""
src/preprocessing.py

Preprocessing pipeline for WM-811K wafer maps.

Steps:
  1. Resize each wafer map to TARGET_SIZE using nearest-neighbor interpolation.
     (Nearest-neighbor is required because values are categorical: 0/1/2.)
  2. Normalize to [0, 1] by dividing by 2.0.
  3. Encode string labels to integer indices.
  4. Stratified train / validation / test split (70 / 15 / 15).
  5. Save arrays + label map to data/processed/.

See wiki/preprocessing-decisions.md for the rationale behind each decision.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import LABEL_MAP, LABEL_MAP_INV, DEFECT_CLASSES

TARGET_SIZE: tuple[int, int] = (64, 64)  # (width, height) for cv2.resize
PROCESSED_DIR = Path("data/processed")


# ── Resize ────────────────────────────────────────────────────────────────────

def resize_wafer_map(wmap: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize a 2D wafer map to `size` using nearest-neighbor interpolation."""
    return cv2.resize(wmap.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)


def normalize_wafer_map(wmap: np.ndarray) -> np.ndarray:
    """Map {0, 1, 2} -> {0.0, 0.5, 1.0} as float32."""
    return (wmap / 2.0).astype(np.float32)


# ── Build feature matrix ──────────────────────────────────────────────────────

def build_arrays(
    df: pd.DataFrame,
    size: tuple[int, int] = TARGET_SIZE,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a labeled DataFrame into (X, y) arrays.

    X shape: (N, 1, H, W) — PyTorch channel-first, normalized float32
    y shape: (N,)          — integer class indices

    Parameters
    ----------
    df      : labeled DataFrame with 'waferMap' and 'label' columns
    size    : resize target (W, H)
    verbose : print progress

    Returns
    -------
    X : np.ndarray, float32, shape (N, 1, H, W)
    y : np.ndarray, int64,   shape (N,)
    """
    n = len(df)
    h, w = size[1], size[0]
    X = np.empty((n, 1, h, w), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)

    for i, (_, row) in enumerate(df.iterrows()):
        wmap = np.array(row["waferMap"])
        resized = resize_wafer_map(wmap, size)
        X[i, 0] = normalize_wafer_map(resized)
        y[i] = LABEL_MAP[row["label"]]

        if verbose and (i + 1) % 10_000 == 0:
            print(f"  Processed {i+1:,} / {n:,}")

    if verbose:
        print(f"  Done. X shape: {X.shape}  y shape: {y.shape}")
    return X, y


# ── Split ─────────────────────────────────────────────────────────────────────

def split_arrays(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, ...]:
    """
    Stratified train / val / test split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    # Split val from remaining
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_fraction,
        stratify=y_train_val,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_processed(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    out_dir: str | Path = PROCESSED_DIR,
) -> None:
    """Save split arrays and label map to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy",   X_val)
    np.save(out_dir / "X_test.npy",  X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy",   y_val)
    np.save(out_dir / "y_test.npy",  y_test)

    label_map_path = out_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({"label_to_idx": LABEL_MAP, "idx_to_label": LABEL_MAP_INV}, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print(f"  label_map.json: {LABEL_MAP}")


def load_processed(
    out_dir: str | Path = PROCESSED_DIR,
) -> tuple[np.ndarray, ...]:
    """Load saved split arrays from `out_dir`."""
    out_dir = Path(out_dir)
    X_train = np.load(out_dir / "X_train.npy")
    X_val   = np.load(out_dir / "X_val.npy")
    X_test  = np.load(out_dir / "X_test.npy")
    y_train = np.load(out_dir / "y_train.npy")
    y_val   = np.load(out_dir / "y_val.npy")
    y_test  = np.load(out_dir / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Compute per-class weights for CrossEntropyLoss.

    Formula: weight_c = total_samples / (num_classes * count_c)
    This up-weights rare classes so the loss penalises their errors more.

    Returns
    -------
    np.ndarray of shape (num_classes,), float32
    """
    num_classes = len(DEFECT_CLASSES)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = len(y_train) / (num_classes * counts)
    return weights.astype(np.float32)
