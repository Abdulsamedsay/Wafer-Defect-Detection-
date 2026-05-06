"""
src/dataset.py

PyTorch Dataset for the preprocessed WM-811K wafer maps.
Loads from the .npy files saved by src/preprocessing.py.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_loader import DEFECT_CLASSES, LABEL_MAP, LABEL_MAP_INV
from src.preprocessing import load_processed, compute_class_weights, PROCESSED_DIR


class WaferDataset(Dataset):
    """
    PyTorch Dataset wrapping the preprocessed wafer map arrays.

    Each item is (tensor, label) where:
      tensor : float32, shape (1, 64, 64)  — already normalized to [0, 1]
      label  : int64 scalar
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        """
        Parameters
        ----------
        X       : float32 array, shape (N, 1, H, W)
        y       : int64 array, shape (N,)
        augment : if True, apply random horizontal/vertical flip
        """
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])  # horizontal flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[1])  # vertical flip
        return x, self.y[idx]


def make_dataloaders(
    processed_dir: str | Path = PROCESSED_DIR,
    batch_size: int = 64,
    num_workers: int = 0,
    augment_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders from saved .npy files.

    Parameters
    ----------
    batch_size    : samples per batch
    num_workers   : parallel data loading workers (0 = main process only)
    augment_train : apply random flips to training data

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed(processed_dir)

    train_ds = WaferDataset(X_train, y_train, augment=augment_train)
    val_ds   = WaferDataset(X_val,   y_val,   augment=False)
    test_ds  = WaferDataset(X_test,  y_test,  augment=False)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def get_class_weights_tensor(
    processed_dir: str | Path = PROCESSED_DIR,
    device: str = "cpu",
) -> torch.Tensor:
    """Return class weights as a torch.Tensor on `device`."""
    _, _, _, y_train, _, _ = load_processed(processed_dir)
    weights = compute_class_weights(y_train)
    return torch.tensor(weights, dtype=torch.float32, device=device)
