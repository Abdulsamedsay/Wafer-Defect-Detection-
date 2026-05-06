"""
src/data_loader.py

Loads the WM-811K dataset from LSWMD.pkl.

The file was pickled with an old version of pandas (Python 2 era).
Modern pandas cannot load it directly. We patch sys.modules to remap
the removed 'pandas.indexes' namespace and load with encoding='latin1'.
See wiki/errors-and-fixes.md for the full explanation.
"""

import sys
import types
import importlib
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


_PANDAS_REMAP = {
    "pandas.indexes":            "pandas.core.indexes",
    "pandas.indexes.base":       "pandas.core.indexes.base",
    "pandas.indexes.numeric":    "pandas.core.indexes.numeric",
    "pandas.indexes.range":      "pandas.core.indexes.range",
    "pandas.indexes.frozen":     "pandas.core.indexes.frozen",
    "pandas.indexes.api":        "pandas.core.indexes.api",
    "pandas.indexes.category":   "pandas.core.indexes.category",
    "pandas.indexes.datetimes":  "pandas.core.indexes.datetimes",
    "pandas.indexes.multi":      "pandas.core.indexes.multi",
    "pandas.indexes.period":     "pandas.core.indexes.period",
    "pandas.indexes.timedeltas": "pandas.core.indexes.timedeltas",
}


def _patch_pandas_modules() -> None:
    for old, new in _PANDAS_REMAP.items():
        if old not in sys.modules:
            try:
                sys.modules[old] = importlib.import_module(new)
            except ImportError:
                sys.modules[old] = types.ModuleType(old)


def _extract_label(failure_type) -> str:
    try:
        value = failure_type[0][0]
        return "none" if (value == "" or value is None) else value
    except (IndexError, TypeError):
        return "unlabeled"


def load_raw(pkl_path: str | Path) -> pd.DataFrame:
    """
    Load LSWMD.pkl and return the raw DataFrame.

    Parameters
    ----------
    pkl_path : path to LSWMD.pkl

    Returns
    -------
    pd.DataFrame with original columns plus an extracted 'label' column.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {pkl_path}.\n"
            "Download LSWMD.pkl from Kaggle and place it at data/raw/LSWMD.pkl"
        )

    _patch_pandas_modules()

    print(f"Loading {pkl_path.name} ({pkl_path.stat().st_size / 1e9:.1f} GB)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(pkl_path, "rb") as f:
            df = pickle.load(f, encoding="latin1")

    df["label"] = df["failureType"].apply(_extract_label)
    print(f"Loaded: {len(df):,} total rows, columns: {df.columns.tolist()}")
    return df


def load_labeled(pkl_path: str | Path) -> pd.DataFrame:
    """
    Load LSWMD.pkl and return only labeled samples.

    Returns
    -------
    pd.DataFrame with 'label' column, unlabeled rows removed.
    """
    df = load_raw(pkl_path)
    labeled = df[df["label"] != "unlabeled"].copy().reset_index(drop=True)
    print(f"Labeled samples: {len(labeled):,} ({len(labeled)/len(df)*100:.1f}% of total)")
    return labeled


DEFECT_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none",
]

LABEL_MAP: dict[str, int] = {cls: i for i, cls in enumerate(DEFECT_CLASSES)}
LABEL_MAP_INV: dict[int, str] = {i: cls for cls, i in LABEL_MAP.items()}
