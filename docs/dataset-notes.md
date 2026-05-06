# Dataset Notes

## Dataset

**Name:** WM-811K Wafer Map Dataset  
**Source:** Published by MHang Wu (National Cheng Kung University), widely distributed via Kaggle  
**Format:** Pickle file (`.pkl`) containing a pandas DataFrame — pickled with Python 2 / old pandas  
**File:** `LSWMD.pkl` (approx. 2 GB)

---

## Structure

The dataset is a single `.pkl` file. When loaded, it is a pandas DataFrame with 811,457 rows and 6 columns.

| Column | Type | Description |
|---|---|---|
| `waferMap` | 2D NumPy array | The wafer die map (variable size) |
| `dieSize` | float | Size of each die |
| `lotName` | object | Lot identifier |
| `waferIndex` | int | Wafer position within the lot |
| `trianTestLabel` | object | Train/test split label *(typo in original dataset — "trian" not "train")* |
| `failureType` | nested list | Defect class label — e.g. `[['Edge-Ring']]` or `[['']]` for unlabeled |

---

## Label Structure

The `failureType` column stores labels as nested lists. Extraction:

```python
def extract_label(ft):
    try:
        v = ft[0][0]
        return "none" if (v == "" or v is None) else v
    except Exception:
        return "unlabeled"
```

After extraction, the 9 defect classes are:

| Class | Count | Notes |
|---|---|---|
| none | 147,431 | Normal wafer — no defect pattern |
| Edge-Ring | 9,680 | Ring defect along wafer edge |
| Edge-Loc | 5,189 | Localized edge defect |
| Center | 4,294 | Defect pattern in center region |
| Loc | 3,593 | Localized cluster (not center/edge) |
| Scratch | 1,193 | Linear scratch pattern |
| Random | 866 | Randomly distributed failures |
| Donut | 555 | Ring around the center |
| Near-full | 149 | Almost entire wafer affected |

---

## waferMap Encoding

Each cell in the wafer map array has one of three values:

| Value | Meaning |
|---|---|
| 0 | Background (outside the wafer boundary) |
| 1 | Passing die |
| 2 | Failing die |

---

## Known Challenges

- **Severe class imbalance:** `none` accounts for 85.2% of labeled samples. `Near-full` has only 149 examples. Imbalance ratio: **989x**.
- **Large proportion of unlabeled data:** 638,507 of 811,457 wafers (78.7%) have no defect label. Only labeled samples are used for supervised training.
- **Variable wafer map sizes:** Maps range from 6×3 to 300×205 pixels with 129 distinct row sizes and 120 distinct column sizes. All maps must be resized before model input.
- **Legacy pickle format:** The file was pickled with an old version of pandas (pre-1.0, Python 2). Modern pandas (2.x+) cannot load it directly. A compatibility layer is required (see `docs/errors-and-fixes.md`).
- **Column name typo:** The `trainTestLabel` column is actually named `trianTestLabel` in the raw data.

---

## How to Load

```python
import pickle
import pandas as pd

# Direct load with latin1 encoding (bypasses pandas compat layer)
with open("data/raw/LSWMD.pkl", "rb") as f:
    df = pickle.load(f, encoding="latin1")
```

See `src/data_loader.py` (to be created in Phase 2) for the full loading pipeline.

---

## Download

Search for **"WM-811K"** or **"LSWMD.pkl"** on Kaggle. After downloading, place the file at:

```
data/raw/LSWMD.pkl
```
