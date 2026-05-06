# EDA Findings

> Based on: `notebooks/01_data_exploration.ipynb` / `eda_runner.py`  
> Dataset: `data/raw/LSWMD.pkl`  
> Date: 2026-05-03

---

## Dataset Overview

| Property | Value |
|---|---|
| Total wafer maps | 811,457 |
| Labeled samples | 172,950 (21.3%) |
| Unlabeled samples | 638,507 (78.7%) |
| Number of columns | 6 |
| Defect classes | 9 |

**Columns in the dataset:**
`waferMap`, `dieSize`, `lotName`, `waferIndex`, `trianTestLabel` *(note: typo in original dataset)*, `failureType`

---

## Class Distribution (Labeled Samples Only)

| Class | Count | % of Labeled |
|---|---|---|
| none | 147,431 | 85.2% |
| Edge-Ring | 9,680 | 5.6% |
| Edge-Loc | 5,189 | 3.0% |
| Center | 4,294 | 2.5% |
| Loc | 3,593 | 2.1% |
| Scratch | 1,193 | 0.7% |
| Random | 866 | 0.5% |
| Donut | 555 | 0.3% |
| Near-full | 149 | 0.1% |

**Imbalance ratio:** 989x (none vs Near-full)

This is severe class imbalance. A naive model that predicts `none` for every wafer would achieve ~85% accuracy on the labeled set — but would be completely useless for detecting actual defects.

---

## Wafer Map Properties

| Property | Value |
|---|---|
| Value encoding | 0 = background, 1 = pass die, 2 = fail die |
| Row range | 6 – 300 |
| Col range | 3 – 205 |
| Mean shape | ~45 × 43 |
| Unique row sizes | 129 |
| Unique col sizes | 120 |

Wafer maps are **variable-sized**. Every map must be resized to a fixed shape before being used as model input. The mean size (~45×43) suggests a target of **64×64** is large enough to preserve spatial structure without excessive padding.

---

## Key Observations

1. **85.2% of labeled data is `none` (normal).** This makes accuracy a misleading metric. Per-class F1 is required.
2. **Near-full has only 149 labeled examples.** This class will likely underperform in training unless augmented or weighted heavily.
3. **Donut (555) and Random (866) also have very few samples.** These are the next most at-risk classes.
4. **78.7% of the dataset is unlabeled.** These samples are not used for supervised training. Possible future use: self-supervised pretraining.
5. **Wafer maps have 129 distinct row sizes and 120 distinct col sizes.** No fixed shape can be assumed. Resizing is mandatory.
6. **The dataset column has a typo:** `trianTestLabel` (not `trainTestLabel`). Important to know when accessing this column.

---

## Implications for Preprocessing

- Filter out unlabeled rows before training
- Resize all wafer maps to 64×64 (preserves spatial structure, reasonable compute cost)
- Use class weights or weighted sampling to address the 989x imbalance
- Stratify train/val/test split by class to ensure all classes appear in each split

---

## Output Figures

- `outputs/figures/class_distribution.png` — bar chart of labeled class counts
- `outputs/figures/sample_wafer_maps.png` — 3×3 grid of one wafer map per class
