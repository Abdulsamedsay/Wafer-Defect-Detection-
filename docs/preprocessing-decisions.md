# Preprocessing Decisions

> Updated after EDA — 2026-05-03

---

## Label Cleaning

**Decision:** Extract the inner string from the nested list `failureType` column.  
**Rationale:** The raw column stores labels as `[['Edge-Ring']]` or `[['']]`. We extract `ft[0][0]` and treat empty strings as `'unlabeled'`.

```python
def extract_label(ft):
    try:
        v = ft[0][0]
        return "none" if (v == "" or v is None) else v
    except Exception:
        return "unlabeled"
```

**Decision:** Drop all unlabeled rows before training.  
**Rationale:** 638,507 samples (78.7%) have no label. Supervised training requires labels. Semi-supervised approaches using unlabeled data are listed in `docs/open-questions.md` as future work.

---

## Wafer Map Resizing

**Decision:** Resize all wafer maps to **64×64** pixels.  
**Rationale:** Wafer maps range from 6×3 to 300×205 with 129+ unique sizes — a fixed shape is mandatory for batched training. 64×64 is large enough to preserve spatial defect structure (Edge-Ring, Scratch, Center patterns) without excess compute. 32×32 loses too much spatial detail for subtle patterns like Donut.

**Method:** `cv2.resize(wmap, (64, 64), interpolation=cv2.INTER_NEAREST)`  
Nearest-neighbor interpolation is preferred over bilinear because wafer maps are categorical (0/1/2), not continuous images. Bilinear would introduce fractional values (e.g., 1.5) that have no physical meaning.

---

## Channel Handling

**Decision:** Treat the resized wafer map as a **single-channel (grayscale) image**.  
**Rationale:** There is no color information. Values are 0, 1, 2. This maps to shape `(1, 64, 64)` in PyTorch's `[C, H, W]` convention.

---

## Normalization

**Decision:** Normalize pixel values to **[0, 1]** by dividing by 2.0.  
**Rationale:** Values are always in {0, 1, 2}. Dividing by 2 maps them to {0.0, 0.5, 1.0}, preserving the categorical spacing. Standard mean/std normalization is not appropriate for categorical maps.

---

## Train / Validation / Test Split

**Decision:** 70% train / 15% validation / 15% test, **stratified by class**.  
**Rationale:** Stratification is critical with 989x class imbalance — without it, the rare Near-full class (149 samples) could end up entirely in one split.

With 172,950 labeled samples:
- Train: ~121,065
- Validation: ~25,943
- Test: ~25,942

---

## Class Imbalance Strategy

**Decision:** Use **class weights in the CrossEntropyLoss** as the primary strategy.  
**Rationale:** With 989x imbalance, the loss function must be aware of class frequency. Class weights proportional to inverse frequency penalize errors on rare classes more strongly.

```python
# Weight = total_samples / (num_classes * class_count)
weights = total / (num_classes * class_counts)
criterion = nn.CrossEntropyLoss(weight=weights)
```

**Secondary strategy:** Investigate **weighted random sampling** in DataLoader if loss weighting alone is insufficient.

**Rejected:** SMOTE for image data is non-trivial and may generate unrealistic wafer maps. Saved as future work in `docs/open-questions.md`.

---

## Label Encoding

**Decision:** Map defect class strings to integer indices (0–8).  

| Index | Class | Class Weight |
|---|---|---|
| 0 | Center | 4.4749 |
| 1 | Donut | 34.5798 |
| 2 | Edge-Loc | 3.7036 |
| 3 | Edge-Ring | 1.9852 |
| 4 | Loc | 5.3485 |
| 5 | Near-full | **129.3419** |
| 6 | Random | 22.1973 |
| 7 | Scratch | 16.1096 |
| 8 | none | 0.1303 |

The mapping is saved as `data/processed/label_map.json`.

Class weights are computed as `total / (num_classes × class_count)`. The Near-full class gets a weight of 129 — the loss penalises errors on it 129× more than errors on `none`.

## Actual Split Sizes (after running pipeline)

| Split | Samples |
|---|---|
| Train | 121,064 |
| Val | 25,943 |
| Test | 25,943 |
| Total | 172,950 |

## Saved Files

| File | Size | Description |
|---|---|---|
| `X_train.npy` | 1,891 MB | (121064, 1, 64, 64) float32 |
| `X_val.npy` | 405 MB | (25943, 1, 64, 64) float32 |
| `X_test.npy` | 405 MB | (25943, 1, 64, 64) float32 |
| `y_train.npy` | 0.9 MB | (121064,) int64 |
| `y_val.npy` | 0.2 MB | (25943,) int64 |
| `y_test.npy` | 0.2 MB | (25943,) int64 |
| `label_map.json` | — | String↔index mapping |
