# Experiment Log

> Chronological record of all model experiments.

---

## Experiment 001 — SGD Logistic Regression (Baseline)

### Goal
Establish a fast linear baseline on the full training set.

### Setup
- Model: `SGDClassifier(loss='log_loss')` — mathematically equivalent to Logistic Regression, trained with online gradient descent
- Input: Flattened 64×64 wafer maps (4,096 features per sample)
- Training samples: 121,064 (full labeled train set)
- Class weighting: `class_weight='balanced'`
- Max iterations: 50 epochs
- Training time: ~2.2 minutes

### Results (Test Set)

| Metric | Value |
|---|---|
| Accuracy | 0.8803 |
| Macro F1 | **0.5515** |
| Weighted F1 | 0.8974 |
| Macro Precision | 0.5891 |
| Macro Recall | 0.6257 |

**Per-class F1 (test set):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Center | 0.8187 | 0.8276 | **0.8232** | 644 |
| Donut | 0.2985 | 0.9639 | 0.4558 | 83 |
| Edge-Loc | 0.3228 | 0.6534 | 0.4321 | 779 |
| Edge-Ring | 0.9005 | 0.9036 | **0.9020** | 1,452 |
| Loc | 0.2734 | 0.1948 | 0.2275 | 539 |
| Near-full | 1.0000 | 0.2727 | 0.4286 | 22 |
| Random | 0.6591 | 0.6692 | 0.6641 | 130 |
| Scratch | 0.0541 | 0.2346 | **0.0880** | 179 |
| none | 0.9744 | 0.9117 | 0.9420 | 22,115 |

### Observations
- High accuracy (88%) is misleading — driven entirely by the `none` class
- Edge-Ring and Center are well-separated linearly (F1 > 0.82)
- Scratch is nearly unsolvable linearly (F1 = 0.088) — needs spatial feature learning
- Loc has very low recall — confused with none and Edge-Loc
- The baseline Macro F1 target the CNN must beat: **0.5515**

### Next step
- Random Forest experiment with stratified subset

---

## Experiment 002 — Random Forest (30k Stratified Subset, Baseline)

### Goal
Test a non-linear baseline using tree-based ensembles on a representative subset.

### Setup
- Model: `RandomForestClassifier(n_estimators=100)`
- Input: Flattened 64×64 wafer maps (4,096 features)
- Training samples: 29,996 (stratified subset of train set, ~25%)
- Class weighting: `class_weight='balanced'`
- Training time: ~2 minutes

### Results (Test Set)

| Metric | Value |
|---|---|
| Accuracy | 0.9246 |
| Macro F1 | **0.5385** |
| Weighted F1 | 0.8993 |
| Macro Precision | 0.9312 |
| Macro Recall | 0.4693 |

**Per-class F1 (test set):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Center | 0.9387 | 0.6661 | **0.7793** | 644 |
| Donut | 0.9362 | 0.5301 | 0.6769 | 83 |
| Edge-Loc | 1.0000 | 0.0308 | **0.0598** | 779 |
| Edge-Ring | 0.9953 | 0.8705 | **0.9287** | 1,452 |
| Loc | 0.9800 | 0.0909 | 0.1664 | 539 |
| Near-full | 0.9375 | 0.6818 | 0.7895 | 22 |
| Random | 0.6719 | 0.3308 | 0.4433 | 130 |
| Scratch | 1.0000 | 0.0223 | **0.0437** | 179 |
| none | 0.9210 | 1.0000 | 0.9589 | 22,115 |

### Observations
- Very high precision across defect classes — but extremely low recall
- RF almost always predicts `none` when uncertain (maximizes accuracy, kills recall on rare classes)
- Scratch F1 = 0.044 — nearly zero, despite perfect precision when it does predict Scratch
- Edge-Loc F1 = 0.060 — same pattern
- Near-full surprisingly good (F1 = 0.79) despite only 22 test samples
- **Macro F1 slightly worse than LR** despite higher accuracy — confirms accuracy is misleading

### Next step
- Build CNN — must beat Macro F1 > 0.5515 (LR baseline)

---

## Hard Classes Summary (from Baseline Phase)

These three classes are hard for **both** baseline models:

| Class | LR F1 | RF F1 | Why Hard |
|---|---|---|---|
| Scratch | 0.088 | 0.044 | Linear pattern — needs spatial/edge detection |
| Loc | 0.228 | 0.166 | Small local cluster — easily confused with noise |
| Edge-Loc | 0.432 | 0.060 | Partial edge — overlaps with Edge-Ring and none |

The CNN must pay special attention to these classes. If it fails here too, augmentation or focal loss may help.

---

## Baseline Target for CNN

| Metric | Target |
|---|---|
| Macro F1 | > 0.5515 (beat LR) |
| Accuracy | > 0.8803 |
| Scratch F1 | > 0.088 |
| Loc F1 | > 0.228 |
| Edge-Loc F1 | > 0.432 |

---

## Experiment 003 — WaferCNN (Custom CNN, Full Dataset)

### Goal
Train a custom 3-block CNN on the full preprocessed dataset and beat the baseline Macro F1 of 0.5515.

### Setup
- Model: WaferCNN (3 conv blocks, BatchNorm, Dropout 0.5) — 2,192,841 parameters
- Input: (B, 1, 64, 64) normalized wafer maps
- Training samples: 121,064 (full labeled train set)
- Optimizer: Adam, lr=1e-3
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Loss: CrossEntropyLoss with class weights
- Augmentation: random horizontal + vertical flips
- Early stopping: patience=7 on val Macro F1
- Batch size: 256
- Device: CPU (~2.1 min/epoch)

### Results (Test Set)

| Metric | Value | vs Baseline |
|---|---|---|
| Accuracy | 0.7865 | -0.0938 |
| **Macro F1** | **0.6413** | **+0.0898** |
| Weighted F1 | 0.8426 | -0.0548 |
| Macro Precision | 0.5914 | — |
| Macro Recall | 0.7735 | — |

**Per-class F1 (test set):**

| Class | Baseline F1 | CNN F1 | Delta | Support |
|---|---|---|---|---|
| Center | 0.8232 | 0.8056 | -0.018 | 644 |
| Donut | 0.4558 | **0.8449** | **+0.389** | 83 |
| Edge-Loc | 0.4321 | 0.3759 | -0.056 | 779 |
| Edge-Ring | 0.9020 | **0.9504** | +0.048 | 1,452 |
| Loc | 0.2275 | 0.3058 | +0.078 | 539 |
| Near-full | 0.4286 | **0.8444** | **+0.416** | 22 |
| Random | 0.6641 | 0.7235 | +0.059 | 130 |
| Scratch | 0.0880 | 0.0481 | -0.040 | 179 |
| none | 0.9420 | 0.8732 | -0.069 | 22,115 |

### Observations
- **Macro F1 = 0.6413 — beats baseline by +0.090.** Experiment successful.
- Accuracy dropped (79% vs 88%) because CNN now actively classifies defects instead of defaulting to `none`. This is the correct behaviour — lower accuracy, higher Macro F1.
- **Donut +0.39, Near-full +0.42** — massive gains on the rarest classes. CNN learned spatial structure baselines couldn't.
- **Scratch F1 = 0.048** — still failing. Linear scratch patterns need edge-detection features. Possible fix: augmentation with rotations, or Grad-CAM analysis of what the model sees.
- **Edge-Loc F1 = 0.376** — still weak. Partial edge ring overlaps with Edge-Ring and none — hard to distinguish.
- High recall on all classes (0.77 macro recall) but low precision on hard classes = model over-predicts defects.

### Saved files
- `models/best_model.pth` (8.8 MB)
- `models/cnn_test_metrics.json`
- `outputs/figures/confusion_matrix_cnn.png`

### Next step
- Phase 5: Full evaluation notebook with confusion matrix analysis and misclassified examples
- Phase 6: Grad-CAM explainability — start with Scratch to understand why it fails
