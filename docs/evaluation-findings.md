# Evaluation Findings

## Evaluation Principle

Because wafer defect classes are imbalanced (989x ratio), the model is evaluated using
per-class precision, recall, and F1-score. Accuracy alone is meaningless — a model
that predicts `none` for every wafer achieves 85% accuracy but fails on every defect.

**Primary metric: Macro F1**

---

## Final Model Results — WaferCNN (Test Set)

| Metric | Value |
|---|---|
| Accuracy | 0.7865 |
| **Macro F1** | **0.6413** |
| Weighted F1 | 0.8426 |
| Macro Precision | 0.5914 |
| Macro Recall | 0.7735 |

---

## Per-Class Results (Test Set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Center | 0.7106 | 0.9301 | **0.8056** | 644 |
| Donut | 0.7596 | 0.9518 | **0.8449** | 83 |
| Edge-Loc | 0.2523 | 0.7368 | 0.3759 | 779 |
| Edge-Ring | 0.9377 | 0.9635 | **0.9504** | 1,452 |
| Loc | 0.2377 | 0.4286 | 0.3058 | 539 |
| Near-full | 0.8261 | 0.8636 | **0.8444** | 22 |
| Random | 0.5857 | 0.9462 | 0.7235 | 130 |
| Scratch | 0.0258 | 0.3575 | **0.0481** | 179 |
| none | 0.9869 | 0.7830 | 0.8732 | 22,115 |

---

## Comparison vs Baselines

| Model | Accuracy | Macro F1 |
|---|---|---|
| SGD Logistic Regression | 0.8803 | 0.5515 |
| Random Forest (30k sub) | 0.9246 | 0.5385 |
| **WaferCNN** | **0.7865** | **0.6413** |

The CNN has lower accuracy than both baselines but higher Macro F1. This is expected and desirable — it means the CNN is successfully trading `none` confidence for defect detection sensitivity.

---

## Confusion Matrix

See: `outputs/figures/confusion_matrix_cnn.png`

Key patterns (from normalized confusion matrix):
- Edge-Ring, Near-full, Center, Donut: high on-diagonal values (well-classified)
- Scratch: very low diagonal — mostly predicted as `none` or Edge-Loc
- Edge-Loc: low diagonal — confused with `none` and Edge-Ring
- Loc: moderate diagonal — confused with `none`

---

## Classes That Are Difficult

| Class | F1 | Why |
|---|---|---|
| Scratch | 0.048 | Thin linear pattern — hard to distinguish from noise at 64×64 |
| Edge-Loc | 0.376 | Partial edge ring — ambiguous between Edge-Ring and none |
| Loc | 0.306 | Small local cluster — easily confused with none |

---

## Misclassified Sample Analysis

*(To be filled in during Phase 5 evaluation notebook)*

---

## What This Means for Engineering

- The model reliably identifies **Edge-Ring, Center, and Donut** patterns — the most common production defects
- **Near-full** detection is strong (F1=0.84) despite only 22 test samples — the spatial signature is distinctive
- **Scratch** detection needs improvement — false negatives mean physical damage could be missed
- The risk scoring layer compensates partially: even when Scratch is misclassified as Edge-Loc, the risk level stays High
