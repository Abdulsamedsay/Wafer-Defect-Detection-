# Modeling Decisions

> This page will be updated as modeling decisions are made.

---

## Framework

**PyTorch** was chosen as the primary deep learning framework.

Reason: Industry standard, flexible, strong community support, native support for custom training loops.

---

## Planned Modeling Order

1. Baseline model (non-deep-learning) to establish a reference point
2. Simple custom CNN
3. Evaluate — only move to transfer learning (ResNet18, EfficientNet-B0) if the simple CNN underperforms significantly

---

## Baseline Model Plan

- Input: Flattened wafer map (1D vector)
- Model options: Random Forest, Logistic Regression, SVM
- Primary choice: Random Forest (robust, handles some imbalance, fast to train)

Reason for baseline: A baseline makes the CNN comparison meaningful. Without it, we can't know how much deep learning actually helps.

---

## CNN Architecture Plan

```text
Conv2D(1, 32, kernel=3, padding=1) -> ReLU -> MaxPool(2)
Conv2D(32, 64, kernel=3, padding=1) -> ReLU -> MaxPool(2)
Conv2D(64, 128, kernel=3, padding=1) -> ReLU -> MaxPool(2)
Flatten
Linear(128 * 8 * 8, 256) -> ReLU -> Dropout(0.5)
Linear(256, num_classes)
```

*(Architecture will be refined during Phase 4)*

---

## Loss Function

Plan: CrossEntropyLoss with class weights to handle imbalance.

---

## Optimizer

Plan: Adam with learning rate 1e-3, reduce on plateau.

---

## Training Configuration (Experiment 003)

| Setting | Value | Reason |
|---|---|---|
| Batch size | 256 | Faster CPU throughput (fewer steps per epoch) |
| Optimizer | Adam, lr=1e-3 | Standard choice for CNNs |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) | Halve LR if val Macro F1 stalls |
| Early stopping | patience=7 | Stop if val Macro F1 doesn't improve |
| Max epochs | 30 | Upper bound — early stopping expected before this |
| Augmentation | Random H/V flips on train | Safe for wafer maps (patterns are symmetric) |
| Loss | CrossEntropyLoss with class weights | Handles 989x imbalance |

## Decisions Made

- **Primary metric for checkpoint saving:** Val Macro F1 (not accuracy or val loss)  
  Reason: accuracy is meaningless with 85% `none` dominance.
- **No rotation augmentation:** rotating an Edge-Ring still gives Edge-Ring, but rotating a Scratch changes its angle, which could confuse the model.
- **Evaluate on val set only during training:** running inference on the full 121k train set each epoch adds 5+ minutes per epoch on CPU. Val set (25k) is sufficient for monitoring.
