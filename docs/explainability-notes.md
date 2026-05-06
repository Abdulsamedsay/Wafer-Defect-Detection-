# Explainability Notes

> Phase 6 — Grad-CAM implemented and ready to run.

---

## Why Explainability Matters Here

In industrial environments like semiconductor manufacturing, a prediction alone is often not enough. Engineers need to understand *why* the model assigned a particular defect class.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which spatial regions of the wafer map most influenced the model's prediction. This makes the AI output useful for engineering review — not just a black-box label.

For example:
- An **Edge-Ring** prediction should highlight the outer ring of the wafer.
- A **Center** prediction should highlight the center region.
- A **Scratch** prediction should highlight a thin diagonal or curved line.

If the heatmap aligns with the actual defect region, it confirms the model learned the right visual features. If it doesn't, it reveals a failure mode.

---

## Implementation

### Method

**Grad-CAM** applied to the last convolutional layer of WaferCNN.

### Target Layer

`model.features[10]` — ReLU activation after the 3rd conv block (last BatchNorm).

Output shape at that layer: **(B, 128, 8, 8)**

The 8×8 feature maps are upsampled to 64×64 using bilinear interpolation for overlay.

### Algorithm

1. Forward pass → capture feature maps at target layer (hook)
2. Compute class score for target class
3. Backward pass → capture gradients at target layer (backward hook)
4. Global-average-pool gradients over spatial dims → weight vector (128,)
5. Weighted sum of feature maps → raw CAM (8×8)
6. Apply ReLU → keep only positive contributions
7. Upsample to 64×64 (bilinear)
8. Normalize to [0, 1]

### Code

- `src/explainability.py` — `GradCAM` class, `predict_single`, `overlay_heatmap`, `plot_gradcam_grid`, `collect_gradcam_samples`, `load_model`
- `notebooks/05_evaluation.ipynb` — full analysis with 6 output figures

---

## Output Figures

| File | Description |
|------|-------------|
| `outputs/figures/gradcam_sanity.png` | Single-sample sanity check |
| `outputs/figures/gradcam_correct.png` | 1 correct prediction per class |
| `outputs/figures/gradcam_incorrect.png` | Misclassified samples, all classes |
| `outputs/figures/gradcam_scratch_deep_dive.png` | Deep dive into Scratch (F1=0.048) |
| `outputs/figures/gradcam_examples.png` | **Main portfolio figure** — 1 sample per class |
| `outputs/figures/gradcam_avg_per_class.png` | Average heatmap per class (N=20 samples) |

---

## Expected Findings

### What the heatmaps should show (ideal)

| Defect Class | Expected Highlighted Region |
|---|---|
| Center | Center of the wafer |
| Donut | Ring around center |
| Edge-Ring | Outer edge ring |
| Edge-Loc | Local region on the edge |
| Loc | Isolated local cluster |
| Scratch | Thin diagonal/curved line |
| Near-full | Almost entire wafer |
| Random | Scattered points |
| None / Normal | No specific region |

### Why Scratch is hardest

Scratch (F1=0.048) is the model's weakest class. Likely reasons:
- Thin linear structure — easily confused with Edge-Loc or Loc at 64×64 resolution
- Low support in training data — extreme class weight but insufficient examples
- Downsampling from original wafer maps to 64×64 may erase fine scratch detail

The Grad-CAM deep dive in the notebook is designed to confirm or refute these hypotheses.

---

## Limitations

1. **Resolution**: Wafer maps are resized to 64×64. Fine-grained defects (especially Scratch) lose detail. Grad-CAM heatmaps are inherently coarse at this resolution.
2. **Grad-CAM is class-discriminative, not faithful**: It shows what regions influenced the predicted class, not a ground-truth defect map. The heatmap is only as good as the model's learned features.
3. **Single-layer view**: Grad-CAM uses only the last conv layer. Mid-layer features (texture, local patterns) are not visualized.
4. **Prototype context**: This is a portfolio project. The explanations are illustrative — not validated for production engineering use.

---

## Portfolio Statement

> Grad-CAM was applied to the last convolutional layer to generate spatial attention maps for each prediction. In industrial AI contexts, visual explanations like these are essential for engineering review — they allow operators to confirm whether the model is focusing on the actual defect region rather than spurious correlations.
