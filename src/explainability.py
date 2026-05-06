"""
src/explainability.py

Grad-CAM implementation for WaferCNN.

Target layer: model.features[8]  (last Conv2d, before BN+ReLU)
Output shape at that layer: (B, 128, 8, 8)

We target the Conv2d (not the ReLU) because ReLU(inplace=True) is
incompatible with register_full_backward_hook — it causes a RuntimeError
about inplace view modifications during backward. Conv2d has no inplace ops.

The spatial information (8x8) is identical; Grad-CAM math is unchanged.
The heatmap is upsampled to 64x64 so it overlays cleanly on the wafer map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.model import WaferCNN
from src.data_loader import DEFECT_CLASSES


# ── Grad-CAM core ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for WaferCNN.

    Usage
    -----
    cam = GradCAM(model)
    heatmap = cam(input_tensor, class_idx)   # input_tensor: (1, 1, 64, 64)
    cam.remove_hooks()
    """

    def __init__(self, model: WaferCNN, target_layer_idx: int = 8):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        target_layer = model.features[target_layer_idx]

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for one sample.

        Parameters
        ----------
        x         : (1, 1, 64, 64) float32 tensor
        class_idx : target class (None = argmax of prediction)

        Returns
        -------
        heatmap : (64, 64) float32 array, values in [0, 1]
        """
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward()

        # global-average-pool gradients → (C,)
        weights = self._gradients.mean(dim=(2, 3))[0]

        # weighted sum of activation maps → (H, W)
        activations = self._activations[0]          # (C, H, W)
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        cam = F.relu(cam)

        # upsample to input resolution
        cam = cam.unsqueeze(0).unsqueeze(0)         # (1, 1, H, W)
        cam = F.interpolate(cam, size=(64, 64), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_single(
    model: WaferCNN,
    x: torch.Tensor,
) -> tuple[int, float, np.ndarray]:
    """
    Run inference on one sample.

    Parameters
    ----------
    x : (1, 1, 64, 64) tensor

    Returns
    -------
    pred_idx  : predicted class index
    confidence: softmax probability of predicted class
    probs     : (9,) softmax probabilities for all classes
    """
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())
    return pred_idx, float(probs[pred_idx]), probs


# ── Visualization ─────────────────────────────────────────────────────────────

def overlay_heatmap(
    wafer_map: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto a wafer map.

    Parameters
    ----------
    wafer_map : (64, 64) float32 in [0, 1]
    heatmap   : (64, 64) float32 in [0, 1]
    alpha     : heatmap opacity

    Returns
    -------
    overlay : (64, 64, 3) uint8 RGB image
    """
    # grayscale wafer map → RGB
    base_rgb = np.stack([wafer_map] * 3, axis=-1)  # (64, 64, 3)

    # heatmap → RGB via jet colormap
    jet = cm.get_cmap("jet")
    heat_rgb = jet(heatmap)[:, :, :3]              # (64, 64, 3), float in [0,1]

    blended = (1 - alpha) * base_rgb + alpha * heat_rgb
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)


def plot_gradcam_grid(
    samples: list[dict],
    save_path: str | Path | None = None,
    title: str = "Grad-CAM Explanations",
) -> None:
    """
    Plot a grid of Grad-CAM results.

    Each entry in `samples` must have:
      wafer_map  : (64, 64) float32
      heatmap    : (64, 64) float32
      true_label : str
      pred_label : str
      confidence : float
      correct    : bool

    Layout per row: [wafer map | overlay]
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes[None, :]  # ensure 2D indexing

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for i, s in enumerate(samples):
        wmap    = s["wafer_map"]
        overlay = overlay_heatmap(wmap, s["heatmap"])
        correct = s["correct"]
        color   = "green" if correct else "red"

        axes[i, 0].imshow(wmap, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 0].set_title(f"True: {s['true_label']}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay, interpolation="nearest")
        verdict = "✓" if correct else "✗"
        axes[i, 1].set_title(
            f"{verdict} Pred: {s['pred_label']} ({s['confidence']:.0%})",
            fontsize=9, color=color,
        )
        axes[i, 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ── Batch analysis helper ─────────────────────────────────────────────────────

def collect_gradcam_samples(
    model: WaferCNN,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str] = DEFECT_CLASSES,
    n_per_class: int = 2,
    correct_only: bool = False,
    incorrect_only: bool = False,
    seed: int = 42,
) -> list[dict]:
    """
    Collect Grad-CAM samples across all classes.

    Parameters
    ----------
    n_per_class    : how many samples to return per class
    correct_only   : only include correctly classified samples
    incorrect_only : only include misclassified samples

    Returns
    -------
    list of sample dicts ready for plot_gradcam_grid
    """
    rng = np.random.default_rng(seed)
    grad_cam = GradCAM(model)
    results = []

    for class_idx, class_name in enumerate(class_names):
        indices = np.where(y == class_idx)[0]
        if len(indices) == 0:
            continue
        rng.shuffle(indices)

        collected = 0
        for idx in indices:
            if collected >= n_per_class:
                break

            x_np = X[idx]                           # (1, 64, 64)
            x_t  = torch.from_numpy(x_np).unsqueeze(0)  # (1, 1, 64, 64)

            pred_idx, confidence, _ = predict_single(model, x_t)
            is_correct = (pred_idx == class_idx)

            if correct_only   and not is_correct:
                continue
            if incorrect_only and is_correct:
                continue

            heatmap = grad_cam(x_t, class_idx=pred_idx)

            results.append({
                "wafer_map":   x_np[0],              # (64, 64)
                "heatmap":     heatmap,
                "true_label":  class_name,
                "pred_label":  class_names[pred_idx],
                "confidence":  confidence,
                "correct":     is_correct,
                "true_idx":    class_idx,
                "pred_idx":    pred_idx,
            })
            collected += 1

    grad_cam.remove_hooks()
    return results


def load_model(model_path: str | Path, num_classes: int = 9) -> WaferCNN:
    """Load a saved WaferCNN checkpoint."""
    model = WaferCNN(num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
