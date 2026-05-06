"""
app/prepare_dashboard_data.py

One-time script: generates the two files the Streamlit dashboard needs.

  models/cnn_full_metrics.json  — per-class metrics for the performance page
  models/demo_samples.npz       — 5 test samples per class for the prediction page

Run once before launching the dashboard:
  python app/prepare_dashboard_data.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explainability import load_model
from src.preprocessing import load_processed, PROCESSED_DIR
from src.data_loader import DEFECT_CLASSES
from src.evaluate import compute_metrics

MODEL_PATH = Path("models/best_model.pth")
METRICS_PATH = Path("models/cnn_full_metrics.json")
DEMO_PATH = Path("models/demo_samples.npz")
BATCH_SIZE = 512
N_DEMO = 5


def run():
    model = load_model(MODEL_PATH)
    _, _, X_test, _, _, y_test = load_processed(PROCESSED_DIR)
    print(f"Test set: {X_test.shape[0]:,} samples")

    # Full test-set inference
    print("Running inference on test set ...")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            batch = torch.from_numpy(X_test[i : i + BATCH_SIZE])
            preds.append(model(batch).argmax(dim=1).cpu().numpy())
            if i % 10000 == 0:
                print(f"  {i:>6} / {len(X_test)}")
    y_pred = np.concatenate(preds)

    # Save per-class metrics
    metrics = compute_metrics(y_test, y_pred, DEFECT_CLASSES)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    mf1 = metrics["macro_f1"]
    acc = metrics["accuracy"]
    print(f"Saved {METRICS_PATH}  (Macro F1={mf1:.4f}  Accuracy={acc:.4f})")

    # Save demo samples
    rng = np.random.default_rng(42)
    demo_X, demo_y = [], []
    for class_idx in range(len(DEFECT_CLASSES)):
        idxs = np.where(y_test == class_idx)[0]
        chosen = rng.choice(idxs, size=min(N_DEMO, len(idxs)), replace=False)
        demo_X.append(X_test[chosen])
        demo_y.append(y_test[chosen])
    demo_X = np.concatenate(demo_X)
    demo_y = np.concatenate(demo_y)
    np.savez_compressed(str(DEMO_PATH), X=demo_X, y=demo_y)
    kb = demo_X.nbytes / 1024
    print(f"Saved {DEMO_PATH}  ({len(demo_y)} samples, {kb:.0f} KB uncompressed)")


if __name__ == "__main__":
    run()
