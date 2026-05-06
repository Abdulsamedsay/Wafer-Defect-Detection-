"""Quick verification that all dashboard logic works without the Streamlit server."""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.explainability import GradCAM, load_model, overlay_heatmap
from src.risk_scoring import score
from src.data_loader import DEFECT_CLASSES

model   = load_model(ROOT / "models" / "best_model.pth")
data    = np.load(str(ROOT / "models" / "demo_samples.npz"))
X_demo  = data["X"]
y_demo  = data["y"]
metrics = json.loads((ROOT / "models" / "cnn_full_metrics.json").read_text())

print("All resources loaded OK")
print(f"Demo samples : {len(X_demo)}")
mf1 = metrics["macro_f1"]
print(f"Macro F1     : {mf1:.4f}")

# Test one prediction
x_np = X_demo[0]
x_t  = torch.from_numpy(x_np).unsqueeze(0)
cam  = GradCAM(model)
with torch.no_grad():
    probs = torch.softmax(model(x_t), dim=1).squeeze().numpy()
pred_idx   = int(probs.argmax())
confidence = float(probs[pred_idx])
heatmap    = cam(x_t, class_idx=pred_idx)
cam.remove_hooks()
result  = score(pred_idx, confidence)
overlay = overlay_heatmap(x_np[0], heatmap)

print(f"Prediction   : {DEFECT_CLASSES[pred_idx]} ({confidence:.0%})")
print(f"Risk level   : {result.risk_level}")
print(f"Action       : {result.recommended_action[:60]}...")
print(f"Heatmap      : {heatmap.shape}  Overlay: {overlay.shape}")
print()
print("All dashboard logic verified OK")
