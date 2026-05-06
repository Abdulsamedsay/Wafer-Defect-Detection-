import json
from pathlib import Path

m = json.loads(Path("models/cnn_full_metrics.json").read_text())
print(f"accuracy       : {m['accuracy']:.4f}")
print(f"macro_f1       : {m['macro_f1']:.4f}")
print(f"weighted_f1    : {m['weighted_f1']:.4f}")
print(f"macro_precision: {m['macro_precision']:.4f}")
print(f"macro_recall   : {m['macro_recall']:.4f}")
print()
print("Per-class F1:")
for cls, v in m["per_class"].items():
    f1 = v["f1"]
    pr = v["precision"]
    re = v["recall"]
    su = v["support"]
    print(f"  {cls:<15} F1={f1:.3f}  P={pr:.3f}  R={re:.3f}  n={su}")
