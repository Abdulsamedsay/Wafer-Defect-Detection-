# Dashboard Design

> Phase 8 — Streamlit dashboard built and verified.

---

## Purpose

The dashboard presents the project as a usable AI decision-support tool, not just a technical demo. It should be understandable to an engineer without an AI background.

**Run command:**
```
streamlit run app/streamlit_app.py
```

**One-time setup (before first launch):**
```
python app/prepare_dashboard_data.py
```
This generates `models/cnn_full_metrics.json` and `models/demo_samples.npz`.

---

## Page Structure

### Page 1 — Project Overview

- Title + subtitle
- Business problem (manufacturing quality control context)
- System pipeline diagram (text)
- Model performance metrics (Macro F1, Accuracy, Weighted F1)
- Dataset summary (811k total, 172k labeled, 9 classes)
- Disclaimer: prototype, not production system

### Page 2 — Prediction (main interactive page)

- Class selector dropdown (Random + 9 defect classes)
- "Pick Sample" button (loads a test-set example from `demo_samples.npz`)
- Side-by-side: original wafer map | Grad-CAM overlay
- Risk card (colored HTML box: risk level, action, confidence note)
- Class probability bar chart (Plotly, 9 bars, predicted class highlighted)

### Page 3 — Model Performance

- 4 summary metric cards (Accuracy, Macro F1, Weighted F1, Macro Recall)
- Baseline vs CNN comparison table (SGD LogReg, Random Forest, WaferCNN)
- Confusion matrix image (`outputs/figures/confusion_matrix_cnn.png`)
- Per-class metrics table (Precision, Recall, F1, Support for all 9 classes)
- Average Grad-CAM heatmap per class

### Page 4 — Dataset Insights

- Dataset stats table (811k total, 172k labeled, 9 classes, etc.)
- Class distribution figure (`outputs/figures/class_distribution.png`)
- Sample wafer maps figure (`outputs/figures/sample_wafer_maps.png`)
- Key EDA findings (class imbalance, Macro F1 rationale, Scratch difficulty)
- Grad-CAM examples figure (`outputs/figures/gradcam_examples.png`)

---

## Architecture Decisions

| Decision | Rationale |
|---|---|
| `@st.cache_resource` for model + data | Load once, reuse across rerenders. Fast UX. |
| Demo samples pre-generated (45 samples) | Avoid loading 405MB test set inside the app. |
| Plotly for probability chart | Interactive hover, cleaner than `st.bar_chart`. |
| HTML risk card via `st.markdown` | Allows color-coded borders/backgrounds. |
| `NEAREST` interpolation for wafer map display | Preserves the categorical pixel structure. |
| Grad-CAM targets `features[8]` (Conv2d) | `features[10]` (ReLU inplace) causes backward hook error. |

---

## Files

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | Main dashboard (4 pages) |
| `app/prepare_dashboard_data.py` | One-time data prep script |
| `app/verify_app.py` | Logic verification without Streamlit server |
| `models/cnn_full_metrics.json` | Per-class metrics for performance page |
| `models/demo_samples.npz` | 5 samples per class (45 total) for prediction page |

---

## Design Principles

- Color-coded risk levels: green (Low `#2ecc71`), amber (Medium `#f39c12`), orange (High `#e67e22`), red (Critical `#e74c3c`)
- Sidebar for navigation
- Wide layout (`layout="wide"`)
- Professional language — not toy demo language
- Disclaimer always visible: prototype, not production system
