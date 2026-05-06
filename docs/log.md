# Project Log

> Append-only chronological record of all project activity.
> Format: `## [YYYY-MM-DD] type | Short Title`

---

## [2026-05-05] portfolio | README and Portfolio Positioning Finalized

### What happened
Rewrote README.md with real metrics, full methodology, per-class F1 table, and project structure. Rewrote docs/portfolio-positioning.md with sharpened CV bullets, a LinkedIn post draft with real numbers, and interview Q&A covering the key technical decisions.

### Files changed
- `README.md` — fully rewritten with real results and complete methodology
- `docs/portfolio-positioning.md` — CV bullets (A and B), LinkedIn post, 5 interview Q&A

### Key decisions
- Lead with Macro F1=0.686, not accuracy (86.6% is misleading given the 85% baseline)
- The "+0.134 over baseline" framing makes the CNN improvement concrete and honest
- Scratch (F1=0.102) is called out explicitly — hiding weak results is worse than explaining them
- Interview answers include specific numbers: 989x imbalance, +0.134 F1, classes by name

### Next step
- Project is complete for portfolio. Optional next steps:
  - Push to GitHub (create repo, `git init`, add remote)
  - Deploy dashboard to Streamlit Cloud
  - ResNet18 transfer learning experiment (would likely improve Scratch and Edge-Loc)

---

## [2026-05-05] dashboard | Streamlit Dashboard Built and Verified

### What happened
Built the full Streamlit dashboard (`app/streamlit_app.py`). 4 pages: Project Overview, Prediction, Model Performance, Dataset Insights. All logic verified end-to-end without the server. Pre-generated `models/demo_samples.npz` (45 samples, 5 per class) and `models/cnn_full_metrics.json` (per-class F1) via `app/prepare_dashboard_data.py`.

### Files changed
- `app/streamlit_app.py` — created (main dashboard, 4 pages)
- `app/prepare_dashboard_data.py` — created (one-time data prep script)
- `app/verify_app.py` — created (logic test without Streamlit server)
- `models/cnn_full_metrics.json` — generated (per-class metrics, Macro F1=0.6856)
- `models/demo_samples.npz` — generated (45 samples for prediction page)
- `docs/dashboard-design.md` — fully written

### Key decisions
- Demo samples pre-generated to avoid loading 405MB test set at startup
- Grad-CAM runs live on each "Pick Sample" click — instant on CPU (single image)
- Risk card uses HTML via `st.markdown(unsafe_allow_html=True)` for color styling
- Plotly bar chart for class probabilities (interactive hover)
- `@st.cache_resource` for model, demo samples, and metrics — loaded once

### Next step
- Launch: `streamlit run app/streamlit_app.py`
- Update README with dashboard section and final metrics
- Polish portfolio-positioning.md for LinkedIn / CV

---

## [2026-05-05] decision | Yield-Risk Scoring Module Complete

### What happened
Built `src/risk_scoring.py` — a risk layer that maps (predicted class, confidence) to a structured RiskResult. All 7 self-test cases pass. The module is ready to be consumed by the Streamlit dashboard.

### Files changed
- `src/risk_scoring.py` — created (`score`, `score_batch`, `print_risk_table`, `RiskResult` dataclass)
- `docs/risk-scoring-notes.md` — fully written with mapping table, confidence adjustment rationale, code examples, and portfolio statement

### Key decisions
- Confidence threshold set at 70% — below this, risk is elevated by one level
- `Near-full` stays at Critical regardless of confidence (cannot go higher)
- `RiskResult` is a dataclass with `.as_dict()` for easy Streamlit consumption
- Colors are hex codes so the dashboard can use them directly in `st.markdown` or metric styling

### Next step
- Phase 8: Streamlit dashboard (`app/streamlit_app.py`)
  - Page 1: Project Overview
  - Page 2: Prediction (upload wafer map, get class + risk + Grad-CAM)
  - Page 3: Model Performance (confusion matrix, per-class metrics)
  - Page 4: Dataset Insights (class distribution, sample maps)

---

## [2026-05-05] explainability | Grad-CAM Phase Complete — All 9 Classes Explained

### What happened
Implemented Grad-CAM explainability for WaferCNN. Targeted the last Conv2d layer (`model.features[8]`, output 128×8×8) to avoid a `RuntimeError` caused by `ReLU(inplace=True)` conflicting with `register_full_backward_hook`. Heatmaps are upsampled to 64×64 and overlaid on the original wafer map. Generated 4 figure sets covering correct predictions, misclassifications, a portfolio figure, and average attention per class.

### Files changed
- `src/explainability.py` — created (`GradCAM`, `predict_single`, `overlay_heatmap`, `plot_gradcam_grid`, `collect_gradcam_samples`, `load_model`)
- `notebooks/05_evaluation.ipynb` — created (full Grad-CAM workflow)
- `outputs/figures/gradcam_correct.png` — 1 correct sample per class (9 rows)
- `outputs/figures/gradcam_incorrect.png` — misclassified samples (17 rows)
- `outputs/figures/gradcam_examples.png` — main portfolio figure (9 rows)
- `outputs/figures/gradcam_avg_per_class.png` — average attention heatmap per class
- `docs/explainability-notes.md` — fully written with method, layer rationale, expected findings, and limitations

### Key decisions
- Target `model.features[8]` (Conv2d) not `features[10]` (ReLU) — ReLU(inplace=True) is incompatible with full_backward_hook
- All 9 defect classes have Grad-CAM samples in the test set — no classes missing
- 17 misclassified samples collected; hard classes (Scratch, Edge-Loc) well represented
- Average heatmap figure (N=20 per class) is the most stable view of what the model learned per pattern

### Next step
- Phase 7: Risk scoring (`src/risk_scoring.py`) — map defect class + confidence → risk level + recommended action
- Phase 8: Streamlit dashboard (`app/streamlit_app.py`)

---

## [2026-05-03] training | WaferCNN Training Complete — Beats Baseline

### What happened
Trained WaferCNN (3 conv blocks, 2.2M parameters) on the full 121k training set with class-weighted CrossEntropyLoss, Adam optimizer, and early stopping. Evaluated the best checkpoint on the test set. CNN beats baseline Macro F1 by +0.090 (0.6413 vs 0.5515). Training took ~39 minutes on CPU.

### Files changed
- `src/dataset.py` — created (PyTorch Dataset + DataLoaders)
- `src/model.py` — created (WaferCNN architecture)
- `src/train.py` — created (training loop with early stopping)
- `notebooks/04_cnn_training.ipynb` — created
- `models/best_model.pth` — saved (8.8 MB)
- `models/cnn_test_metrics.json` — saved
- `outputs/figures/confusion_matrix_cnn.png` — saved
- `docs/experiment-log.md` — Experiment 003 filled in
- `docs/evaluation-findings.md` — filled with actual results
- `docs/modeling-decisions.md` — training config documented

### Key decisions
- batch_size=256 for CPU throughput
- Checkpoint saved by val Macro F1 (not accuracy or val loss)
- Augmentation: H/V flips only (no rotation — preserves defect orientation meaning)
- Scratch F1=0.048 and Edge-Loc F1=0.376 are the two weak spots — flag for Grad-CAM analysis

### Next step
- Phase 6: Grad-CAM explainability (`src/explainability.py`, `notebooks/05_evaluation.ipynb`)
- Phase 7: Risk scoring module (`src/risk_scoring.py`)
- Phase 8: Streamlit dashboard

---

## [2026-05-03] baseline | Completed Baseline Model Experiments

### What happened
Trained two baseline models (SGD Logistic Regression on full training set, Random Forest on 30k stratified subset). Both evaluated on the full test set with per-class metrics. Identified three hard classes the CNN must address. Established benchmark Macro F1 = 0.5515 that the CNN must beat.

Also built `src/dataset.py`, `src/model.py`, and `src/train.py` while baseline was training — CNN code is ready.

### Files changed
- `src/evaluate.py` — created (shared evaluation functions for all experiments)
- `src/dataset.py` — created (PyTorch Dataset + DataLoaders)
- `src/model.py` — created (WaferCNN architecture)
- `src/train.py` — created (training loop with early stopping)
- `notebooks/03_baseline_model.ipynb` — created
- `models/baseline_logreg.pkl` — saved
- `models/baseline_rf.pkl` — saved
- `outputs/figures/confusion_matrix_logreg.png` — saved
- `outputs/figures/confusion_matrix_rf.png` — saved
- `docs/experiment-log.md` — filled with real results

### Key decisions
- SGDClassifier used instead of batch SAGA LogReg (SAGA on 121k×4096 is impractical — took 30+ minutes without convergence; SGD equivalent trains in 2.2 minutes)
- Macro F1 = 0.5515 is the CNN target (not accuracy)
- Hard classes identified: Scratch (F1=0.088), Loc (F1=0.228), Edge-Loc (F1=0.432)
- Random Forest has near-perfect precision but near-zero recall on rare classes — confirms that accuracy alone is deeply misleading

### Next step
- Run CNN training: `src/train.py` + `notebooks/04_cnn_training.ipynb`

---

## [2026-05-03] preprocessing | Completed Full Preprocessing Pipeline

### What happened
Built and ran the complete Phase 2 preprocessing pipeline. Loaded 172,950 labeled wafer maps, resized all to 64×64 using nearest-neighbor interpolation, normalized to float32, encoded labels, applied stratified 70/15/15 split, computed class weights, and saved 7 files to `data/processed/`.

### Files changed
- `src/data_loader.py` — created (pickle compat + label extraction)
- `src/preprocessing.py` — created (resize, normalize, split, save, class weights)
- `notebooks/02_preprocessing.ipynb` — created
- `data/processed/X_train.npy` — 1,891 MB
- `data/processed/X_val.npy` — 405 MB
- `data/processed/X_test.npy` — 405 MB
- `data/processed/y_train.npy`, `y_val.npy`, `y_test.npy`
- `data/processed/label_map.json`
- `docs/preprocessing-decisions.md` — finalized with actual numbers

### Key decisions
- 64×64 resize with nearest-neighbor (categorical values — no bilinear)
- Normalize {0,1,2} → {0.0, 0.5, 1.0} by dividing by 2.0
- Channel-first shape: (N, 1, 64, 64) matching PyTorch convention
- Class weight for Near-full = 129.3x — extreme penalty for the rarest class
- Data stored as float32 .npy files (~2.7 GB total on disk)

### Next step
- Build `notebooks/03_baseline_model.ipynb` (Random Forest on flattened maps)
- This establishes the non-deep-learning benchmark

---

## [2026-05-03] bugfix | Resolved Legacy Pickle Compatibility for WM-811K

### What happened
Discovered that `pd.read_pickle()` cannot load LSWMD.pkl with modern pandas (3.x) because the file was created with pandas 0.x (Python 2). Two separate errors occurred: `ModuleNotFoundError: No module named 'pandas.indexes'` and `UnicodeDecodeError` on the fallback path.

Fixed by loading directly with `pickle.load(f, encoding='latin1')` and patching `sys.modules` to remap old pandas namespace paths.

### Files changed
- `eda_runner.py` (temporary EDA script)
- `docs/errors-and-fixes.md`

### Key decisions
- Never use `pd.read_pickle()` for LSWMD.pkl — always use `pickle.load(..., encoding='latin1')`
- The fix will be baked into `src/data_loader.py` in Phase 2

### Next step
- Proceed with EDA findings

---

## [2026-05-03] eda | Completed Exploratory Data Analysis

### What happened
Ran full EDA on the WM-811K dataset (811,457 wafer maps). Extracted real class distribution, wafer map size statistics, and generated figures. Key finding: extreme class imbalance (989x ratio, `none` class dominates at 85.2%). Preprocessing decisions finalized based on findings.

### Files changed
- `docs/eda-findings.md` — filled with actual numbers
- `docs/dataset-notes.md` — updated with real column names, encoding, typo note
- `docs/preprocessing-decisions.md` — decisions finalized (64×64, nearest-neighbor, stratified split, class weights)
- `docs/errors-and-fixes.md` — two errors documented
- `docs/open-questions.md` — updated with decisions made
- `outputs/figures/class_distribution.png` — generated
- `outputs/figures/sample_wafer_maps.png` — generated

### Key decisions
- **Resize to 64×64** using nearest-neighbor interpolation (categorical values)
- **Drop 638,507 unlabeled samples** — only 172,950 labeled samples used for training
- **Class weights in CrossEntropyLoss** as primary imbalance strategy
- **Stratified 70/15/15 split** to ensure rare classes appear in all splits
- **Accuracy alone is not a valid metric** — 989x imbalance makes it meaningless

### Next step
- Build `src/data_loader.py` with the compatibility fix
- Build `notebooks/02_preprocessing.ipynb`
- Implement resizing, normalization, label encoding, and train/val/test split

---

## [2026-05-03] setup | Created Initial Repository Structure

### What happened
Created the full project folder structure and initialized all core project files. This marks the start of the Wafer Defect Detection & Yield Risk Dashboard project.

The following was completed:
- All project directories created (`data/`, `notebooks/`, `src/`, `app/`, `models/`, `outputs/`, `docs/`, `sources/`)
- All required wiki pages initialized (index, log, project-overview, dataset-notes, eda-findings, preprocessing-decisions, modeling-decisions, experiment-log, evaluation-findings, explainability-notes, risk-scoring-notes, dashboard-design, errors-and-fixes, portfolio-positioning, open-questions)
- `requirements.txt` created with all required Python dependencies
- `.gitignore` created (excludes raw data, model weights, and generated figures)
- `README.md` initial draft created with project description, dataset overview, and placeholder sections
- `src/__init__.py` created to make `src` a Python package
- `sources/references.md` created
- `notebooks/01_data_exploration.ipynb` created with full EDA workflow

### Files changed
- `requirements.txt`
- `.gitignore`
- `README.md`
- `src/__init__.py`
- `docs/index.md`
- `docs/log.md`
- `docs/project-overview.md`
- `docs/dataset-notes.md`
- `docs/eda-findings.md`
- `docs/preprocessing-decisions.md`
- `docs/modeling-decisions.md`
- `docs/experiment-log.md`
- `docs/evaluation-findings.md`
- `docs/explainability-notes.md`
- `docs/risk-scoring-notes.md`
- `docs/dashboard-design.md`
- `docs/errors-and-fixes.md`
- `docs/portfolio-positioning.md`
- `docs/open-questions.md`
- `sources/references.md`
- `notebooks/01_data_exploration.ipynb`

### Key decisions
- Use PyTorch as the primary deep learning framework (industry standard, flexible custom training)
- Use Streamlit for the dashboard (fast to build, professional enough for portfolio)
- Start with EDA before any modeling — no model training until data is fully understood
- Use WM-811K (LSWMD.pkl) as the primary dataset
- Project is positioned as an end-to-end industrial AI decision-support tool, not a simple classifier
- Wiki will be maintained as a persistent knowledge base throughout the project

### Next step
- Download the WM-811K dataset (`LSWMD.pkl`) and place it at `data/raw/LSWMD.pkl`
- Run `notebooks/01_data_exploration.ipynb` from top to bottom
- Fill in `docs/eda-findings.md` with actual findings
- Update `docs/index.md` project status table
