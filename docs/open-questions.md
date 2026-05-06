# Open Questions

> Research questions, future improvements, unresolved decisions, and ideas.
> This page grows as the project evolves.

---

## Technical Questions

- What is the best resize target for wafer maps? **Decided: 64×64** (32×32 loses too much spatial detail). Still open: does 128×128 meaningfully improve results?
- Resizing method: **Decided: nearest-neighbor** (categorical values, not continuous). Confirmed correct.
- Does augmentation help or hurt on structured defect patterns like Edge-Ring? **Open** — rotation could destroy pattern meaning (e.g., a rotated Edge-Ring is still Edge-Ring, but a rotated Scratch changes direction). Horizontal/vertical flips may be safe.
- Is transfer learning (ResNet18) meaningful for single-channel wafer maps? These are not natural images. **Open** — ResNet was trained on RGB photos, not die maps. May need adaptation (convert 1-channel to 3-channel by repeating).
- How do we handle the 638k unlabeled samples? **Current plan: ignore for supervised training.** Future: self-supervised pretraining (SimCLR, BYOL).

---

## Modeling Questions

- At what accuracy gap between baseline and CNN is transfer learning worth the added complexity?
- Should Scratch be treated as a separate category or grouped with Edge-Loc given its similar risk profile?
- Can we use the confidence score reliably as a risk modifier, or does it need calibration first?

---

## Dataset Questions

- Are the train/test labels in the original dataset reliable, or should we create our own split?
- How were the defect labels determined? Manual labeling? Algorithm? This affects label noise estimates.

---

## Future Improvements

- [ ] Try ResNet18 or EfficientNet-B0 if custom CNN underperforms
- [ ] Add data augmentation (careful with rotations for structured patterns)
- [ ] Explore semi-supervised learning using the 638k unlabeled samples
- [ ] Add MLflow for experiment tracking
- [ ] Deploy Streamlit dashboard to Streamlit Cloud
- [ ] Add FastAPI endpoint for model inference
- [ ] Add anomaly detection for defect types not seen during training
- [ ] Explore synthetic wafer defect generation for minority class augmentation

---

## Portfolio Questions

- When is the project ready to post on LinkedIn? After Phase 5 (evaluation results available)?
- Should the dashboard include a live camera feed simulation for demo effect?
