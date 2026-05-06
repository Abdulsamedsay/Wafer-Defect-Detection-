# Portfolio Positioning

> How to present this project on GitHub, CV, LinkedIn, and in interviews.
> All numbers are from the actual test set evaluation — do not exaggerate.

---

## One-Line Description

> An end-to-end AI decision-support tool for wafer defect classification, visual explainability, and yield-risk estimation in semiconductor manufacturing.

**Not:**
> A simple wafer image classifier.

---

## GitHub README Tagline

```
Wafer Defect Detection & Yield Risk Dashboard
An end-to-end computer vision system for semiconductor manufacturing quality control.
PyTorch · Grad-CAM · Streamlit · WM-811K dataset
```

---

## CV Bullet Point

**Version A — concise (one line):**
```
Built an end-to-end wafer defect classification system (PyTorch CNN, Macro F1=0.686 on 9
classes) with Grad-CAM explainability and a yield-risk scoring layer, deployed as a Streamlit
decision-support dashboard.
```

**Version B — detailed (3 bullets):**
```
• Trained a custom PyTorch CNN on 172k semiconductor wafer maps (WM-811K dataset),
  achieving Macro F1=0.686 across 9 defect classes with 989× class imbalance — a +0.134
  improvement over the logistic regression baseline.

• Implemented Grad-CAM to generate spatial attention maps for each prediction, enabling
  engineers to verify which wafer regions drove the model's output.

• Built a yield-risk scoring layer mapping predicted defect class and model confidence to
  one of four risk levels (Low/Medium/High/Critical) with plain-English engineering actions,
  delivered through a 4-page interactive Streamlit dashboard.
```

---

## LinkedIn Post

```
I just finished an end-to-end AI project inspired by semiconductor manufacturing and the high-tech ecosystem around companies like ASML.

**Wafer Defect Detection & Yield Risk Dashboard**

What it does:
→ Classifies wafer defect patterns from wafer map images (9 defect classes)
→ Generates Grad-CAM heatmaps showing which regions of the wafer drove the prediction
→ Assigns a yield-risk level (Low / Medium / High / Critical) with a recommended engineering action
→ Delivers everything through an interactive Streamlit dashboard

What made it technically interesting:

The dataset (WM-811K, 811k wafer maps) has a 989× class imbalance — the "no defect" class makes up 85% of labeled data. A model that always predicts "no defect" gets 85% accuracy. That's why I used Macro F1 as the primary metric — it weights all 9 classes equally regardless of how many samples they have.

The custom CNN achieved Macro F1 = 0.686, a +0.134 improvement over the logistic regression baseline. Edge-Ring and Center defects are detected reliably (F1 = 0.96 and 0.84). Scratch is the hardest class (F1 = 0.10) — thin linear defects lose structural detail when resized to 64×64, and the class has very few training examples even with class weighting.

The Grad-CAM heatmaps confirmed the model learned the right features: Edge-Ring predictions focus on the outer ring, Center predictions focus on the center zone. For Scratch, the attention is diffuse — which explains the low F1 before you even look at the metric.

This is the kind of AI I want to build — not just a classification notebook, but a complete decision-support system grounded in a real industrial problem.

🔗 Live dashboard: https://wafer-defect-detection-dashboard-jbxkmaiauhllreewbzctav.streamlit.app
💻 GitHub: https://github.com/Abdulsamedsay/wafer-defect-detection-dashboard
```

---

## Interview Answers

### "Tell me about this project."

> I built an end-to-end computer vision system for classifying wafer defect patterns in
> semiconductor manufacturing, using the WM-811K dataset — about 800,000 wafer maps with
> nine defect classes.
>
> The main technical challenges were extreme class imbalance — a 989-to-1 ratio — and the
> fact that accuracy is completely misleading in that setting. So I used Macro F1 as the
> primary metric and class-weighted loss during training.
>
> I trained a custom CNN in PyTorch and got Macro F1 of 0.686 on the test set, which is a
> 0.134 improvement over the logistic regression baseline. Edge-Ring and Center defects
> are detected reliably. Scratch is the hardest class — thin linear defects lose detail
> when you resize to 64x64, and there are very few training examples.
>
> On top of the model, I added Grad-CAM explainability and a yield-risk scoring layer that
> maps each prediction to a risk level with a recommended action. Everything runs in a
> Streamlit dashboard.

### "Why Macro F1 and not accuracy?"

> With a 989-to-1 class imbalance, a model that always predicts "no defect" achieves 85%
> accuracy. That model is useless — it never detects any defect. Macro F1 averages F1 across
> all nine classes equally, so a class with 22 test samples gets the same weight as a class
> with 22,000. That's the right lens for this problem.

### "Why add explainability?"

> In industrial contexts, a prediction alone is often not enough to act on. An engineer
> looking at a "High Risk — Edge-Ring" output needs to trust that the model is focusing on
> the right region of the wafer, not on some spurious pixel pattern. Grad-CAM provides
> that check. It also revealed a real insight: the Scratch class has diffuse, unfocused
> heatmaps — which explains why the model struggles with it before you even look at the F1.

### "Why is Scratch the hardest class?"

> Two reasons. First, at 64×64 resolution, a scratch is a thin diagonal or curved line
> that spans only a few pixels — a lot of that fine structure is lost when you downsample
> from the original wafer map size. Second, it's a rare class even after class weighting,
> so the model has limited examples to learn from. The Grad-CAM heatmaps confirm this:
> instead of focusing on a linear region, the model's attention is scattered. The fix would
> be higher input resolution and more aggressive augmentation for that class.

### "Why is this relevant to ASML?"

> ASML builds the lithography machines that create the exposure patterns on wafers. Defects
> like Edge-Ring and Center are often caused by issues in the exposure step — alignment,
> dose uniformity, or illumination. A tool that classifies these patterns quickly and flags
> them to engineers supports exactly the kind of process monitoring and yield improvement
> that matters in that environment. I'm not claiming this is a production system — it's a
> prototype that demonstrates the design thinking behind industrial AI decision-support tools.

---

## What NOT to say

- "I built a wafer defect classifier" — sounds like a homework assignment
- "It achieves 86% accuracy" — misleading; the baseline is 85% by predicting nothing
- "I deployed it to production" — not true; it's a portfolio prototype
- "It's ready to use at ASML" — overclaiming; say "prototype" and "proof of concept"
