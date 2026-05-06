# Project Overview

## Summary

This project builds an end-to-end AI decision-support tool for semiconductor wafer defect classification and yield-risk estimation.

The goal is not only to train a model, but to build a system that can classify wafer defect patterns, explain its predictions, estimate yield risk, and present results in an interactive dashboard.

---

## Business Context

In semiconductor manufacturing, wafers are tested after fabrication to identify defective dies. The spatial arrangement of these defective dies on the wafer surface often reveals patterns linked to specific production issues.

Identifying these patterns can support:

- Manufacturing quality control
- Yield improvement
- Process monitoring and root-cause analysis
- Early detection of production problems
- Engineering decision-making

This project is inspired by the precision-manufacturing environment at companies like **ASML**, a world leader in semiconductor lithography equipment.

---

## ASML Relevance

ASML develops lithography machines that are essential to the semiconductor manufacturing process. Understanding wafer defect patterns — and building tools that help engineers act on that understanding — directly aligns with ASML's focus on process quality, yield optimization, and precision manufacturing.

This project demonstrates the kind of thinking that is relevant to that context:

- Data-driven quality control
- Computer vision for pattern recognition
- Explainable AI for engineering decision support
- End-to-end system thinking

---

## Main System Input and Output

**Input:**
A wafer map — a 2D binary grid where each cell represents a die on the wafer. Defective dies are marked (typically as 2), passing dies as 1, and background as 0.

**Output:**

```text
Predicted Defect Class: Edge-Ring
Confidence: 91%
Estimated Risk Level: High
Recommended Action: Review process alignment and edge exposure conditions.
```

---

## Technical Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Data Processing | pandas, NumPy, scikit-learn |
| Visualization | matplotlib, seaborn, plotly |
| Explainability | Grad-CAM |
| Dashboard | Streamlit |

---

## Current Project Status

**Phase:** Phase 1 — Project Setup & Dataset Exploration

**Completed:**
- Project folder structure created
- Wiki initialized
- requirements.txt created
- .gitignore created
- README.md (initial draft) created
- First data exploration notebook created

**Next:**
- Load WM-811K dataset
- Inspect dataset structure and labels
- Visualize sample wafer maps
- Analyze class distribution
