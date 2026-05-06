# Project Description — Wafer Defect Detection & Yield Risk Dashboard

## 1. Project Identity

**Project Name:** Wafer Defect Detection & Yield Risk Dashboard  
**Domain:** Computer Vision / Industrial AI / Semiconductor Manufacturing  
**Target Company Context:** ASML / high-tech semiconductor ecosystem  
**Project Type:** End-to-end AI portfolio project  
**Main Output:** Trained model + evaluation report + explainability + Streamlit dashboard + professional GitHub repository  
**Primary Goal:** Build a realistic AI decision-support tool for wafer defect classification and yield-risk estimation.

---

## 2. Project Summary

This project focuses on detecting defect patterns in semiconductor wafer maps using computer vision.

The goal is not only to train a classification model, but to build an end-to-end system that can:

- Load and preprocess wafer map data
- Classify wafer defect patterns
- Evaluate model performance with proper metrics
- Explain predictions visually
- Estimate yield-risk level
- Present results in an interactive dashboard

The project should be positioned as an **industrial AI quality-control tool**, not as a simple image classification exercise.

---

## 3. Business Context

In semiconductor manufacturing, wafers are tested to identify defective dies. The spatial distribution of defective dies can form patterns that indicate possible production or process issues.

Detecting these defect patterns can support:

- Manufacturing quality control
- Yield improvement
- Process monitoring
- Early warning for production issues
- Engineering decision-making

This project is inspired by the kind of high-precision manufacturing environment found in companies such as **ASML**, where process quality, defect reduction, optimization, and reliability are critical.

### Core Business Question

> How can computer vision help detect wafer defect patterns and support quality-control decisions in semiconductor manufacturing?

---

## 4. Why This Project Matters

This project is designed to show more than basic machine learning knowledge.

It should demonstrate:

- Understanding of an industrial problem
- Ability to work with real-world, imperfect data
- Computer vision modeling skills
- Model evaluation beyond accuracy
- Explainable AI thinking
- End-to-end system design
- Ability to present results clearly through a dashboard
- Portfolio-level communication suitable for GitHub, LinkedIn, and interviews

The final message of the project should be:

> I did not only build a model. I built a small AI decision-support system for semiconductor quality control.

---

## 5. Dataset

### Recommended Dataset

**WM-811K Wafer Map Dataset**

This dataset is commonly used for wafer defect classification tasks. It contains wafer maps and defect pattern labels.

Expected defect classes may include:

- Center
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Near-full
- Random
- Scratch
- None / Normal

The dataset may include real-world challenges such as:

- Class imbalance
- Missing labels
- Irregular wafer map shapes
- Different wafer sizes
- Noisy data

These challenges should not be ignored. They should be analyzed and handled professionally.

---

## 6. Main System Inputs and Outputs

### Input

The system receives a wafer map as input. This may come from:

- Dataset sample
- Preprocessed wafer map array
- Uploaded wafer map image or array in the dashboard

### Output

The system should produce:

- Predicted defect class
- Prediction confidence score
- Estimated yield-risk level
- Recommended engineering action
- Visual explanation of the model prediction

### Example Output

```text
Predicted Defect: Edge-Ring
Confidence: 91%
Estimated Risk Level: High
Recommended Action: Review process alignment and edge exposure conditions.
```

---

## 7. Technical Stack

### Programming Language

- Python

### Data Processing

- pandas
- NumPy
- scikit-learn

### Visualization

- matplotlib
- seaborn or plotly where useful

### Image Processing

- OpenCV
- PIL
- torchvision transforms

### Deep Learning Framework

Preferred:

- PyTorch

Alternative if necessary:

- TensorFlow / Keras

### Machine Learning Baseline Models

Possible baseline models:

- Logistic Regression
- Random Forest
- Support Vector Machine

### Deep Learning Models

Possible models:

- Custom CNN
- ResNet18
- EfficientNet-B0

Start with a simple CNN before moving to more advanced models.

### Explainability

Possible methods:

- Grad-CAM
- Saliency maps
- Misclassified sample analysis

### Dashboard

Preferred framework:

- Streamlit

---

## 8. Recommended Repository Structure

The repository should be clean and professional.

```text
wafer-defect-detection-dashboard/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_cnn_training.ipynb
│   ├── 05_evaluation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── risk_scoring.py
│   ├── utils.py
│
├── app/
│   ├── streamlit_app.py
│
├── models/
│   ├── best_model.pth
│
├── outputs/
│   ├── figures/
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   ├── gradcam_examples.png
│
├── README.md
├── requirements.txt
├── .gitignore
└── project_desc.md
```

---

## 9. Development Philosophy

This project should be built step by step.

Do not rush directly to model training.

The correct order is:

1. Understand the data
2. Clean and preprocess the data
3. Build a simple baseline
4. Build a CNN model
5. Evaluate the model properly
6. Add explainability
7. Add risk scoring
8. Build the dashboard
9. Polish the GitHub repository
10. Prepare portfolio communication

The project should avoid becoming a notebook-only project. Notebooks are allowed for exploration, but reusable logic should be moved into the `src/` folder.

---

## 10. Project Phases

## Phase 1 — Project Setup and Dataset Understanding

### Goal

Create the project structure and understand the dataset.

### Tasks

- Create the repository folder structure
- Set up a Python virtual environment
- Create `requirements.txt`
- Load the WM-811K dataset
- Inspect dataset columns and data types
- Understand how wafer maps are stored
- Check available defect labels
- Check missing labels
- Plot sample wafer maps
- Analyze class distribution
- Write the first version of the README

### Expected Deliverables

- Clean folder structure
- `requirements.txt`
- `notebooks/01_data_exploration.ipynb`
- Sample wafer map visualizations
- Class distribution chart
- Initial README draft

---

## Phase 2 — Data Preprocessing

### Goal

Prepare the wafer map data for machine learning and deep learning.

### Tasks

- Clean invalid or missing labels
- Convert wafer maps into image-like arrays
- Resize wafer maps to a fixed shape, for example 32x32 or 64x64
- Normalize pixel values
- Encode class labels
- Create train, validation, and test splits
- Analyze class imbalance
- Decide how to handle class imbalance
- Save processed data

### Possible Class Imbalance Strategies

- Class weights
- Weighted loss function
- Oversampling minority classes
- Data augmentation

### Expected Deliverables

- `notebooks/02_preprocessing.ipynb`
- `src/preprocessing.py`
- Processed dataset files
- Label mapping file
- Preprocessing explanation in README

---

## Phase 3 — Baseline Model

### Goal

Create a simple machine learning baseline before training a deep learning model.

### Why This Matters

A baseline model makes the project more professional. It shows that the CNN is compared against simpler models instead of being used without justification.

### Tasks

- Flatten wafer maps into vectors
- Train one or more baseline models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
- Evaluate with accuracy, precision, recall, and F1-score
- Create a confusion matrix
- Save baseline results

### Expected Deliverables

- `notebooks/03_baseline_model.ipynb`
- Baseline performance table
- Baseline confusion matrix
- Short comparison explanation

---

## Phase 4 — CNN Model Training

### Goal

Train a deep learning model for wafer defect classification.

### Tasks

- Create a PyTorch Dataset class
- Create DataLoaders
- Build a custom CNN model
- Train the model
- Validate the model
- Track loss and accuracy over epochs
- Save the best model
- Plot training curves

### Suggested CNN Architecture

- Conv2D
- ReLU
- MaxPooling
- Conv2D
- ReLU
- MaxPooling
- Flatten
- Fully connected layer
- Dropout
- Output layer

### Expected Deliverables

- `notebooks/04_cnn_training.ipynb`
- `src/dataset.py`
- `src/model.py`
- `src/train.py`
- `models/best_model.pth`
- Training and validation curves

---

## Phase 5 — Model Evaluation

### Goal

Evaluate the model properly and understand where it performs well or poorly.

### Tasks

- Test the model on unseen data
- Calculate:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Per-class F1-score
- Create a confusion matrix
- Analyze misclassified samples
- Compare CNN performance with baseline performance
- Identify which classes are difficult for the model

### Important Evaluation Principle

Do not rely only on accuracy because wafer defect classes may be imbalanced.

The README should include an explanation such as:

> Because wafer defect classes are imbalanced, the model was evaluated using per-class precision, recall, and F1-score instead of relying only on accuracy.

### Expected Deliverables

- `notebooks/05_evaluation.ipynb`
- `src/evaluate.py`
- Confusion matrix image
- Classification report
- Misclassified examples visualization
- Evaluation section in README

---

## Phase 6 — Explainability

### Goal

Make the model output more useful for engineering decision-making.

### Why This Matters

In industrial environments, a prediction alone is often not enough. Engineers need to understand why the model made a certain prediction.

### Tasks

- Implement Grad-CAM or a similar explainability method
- Show which parts of the wafer map influenced the model prediction
- Visualize original wafer map and explanation heatmap side by side
- Include examples of correct and incorrect predictions

### Expected Deliverables

- `src/explainability.py`
- Grad-CAM or explanation visualizations
- Explainability page in dashboard
- Explainability section in README

### Key Message

> I added explainability to make the model output more useful for engineering decision-making, not just for prediction.

---

## Phase 7 — Yield-Risk Scoring

### Goal

Add a business-relevant risk layer on top of the model prediction.

The model should not only predict a defect class. It should also estimate a yield-risk level based on:

- Predicted defect class
- Confidence score
- Defect severity

### Example Risk Mapping

```text
None / Normal  -> Low Risk
Random         -> Medium Risk
Loc            -> Medium Risk
Donut          -> Medium Risk
Center         -> High Risk
Edge-Loc       -> High Risk
Edge-Ring      -> High Risk
Scratch        -> High Risk
Near-full      -> Critical Risk
```

### Example Output

```text
Predicted Defect: Near-full
Confidence: 94%
Estimated Risk Level: Critical
Recommended Action: Immediate engineering review recommended.
```

### Tasks

- Create a risk scoring function
- Map predicted classes to risk levels
- Adjust risk by confidence score where useful
- Add recommended engineering actions for each defect type

### Expected Deliverables

- `src/risk_scoring.py`
- Risk scoring table
- Dashboard risk card
- Risk scoring explanation in README

---

## Phase 8 — Streamlit Dashboard

### Goal

Create an interactive dashboard that presents the project as a usable AI decision-support tool.

### Dashboard Page 1 — Project Overview

Should include:

- Project title
- Short semiconductor manufacturing context
- Why wafer defect detection matters
- Why this project is relevant to ASML
- Dataset summary
- Model summary

### Dashboard Page 2 — Prediction

Should include:

- Select a sample wafer map or upload one
- Show original wafer map
- Show predicted defect class
- Show confidence score
- Show estimated risk level
- Show recommended action

### Dashboard Page 3 — Model Performance

Should include:

- Accuracy
- Precision
- Recall
- F1-score
- Per-class performance table
- Confusion matrix

### Dashboard Page 4 — Explainability

Should include:

- Original wafer map
- Grad-CAM or explanation heatmap
- Predicted class
- True class
- Confidence score

### Dashboard Page 5 — Dataset Insights

Should include:

- Class distribution chart
- Sample wafer maps per class
- Dataset imbalance explanation

### Expected Deliverable

- `app/streamlit_app.py`

---

## 11. README Requirements

The README must be portfolio-ready and professional.

Recommended README structure:

```markdown
# Wafer Defect Detection & Yield Risk Dashboard

## Business Problem
Explain the semiconductor manufacturing context and why wafer defect detection matters.

## Project Goal
Explain what the system does.

## Dataset
Describe the WM-811K dataset and its challenges.

## Methodology
Explain data preprocessing, baseline model, CNN model, evaluation, explainability, risk scoring, and dashboard.

## Results
Include model metrics, confusion matrix, example predictions, and Grad-CAM examples.

## Dashboard
Show screenshots or GIFs of the Streamlit dashboard.

## What I Learned
Explain technical and business learnings.

## Future Improvements
Mention possible next steps.
```

### Possible Future Improvements

- Try more advanced CNN architectures
- Improve handling of class imbalance
- Add anomaly detection
- Add synthetic wafer defect generation
- Deploy the dashboard online
- Add MLflow experiment tracking
- Add FastAPI prediction endpoint

---

## 12. Coding Standards

The code should be clean, modular, and understandable.

### Expectations

- Use clear function names
- Add comments where useful
- Use type hints where appropriate
- Avoid duplicated code
- Keep notebooks clean
- Move reusable logic into `src/`
- Handle errors where useful
- Keep file paths configurable
- Make the project easy to run for another person

### Important Principle

Notebooks are for exploration. The final reusable code should live in the `src/` folder.

---

## 13. How an AI Agent Should Help With This Project

The AI agent should act as a senior AI engineer and project mentor.

The agent should help with:

- Planning the project step by step
- Creating the repository structure
- Writing clean Python code
- Explaining technical decisions
- Debugging issues
- Improving model performance
- Designing the dashboard
- Writing README sections
- Preparing LinkedIn and CV descriptions

When asked for code, the agent should:

1. Explain the purpose of the code
2. Provide clean code
3. Explain where to save the file
4. Explain how to run it
5. Explain the expected output
6. Mention possible errors or next steps

The agent should not rush directly to advanced modeling. It should guide the project in a logical order.

---

## 14. Success Criteria

The project is successful when it includes:

- Working dataset loading pipeline
- Clear exploratory data analysis
- Wafer map visualizations
- Clean preprocessing pipeline
- Train/validation/test split
- Baseline model
- CNN model
- Per-class evaluation metrics
- Confusion matrix
- Misclassified sample analysis
- Explainability method
- Yield-risk scoring logic
- Streamlit dashboard
- Clean GitHub repository
- Professional README
- Portfolio-ready project explanation

---

## 15. Portfolio Positioning

This project should be described as:

> An end-to-end computer vision project for semiconductor manufacturing quality control.

It should not be described as:

> A simple image classification project.

The project should communicate that the developer understands:

- AI modeling
- Industrial context
- Data challenges
- Evaluation metrics
- Explainability
- Business impact
- End-to-end system thinking

---

## 16. CV Description

Possible CV entry:

```text
Wafer Defect Detection & Yield Risk Dashboard
Built an end-to-end computer vision system for semiconductor wafer defect classification using PyTorch. Developed a CNN-based model, evaluated performance with per-class F1-scores, added explainability using Grad-CAM, and deployed a Streamlit dashboard showing defect class, confidence score, and estimated yield-risk level.
```

---

## 17. LinkedIn Project Description

Possible LinkedIn post:

```text
I started building an end-to-end AI project inspired by semiconductor manufacturing and ASML’s high-tech ecosystem.

The project focuses on wafer defect detection using computer vision.

The goal is not only to classify wafer defect patterns, but also to make the model output useful for engineering decision-making through confidence scores, explainability, and yield-risk estimation.

Key components:
- Wafer map preprocessing
- CNN-based defect classification
- Per-class model evaluation
- Grad-CAM explainability
- Yield-risk scoring
- Streamlit dashboard

With this project, I want to show how AI can support quality control and process understanding in high-tech manufacturing.
```

---

## 18. First Task for the AI Agent

The first task is not model training.

The first task is:

1. Create the project folder structure
2. Set up the Python environment
3. Create `requirements.txt`
4. Create the first notebook: `notebooks/01_data_exploration.ipynb`
5. Load and inspect the WM-811K dataset
6. Plot sample wafer maps from each defect class
7. Analyze class distribution
8. Write the first README draft

After this, continue step by step through preprocessing, baseline modeling, CNN training, evaluation, explainability, risk scoring, and dashboard development.

---

## 19. Most Important Project Rule

> Do not build only a model. Build a decision-support tool.

The final project should show not only prediction performance, but also how the prediction can help an engineer understand and act on a wafer quality issue.

