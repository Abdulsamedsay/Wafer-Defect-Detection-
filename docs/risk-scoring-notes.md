# Risk Scoring Notes

> Phase 7 — Yield-risk scoring implemented in `src/risk_scoring.py`.

---

## Purpose

The risk layer transforms a raw model prediction into a decision-support output.

Instead of just returning a class name, the system returns:
- A **risk level** (Low / Medium / High / Critical)
- A **recommended engineering action**
- A **confidence note** explaining whether the risk was elevated

This makes the AI output useful for an engineer reviewing wafer output — not just a label from a black box.

> This is a prototype decision-support feature, not a validated production quality-control system.

---

## Risk Mapping

| Defect Class | Base Risk | Rationale |
|---|---|---|
| none | Low | No detected defect. Wafer cleared for processing. |
| Random | Medium | Scattered defects. Yield impact uncertain — depends on die layout. |
| Loc | Medium | Localised cluster. Partial die loss in the affected region. |
| Donut | Medium | Ring around center. Moderate spatial extent. |
| Center | High | Center defect hits the highest die density zone. |
| Edge-Loc | High | Localised edge defect. Edge-die loss. |
| Edge-Ring | High | Full edge ring — indicates a systematic process problem. |
| Scratch | High | Linear defect crossing multiple dies. |
| Near-full | Critical | Almost entire wafer affected. Severe yield loss. |

---

## Confidence Adjustment

**Threshold: 70%**

| Condition | Effect |
|---|---|
| confidence >= 70% | Use base risk as-is |
| confidence < 70% | Elevate risk by one level |

Examples:
- `none` at 55% confidence -> **Medium** (elevated from Low)
- `Center` at 45% confidence -> **Critical** (elevated from High)
- `Near-full` at 31% confidence -> **Critical** (already maximum, no change)

**Why elevate on low confidence?**

If the model is uncertain, it may have misclassified a more severe defect as a less severe one. Elevating the risk conservatively ensures an uncertain prediction triggers review rather than automatic clearance.

---

## Risk Levels and Actions

| Risk Level | Score | Color | Recommended Action |
|---|---|---|---|
| Low | 1 | Green `#2ecc71` | No action required. Wafer cleared for continued processing. |
| Medium | 2 | Amber `#f39c12` | Flag for manual inspection. Secondary review recommended before proceeding. |
| High | 3 | Orange `#e67e22` | Hold wafer. Engineer review required before processing continues. |
| Critical | 4 | Red `#e74c3c` | Reject wafer. Defect coverage indicates severe yield loss. |

---

## Code

**Module:** `src/risk_scoring.py`

**Main function:**
```python
from src.risk_scoring import score

result = score(pred_class="Center", confidence=0.82)
print(result.risk_level)          # "High"
print(result.recommended_action)  # "Hold wafer. Engineer review required..."
print(result.color)               # "#e67e22"
print(result.elevated)            # False
```

**RiskResult fields:**
- `defect_class` — predicted class name
- `confidence` — model softmax probability
- `base_risk` — risk before confidence adjustment
- `risk_level` — final risk level after adjustment
- `risk_score` — integer 1-4 for sorting/coloring
- `recommended_action` — plain-English action string
- `confidence_note` — explanation of confidence effect
- `color` — hex color for dashboard display
- `elevated` — bool, True if risk was bumped up

**Batch scoring:**
```python
from src.risk_scoring import score_batch
results = score_batch(pred_classes, confidences)
```

---

## Portfolio Statement

> A yield-risk scoring layer was added on top of the CNN output. Each prediction is mapped to one of four risk levels (Low / Medium / High / Critical) based on the defect class severity and model confidence. Low-confidence predictions are conservatively escalated by one risk level to ensure uncertain cases trigger engineer review rather than automatic clearance. This transforms a classification model into a decision-support tool suitable for quality-control workflows.
