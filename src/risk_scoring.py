"""
src/risk_scoring.py

Yield-risk scoring layer for the Wafer Defect Detection dashboard.

Maps a (predicted_class, confidence) pair to:
  - risk_level       : "Low" | "Medium" | "High" | "Critical"
  - risk_score       : 1 | 2 | 3 | 4   (for sorting / coloring)
  - recommended_action: plain-English engineering recommendation
  - confidence_note  : whether risk was elevated due to low confidence
  - color            : hex color for dashboard display

Confidence adjustment rule:
  - confidence >= 0.70 : use base risk from defect class
  - confidence <  0.70 : elevate one level (Low→Medium, Medium→High, High→Critical)

This is a prototype decision-support feature, not a validated production
quality-control system. Risk mappings are illustrative.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data_loader import DEFECT_CLASSES


# ── Risk level constants ───────────────────────────────────────────────────────

LOW      = "Low"
MEDIUM   = "Medium"
HIGH     = "High"
CRITICAL = "Critical"

_RISK_ORDER = [LOW, MEDIUM, HIGH, CRITICAL]

_RISK_SCORE = {LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4}

_RISK_COLOR = {
    LOW:      "#2ecc71",   # green
    MEDIUM:   "#f39c12",   # amber
    HIGH:     "#e67e22",   # orange
    CRITICAL: "#e74c3c",   # red
}

# Confidence threshold below which base risk is elevated by one level
CONFIDENCE_THRESHOLD = 0.70


# ── Base risk mapping by defect class ─────────────────────────────────────────
#
# Rationale (see wiki/risk-scoring-notes.md for full documentation):
#   none       → Low     : no detected defect; wafer cleared
#   Random     → Medium  : scattered defects; yield impact uncertain
#   Loc        → Medium  : localised cluster; partial die loss
#   Donut      → Medium  : ring around center; moderate spatial extent
#   Center     → High    : center defect; high die density zone
#   Edge-Loc   → High    : localised edge defect; edge-die loss
#   Edge-Ring  → High    : full edge ring; systematic process issue
#   Scratch    → High    : linear defect across multiple dies
#   Near-full  → Critical: almost entire wafer affected; severe yield loss

BASE_RISK: dict[str, str] = {
    "none":      LOW,
    "Random":    MEDIUM,
    "Loc":       MEDIUM,
    "Donut":     MEDIUM,
    "Center":    HIGH,
    "Edge-Loc":  HIGH,
    "Edge-Ring": HIGH,
    "Scratch":   HIGH,
    "Near-full": CRITICAL,
}


# ── Recommended engineering actions ───────────────────────────────────────────

_ACTION: dict[str, str] = {
    LOW:      "No action required. Wafer cleared for continued processing.",
    MEDIUM:   "Flag for manual inspection. Secondary review recommended before proceeding.",
    HIGH:     "Hold wafer. Engineer review required before processing continues.",
    CRITICAL: "Reject wafer. Defect coverage indicates severe yield loss.",
}


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RiskResult:
    defect_class:       str
    confidence:         float
    base_risk:          str
    risk_level:         str
    risk_score:         int
    recommended_action: str
    confidence_note:    str
    color:              str
    elevated:           bool   # True if risk was bumped due to low confidence

    def as_dict(self) -> dict:
        return {
            "defect_class":        self.defect_class,
            "confidence":          self.confidence,
            "base_risk":           self.base_risk,
            "risk_level":          self.risk_level,
            "risk_score":          self.risk_score,
            "recommended_action":  self.recommended_action,
            "confidence_note":     self.confidence_note,
            "color":               self.color,
            "elevated":            self.elevated,
        }


# ── Core scoring function ─────────────────────────────────────────────────────

def score(
    pred_class: str | int,
    confidence: float,
) -> RiskResult:
    """
    Compute yield-risk score for one prediction.

    Parameters
    ----------
    pred_class : predicted defect class name (str) or index (int)
    confidence : model softmax probability for the predicted class, in [0, 1]

    Returns
    -------
    RiskResult dataclass
    """
    if isinstance(pred_class, int):
        pred_class = DEFECT_CLASSES[pred_class]

    if pred_class not in BASE_RISK:
        raise ValueError(f"Unknown class '{pred_class}'. Expected one of {list(BASE_RISK)}")

    base = BASE_RISK[pred_class]
    elevated = confidence < CONFIDENCE_THRESHOLD and base != CRITICAL

    if elevated:
        level_idx = min(_RISK_ORDER.index(base) + 1, len(_RISK_ORDER) - 1)
        risk_level = _RISK_ORDER[level_idx]
        confidence_note = (
            f"Low confidence ({confidence:.0%}) — risk elevated from {base} to {risk_level}."
        )
    else:
        risk_level = base
        confidence_note = f"Confident prediction ({confidence:.0%})."

    return RiskResult(
        defect_class       = pred_class,
        confidence         = confidence,
        base_risk          = base,
        risk_level         = risk_level,
        risk_score         = _RISK_SCORE[risk_level],
        recommended_action = _ACTION[risk_level],
        confidence_note    = confidence_note,
        color              = _RISK_COLOR[risk_level],
        elevated           = elevated,
    )


# ── Batch scoring ─────────────────────────────────────────────────────────────

def score_batch(
    pred_classes: list[str | int],
    confidences: list[float],
) -> list[RiskResult]:
    """Score a list of predictions. Returns one RiskResult per sample."""
    return [score(c, p) for c, p in zip(pred_classes, confidences)]


# ── Risk summary table ────────────────────────────────────────────────────────

def print_risk_table() -> None:
    """Print the full risk mapping table for documentation / inspection."""
    print(f"\n{'='*65}")
    print("  Yield-Risk Scoring Table")
    print(f"{'='*65}")
    print(f"  {'Class':<15} {'Base Risk':<12} {'Action'}")
    print(f"  {'-'*62}")
    for cls, risk in BASE_RISK.items():
        action_short = _ACTION[risk][:42] + "..."
        print(f"  {cls:<15} {risk:<12} {action_short}")
    print(f"\n  Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Below threshold: base risk elevated by one level.")
    print(f"{'='*65}\n")


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_risk_table()

    test_cases = [
        ("none",      0.89, "High conf 'none' -> Low"),
        ("none",      0.55, "Low conf 'none' -> Medium (elevated)"),
        ("Center",    0.82, "High conf Center -> High"),
        ("Center",    0.45, "Low conf Center -> Critical (elevated)"),
        ("Near-full", 0.31, "Near-full always Critical (cannot go higher)"),
        ("Scratch",   0.72, "Scratch confident -> High"),
        ("Random",    0.61, "Random low-conf -> High (elevated)"),
    ]

    print("Self-test:")
    for cls, conf, note in test_cases:
        r = score(cls, conf)
        elevated_marker = " [elevated]" if r.elevated else ""
        print(f"  {cls:<12} conf={conf:.0%}  ->  {r.risk_level:<9}{elevated_marker}")
        print(f"    {note}")
