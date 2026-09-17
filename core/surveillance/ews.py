"""Early warning signal adapter."""
from ..analytics import hitung_early_warning_score


def early_warning(epidemic_curve):
    score, trend_pct, level = hitung_early_warning_score(epidemic_curve)
    return {"score": score, "trend_7d_percent": trend_pct, "level": level, "calibrated_probability": False}
