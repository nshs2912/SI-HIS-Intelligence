"""Geographic risk stratification."""
from __future__ import annotations
import pandas as pd
from ..analytics import hitung_risk_stratification
from .mortality import analyze_mortality


def analyze_risk(df: pd.DataFrame) -> pd.DataFrame:
    mortality = analyze_mortality(df)
    regional = mortality["regional"]
    if regional.empty:
        return regional
    return hitung_risk_stratification(regional)
