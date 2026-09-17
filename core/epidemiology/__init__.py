"""Modular epidemiology layer for SI-HIS.

The modules intentionally wrap the validated analytics implementation first;
algorithms can be replaced independently without changing the public engine API.
"""
from .trias import analyze_trias
from .mortality import analyze_mortality
from .risk import analyze_risk
from .time import build_epidemic_curve

__all__ = ["analyze_trias", "analyze_mortality", "analyze_risk", "build_epidemic_curve"]
