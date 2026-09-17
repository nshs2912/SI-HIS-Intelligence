"""Temporal wave detection adapter."""
from ..analytics import deteksi_gelombang


def detect_waves(epidemic_curve):
    return deteksi_gelombang(epidemic_curve)
