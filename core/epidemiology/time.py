"""Temporal epidemiology helpers."""
from __future__ import annotations
import pandas as pd


def build_epidemic_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Tanggal Sakit" not in df.columns:
        return pd.DataFrame(columns=["Tanggal Sakit", "Jumlah Kasus"])
    dates = pd.to_datetime(df["Tanggal Sakit"], errors="coerce").dropna()
    if dates.empty:
        return pd.DataFrame(columns=["Tanggal Sakit", "Jumlah Kasus"])
    out = dates.dt.date.value_counts().sort_index().rename_axis("Tanggal Sakit").reset_index(name="Jumlah Kasus")
    out["Tanggal Sakit"] = pd.to_datetime(out["Tanggal Sakit"])
    return out
