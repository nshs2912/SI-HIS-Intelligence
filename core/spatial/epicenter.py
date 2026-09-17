"""Cluster-specific geographic centroid (epicenter proxy)."""
from __future__ import annotations
import pandas as pd


def compute_epicenter(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Latitude", "Longitude", "Cluster"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["Cluster", "Latitude", "Longitude", "Jumlah Kasus"])
    valid = df.dropna(subset=["Latitude", "Longitude"]).copy()
    valid = valid[valid["Cluster"] >= 0]
    if valid.empty:
        return pd.DataFrame(columns=["Cluster", "Latitude", "Longitude", "Jumlah Kasus"])
    return valid.groupby("Cluster").agg(
        Latitude=("Latitude", "mean"), Longitude=("Longitude", "mean"), Jumlah_Kasus=("Cluster", "size")
    ).reset_index()
