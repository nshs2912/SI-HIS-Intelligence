"""DBSCAN spatial clustering using geographic coordinates."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def run_dbscan(df: pd.DataFrame, eps_km: float = 3.0, min_samples: int = 4) -> pd.DataFrame:
    work = df.copy(deep=True)
    if not {"Latitude", "Longitude"}.issubset(work.columns):
        work["Cluster"] = -1
        return work
    valid = work["Latitude"].notna() & work["Longitude"].notna()
    if valid.sum() < min_samples:
        work["Cluster"] = -1
        return work
    coords_rad = np.radians(work.loc[valid, ["Latitude", "Longitude"]].astype(float))
    model = DBSCAN(eps=eps_km / 6371.0088, min_samples=min_samples, metric="haversine").fit(coords_rad)
    work["Cluster"] = -1
    work.loc[valid, "Cluster"] = model.labels_
    return work
