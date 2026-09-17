"""Mortality and case-fatality analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd


def analyze_mortality(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_cases": 0, "deaths": 0, "cfr_percent": 0.0, "regional": pd.DataFrame()}
    work = df.copy(deep=True)
    if "Is_Meninggal" in work.columns:
        death = pd.to_numeric(work["Is_Meninggal"], errors="coerce").fillna(0).astype(int)
    else:
        death = work.get("Status Penderita", pd.Series(index=work.index, dtype=str)).astype(str).str.lower().eq("meninggal").astype(int)
    total = len(work)
    deaths = int(death.sum())
    cfr = deaths / total * 100 if total else 0.0
    geo = [c for c in ["Provinsi", "Kabupaten", "Kecamatan", "Desa/Kelurahan"] if c in work.columns]
    regional = pd.DataFrame()
    if geo:
        regional = work.assign(_death=death).groupby(geo, dropna=False).agg(Total_Kasus=(geo[-1], "size"), Meninggal=("_death", "sum")).reset_index()
        regional["CFR (%)"] = np.where(regional["Total_Kasus"] > 0, regional["Meninggal"] / regional["Total_Kasus"] * 100, 0).round(2)
    return {"total_cases": int(total), "deaths": deaths, "cfr_percent": round(cfr, 2), "regional": regional}
