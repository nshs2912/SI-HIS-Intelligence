"""Person-Place-Time (Trias Epidemiology) analysis."""
from __future__ import annotations
import pandas as pd


def analyze_trias(df: pd.DataFrame) -> dict:
    work = df.copy(deep=True)
    if work.empty:
        return {"person": {}, "place": pd.DataFrame(), "time": pd.DataFrame()}

    person = {}
    for col in ["Jenis Kelamin", "Pekerjaan", "Status Komorbid", "Status Imunisasi", "Riwayat Perjalanan"]:
        if col in work.columns:
            person[col] = work[col].astype(str).value_counts().rename_axis(col).reset_index(name="Jumlah Kasus")

    place_cols = [c for c in ["Provinsi", "Kabupaten", "Kecamatan", "Desa/Kelurahan"] if c in work.columns]
    place = work.groupby(place_cols, dropna=False).size().reset_index(name="Jumlah Kasus") if place_cols else pd.DataFrame()
    if not place.empty:
        place = place.sort_values("Jumlah Kasus", ascending=False)

    if "Tanggal Sakit" in work.columns:
        dates = pd.to_datetime(work["Tanggal Sakit"], errors="coerce").dropna()
        time = dates.dt.date.value_counts().sort_index().rename_axis("Tanggal Sakit").reset_index(name="Jumlah Kasus")
        time["Tanggal Sakit"] = pd.to_datetime(time["Tanggal Sakit"])
    else:
        time = pd.DataFrame(columns=["Tanggal Sakit", "Jumlah Kasus"])
    return {"person": person, "place": place, "time": time}
