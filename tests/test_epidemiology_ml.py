"""Quick executable checks for the SI-HIS epidemiological ML extension."""
import numpy as np
import pandas as pd

from core.epidemiology_ml import (
    build_temporal_features,
    detect_temporal_anomalies,
    detect_change_points,
    build_spatial_neighbor_features,
    cluster_population_vulnerability,
    rank_areas_by_continuous_signal,
    train_growth_risk_model,
)

def sample_data(n_days=90):
    rng = np.random.default_rng(7)
    rows = []
    areas = [
        ("A",-7.75,110.36),
        ("B",-7.77,110.39),
        ("C",-6.90,107.61),
    ]
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    for area, lat, lon in areas:
        for i, dt in enumerate(dates):
            base = 2 + (i % 7 == 4) * 2
            if area == "A" and 70 <= i <= 77:
                base += 10
            for _ in range(int(rng.poisson(base))):
                rows.append({
                    "Tanggal Sakit": dt,
                    "Desa/Kelurahan": area,
                    "Latitude": lat + rng.normal(0, 0.001),
                    "Longitude": lon + rng.normal(0, 0.001),
                    "Umur": int(rng.integers(1,85)),
                    "Status Komorbid": "Ada Komorbid" if rng.random() < .25 else "Tidak Ada",
                    "Is_Meninggal": 1 if rng.random() < .01 else 0,
                })
    return pd.DataFrame(rows)

def test_ml_epi_smoke():
    df = sample_data()
    assert not build_temporal_features(
        df.groupby(["Desa/Kelurahan","Tanggal Sakit"]).size().reset_index(name="Cases")
    ).empty
    an = detect_temporal_anomalies(df)
    cp = detect_change_points(df)
    sp = build_spatial_neighbor_features(df)
    vp = cluster_population_vulnerability(df)
    rank = rank_areas_by_continuous_signal(df)
    growth = train_growth_risk_model(df)
    assert {"Anomaly_Flag","Anomaly_Score"}.issubset(an.columns)
    assert {"Change_Point_Flag","Change_Z"}.issubset(cp.columns)
    assert {"Neighbor_Cases","Neighbor_Areas"}.issubset(sp.columns)
    assert vp["status"] in {"ok","error"}
    assert rank.empty or "Continuous_Signal_Score" in rank.columns
    assert growth["status"] in {"ok","error"}
