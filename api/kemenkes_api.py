"""Kemenkes intelligence API adapter.

The API layer is intentionally thin: data access and scope semantics live in core;
analytics are produced by the SI-HIS intelligence layer rather than by the dashboard.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.data_provider import get_cases, get_source_descriptor
from core.scope import QueryScope, apply_scope, scope_label


def _risk_level(score: float) -> str:
    if score >= 60:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


def build_kemenkes_intelligence(
    period_days: int = 14,
    scope: dict[str, Any] | None = None,
    source_id: str = "dummy",
) -> dict[str, Any]:
    """Build Kemenkes intelligence for one isolated read/query scope."""
    raw = scope or {}
    query_scope = QueryScope(
        province=raw.get("province"),
        district=raw.get("district"),
        kecamatan=raw.get("kecamatan"),
        village=raw.get("village"),
        puskesmas=raw.get("puskesmas"),
        disease=raw.get("disease"),
        period_days=period_days,
    ).normalized()

    descriptor = get_source_descriptor(source_id)
    df = get_cases(source_id=source_id).copy(deep=True)
    df["Tanggal Sakit"] = pd.to_datetime(df["Tanggal Sakit"], errors="coerce")
    df = df.dropna(subset=["Tanggal Sakit"])
    df = apply_scope(df, query_scope)

    label = scope_label(query_scope)
    if df.empty:
        return _empty_response(query_scope, label)

    latest = df["Tanggal Sakit"].max()
    recent = df[df["Tanggal Sakit"] >= latest - pd.Timedelta(days=query_scope.period_days - 1)].copy()
    last7 = df[df["Tanggal Sakit"] >= latest - pd.Timedelta(days=6)].copy()
    prev7 = df[(df["Tanggal Sakit"] >= latest - pd.Timedelta(days=13)) & (df["Tanggal Sakit"] < latest - pd.Timedelta(days=6))].copy()

    disease = (
        recent.groupby("Diagnosis Konfirm", dropna=False).size()
        .sort_values(ascending=False).head(10).reset_index(name="cases")
        .rename(columns={"Diagnosis Konfirm": "disease"})
    )

    geo_column = "Provinsi"
    if query_scope.province:
        geo_column = "Kabupaten"
    if query_scope.district:
        geo_column = "Kecamatan"
    if query_scope.kecamatan:
        geo_column = "Desa/Kelurahan"
    if query_scope.village:
        geo_column = "Puskesmas"

    if geo_column in recent.columns:
        geo = recent.groupby(geo_column).size().reset_index(name="cases")
        max_cases = max(float(geo["cases"].max()), 1.0)
        geo["risk"] = (geo["cases"] / max_cases * 100).round(1)
        if {"Latitude", "Longitude"}.issubset(recent.columns):
            coords = recent.groupby(geo_column)[["Latitude", "Longitude"]].mean().reset_index()
            geo = geo.merge(coords, on=geo_column, how="left")
        else:
            geo["Latitude"] = 0.0
            geo["Longitude"] = 0.0
    else:
        geo = pd.DataFrame(columns=[geo_column, "cases", "risk", "Latitude", "Longitude"])

    province_risk = [
        {
            "province": r[geo_column],
            "risk": float(r["risk"]),
            "latitude": float(r.get("Latitude", 0.0)),
            "longitude": float(r.get("Longitude", 0.0)),
        }
        for r in geo.sort_values("risk", ascending=False).to_dict("records")
    ]

    trend = (len(last7) - len(prev7)) / len(prev7) if len(prev7) else (1.0 if len(last7) else 0.0)
    klb_signal = max(0.0, min(1.0, 0.50 + trend * 0.25))
    high_risk = int((geo["risk"] >= 60).sum()) if not geo.empty else 0
    spatial_risk = float(min(1.0, high_risk / max(len(geo), 1)))

    early_warning = []
    for _, row in geo.sort_values("risk", ascending=False).head(10).iterrows():
        level = _risk_level(float(row["risk"]))
        if level != "LOW":
            early_warning.append({"area": row[geo_column], "level": level, "reason": "Case burden signal"})

    daily = df.set_index("Tanggal Sakit").resample("D").size()
    forecast = []
    if len(daily) >= 7:
        baseline = float(daily.tail(7).mean())
        forecast = [
            {"date": (latest + pd.Timedelta(days=i)).strftime("%Y-%m-%d"), "cases": round(baseline, 1)}
            for i in range(1, 8)
        ]

    return {
        "area": label,
        "period": f"{query_scope.period_days} Hari",
        "total_cases": int(len(recent)),
        "cases_7d": int(len(last7)),
        "active_alerts": high_risk,
        "high_risk_areas": high_risk,
        "top_disease": disease.to_dict("records"),
        "early_warning": early_warning,
        "province_risk": province_risk,
        "ml": {"klb_signal": round(klb_signal, 3), "spatial_risk": round(spatial_risk, 3), "vulnerable_population": 0.0},
        "forecast": forecast,
        "vulnerable_population": {"group": "Agregat — modul vulnerable population SI-HIS", "signal": 0.0},
        "recommendations": [
            "Verifikasi sinyal wilayah prioritas melalui surveilans.",
            "Pantau tren 7 hari dan perubahan distribusi penyakit.",
            "Evaluasi kesiapan logistik pada wilayah dengan sinyal risiko meningkat.",
        ],
        "data_provenance": {
            "source": descriptor.label,
            "source_id": source_id,
            "dataset_type": "synthetic_national" if source_id == "dummy" else "factual",
            "engine": "SI-HIS Intelligence",
            "query_scope": query_scope.to_dict(),
            "scope_isolated": True,
        },
    }


def _empty_response(scope: QueryScope, label: str) -> dict[str, Any]:
    return {
        "area": label,
        "period": f"{scope.period_days} Hari",
        "total_cases": 0,
        "cases_7d": 0,
        "active_alerts": 0,
        "high_risk_areas": 0,
        "top_disease": [],
        "early_warning": [],
        "province_risk": [],
        "ml": {"klb_signal": 0.0, "spatial_risk": 0.0, "vulnerable_population": 0.0},
        "forecast": [],
        "vulnerable_population": {},
        "recommendations": [],
        "data_provenance": {
            "source": "SI-HIS canonical data provider",
            "source_id": "unknown",
            "dataset_type": "unknown",
            "engine": "SI-HIS Intelligence",
            "query_scope": scope.to_dict(),
            "scope_isolated": True,
        },
    }
