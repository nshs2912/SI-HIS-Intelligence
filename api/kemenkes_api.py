from __future__ import annotations

from typing import Any

import pandas as pd

from core.analytics import generate_data_simulasi


def _risk_level(score: float) -> str:
    if score >= 60:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


def _clean_scope(scope: dict[str, Any] | None) -> dict[str, str | None]:
    scope = scope or {}
    return {
        key: (str(value).strip() if value not in (None, "", "Semua", "Semua Provinsi") else None)
        for key, value in scope.items()
    }


def _filter_scope(df: pd.DataFrame, scope: dict[str, Any] | None) -> pd.DataFrame:
    """Create an isolated query dataframe; never mutate the canonical dataframe."""
    clean = _clean_scope(scope)
    work = df.copy()

    mapping = {
        "province": "Provinsi",
        "district": "Kabupaten",
        "kecamatan": "Kecamatan",
        "village": "Desa/Kelurahan",
        "puskesmas": "Puskesmas",
        "disease": "Diagnosis Konfirm",
    }
    for param, column in mapping.items():
        value = clean.get(param)
        if value and column in work.columns:
            work = work[work[column].astype(str).str.strip().eq(value)].copy()
    return work


def build_kemenkes_intelligence(
    period_days: int = 14,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Kemenkes intelligence from the canonical SI-HIS source.

    ``scope`` is a read/query context. Filtering is performed on a request-local
    copy and therefore cannot change the canonical dataset or another request.
    """
    scope = _clean_scope(scope)
    df = generate_data_simulasi().copy()
    df["Tanggal Sakit"] = pd.to_datetime(df["Tanggal Sakit"], errors="coerce")
    df = df.dropna(subset=["Tanggal Sakit"])
    df = _filter_scope(df, scope)

    scope_label = next(
        (scope[k] for k in ("puskesmas", "village", "kecamatan", "district", "province") if scope.get(k)),
        "Indonesia",
    )

    if df.empty:
        return {
            "area": scope_label,
            "period": f"{period_days} Hari",
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
                "source": "SI-HIS generate_data_simulasi",
                "dataset_type": "synthetic_demo",
                "engine": "SI-HIS Intelligence",
                "query_scope": scope,
                "scope_isolated": True,
            },
        }

    latest = df["Tanggal Sakit"].max()
    recent = df[df["Tanggal Sakit"] >= latest - pd.Timedelta(days=period_days - 1)].copy()
    last7 = df[df["Tanggal Sakit"] >= latest - pd.Timedelta(days=6)].copy()
    prev7 = df[(df["Tanggal Sakit"] >= latest - pd.Timedelta(days=13)) &
               (df["Tanggal Sakit"] < latest - pd.Timedelta(days=6))].copy()

    disease = (recent.groupby("Diagnosis Konfirm", dropna=False).size()
               .sort_values(ascending=False).head(10).reset_index(name="cases"))
    disease = disease.rename(columns={"Diagnosis Konfirm": "disease"})

    # At national scope, show province intelligence. At a narrowed scope,
    # the same component naturally becomes the next available geographic level.
    geo_column = "Provinsi"
    if scope.get("province"):
        geo_column = "Kabupaten"
    if scope.get("district"):
        geo_column = "Kecamatan"
    if scope.get("kecamatan"):
        geo_column = "Desa/Kelurahan"
    if scope.get("village"):
        geo_column = "Puskesmas"

    if geo_column in recent.columns:
        prov = recent.groupby(geo_column).size().reset_index(name="cases")
        max_cases = max(float(prov["cases"].max()), 1.0)
        prov["risk"] = (prov["cases"] / max_cases * 100).round(1)
        if {"Latitude", "Longitude"}.issubset(recent.columns):
            coords = recent.groupby(geo_column)[["Latitude", "Longitude"]].mean().reset_index()
            prov = prov.merge(coords, on=geo_column, how="left")
        else:
            prov["Latitude"] = 0.0
            prov["Longitude"] = 0.0
    else:
        prov = pd.DataFrame(columns=[geo_column, "cases", "risk", "Latitude", "Longitude"])

    province_risk = []
    for r in prov.sort_values("risk", ascending=False).to_dict("records"):
        province_risk.append({
            "province": r[geo_column],
            "risk": float(r["risk"]),
            "latitude": float(r.get("Latitude", 0.0)),
            "longitude": float(r.get("Longitude", 0.0)),
        })

    trend = (len(last7) - len(prev7)) / len(prev7) if len(prev7) else (1.0 if len(last7) else 0.0)
    klb_signal = max(0.0, min(1.0, 0.50 + trend * 0.25))
    high_risk = int((prov["risk"] >= 60).sum()) if not prov.empty else 0
    active_alerts = high_risk
    spatial_risk = float(min(1.0, high_risk / max(len(prov), 1)))

    ew = []
    for _, r in prov.sort_values("risk", ascending=False).head(10).iterrows():
        level = _risk_level(float(r["risk"]))
        if level != "LOW":
            ew.append({"area": r[geo_column], "level": level, "reason": "Case burden signal"})

    daily = df.set_index("Tanggal Sakit").resample("D").size()
    forecast = []
    if len(daily) >= 7:
        baseline = float(daily.tail(7).mean())
        for i in range(1, 8):
            forecast.append({"date": (latest + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                             "cases": round(baseline, 1)})

    return {
        "area": scope_label,
        "period": f"{period_days} Hari",
        "total_cases": int(len(recent)),
        "cases_7d": int(len(last7)),
        "active_alerts": active_alerts,
        "high_risk_areas": high_risk,
        "top_disease": disease.to_dict("records"),
        "early_warning": ew,
        "province_risk": province_risk,
        "ml": {"klb_signal": round(klb_signal, 3),
               "spatial_risk": round(spatial_risk, 3),
               "vulnerable_population": 0.0},
        "forecast": forecast,
        "vulnerable_population": {"group": "Agregat — perlu modul vulnerable population SI-HIS",
                                   "signal": 0.0},
        "recommendations": [
            "Verifikasi sinyal wilayah prioritas melalui surveilans.",
            "Pantau tren 7 hari dan perubahan distribusi penyakit.",
            "Evaluasi kesiapan logistik pada wilayah dengan sinyal risiko meningkat.",
        ],
        "data_provenance": {
            "source": "SI-HIS generate_data_simulasi",
            "dataset_type": "synthetic_demo",
            "engine": "SI-HIS Intelligence",
            "query_scope": scope,
            "scope_isolated": True,
        },
    }
