"""Canonical SI-HIS intelligence orchestration facade.

This module is the boundary between data/scope and analytical algorithms. It keeps
request filtering isolated and makes the same analytics reusable by Kemenkes,
Dinkes, Puskesmas, clinical and mobile consumers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .analytics import (
    hitung_risk_stratification,
    identifikasi_vulnerable_profile,
    hitung_early_warning_score,
    deteksi_bentuk_kurva,
    prediksi_kurva_holt_winters,
    hitung_effective_rt,
    deteksi_gelombang,
)
from .ml_engine import (
    train_case_severity,
    predict_case_severity,
    train_klb_prediction,
    predict_klb,
    train_spatial_outbreak,
    predict_spatial_outbreak,
    train_vulnerable_population,
    predict_vulnerable_population,
    robust_forecast,
)
from .scope import QueryScope, apply_scope, scope_label


@dataclass(frozen=True)
class IntelligenceResult:
    scope: QueryScope
    dataframe: pd.DataFrame
    provenance: dict[str, Any]


class IntelligenceEngine:
    """Stateless facade for one isolated intelligence request."""

    def prepare(self, df: pd.DataFrame, scope: QueryScope | None = None) -> IntelligenceResult:
        query_scope = (scope or QueryScope()).normalized()
        scoped = apply_scope(df, query_scope)
        return IntelligenceResult(
            scope=query_scope,
            dataframe=scoped,
            provenance={
                "engine": "SI-HIS Intelligence",
                "scope": query_scope.to_dict(),
                "scope_isolated": True,
                "source_rows": int(len(df)),
                "scoped_rows": int(len(scoped)),
                "area": scope_label(query_scope),
            },
        )

    def epidemiology(self, df: pd.DataFrame) -> dict[str, Any]:
        work = df.copy(deep=True)
        if work.empty:
            return {"cases": 0, "vulnerable_population": [], "risk": pd.DataFrame()}
        death = pd.to_numeric(work.get("Is_Meninggal", 0), errors="coerce").fillna(0)
        geo = [c for c in ["Provinsi", "Kabupaten", "Kecamatan", "Desa/Kelurahan"] if c in work.columns]
        if geo:
            risk = work.assign(_death=death).groupby(geo, dropna=False).agg(
                Total_Kasus=(geo[-1], "size"), Meninggal=("_death", "sum")
            ).reset_index()
            risk["CFR (%)"] = (risk["Meninggal"] / risk["Total_Kasus"].replace(0, pd.NA) * 100).fillna(0).round(2)
            risk = hitung_risk_stratification(risk)
        else:
            risk = pd.DataFrame()
        epi = pd.DataFrame()
        if "Tanggal Sakit" in work.columns:
            dates = pd.to_datetime(work["Tanggal Sakit"], errors="coerce").dropna()
            if not dates.empty:
                epi = dates.dt.date.value_counts().sort_index().rename_axis("Tanggal Sakit").reset_index(name="Jumlah Kasus")
                epi["Tanggal Sakit"] = pd.to_datetime(epi["Tanggal Sakit"])
        return {
            "cases": int(len(work)),
            "risk": risk,
            "vulnerable_population": identifikasi_vulnerable_profile(work),
            "epidemic_curve": epi,
            "ews": hitung_early_warning_score(epi) if not epi.empty else (None, None, None),
            "rt": hitung_effective_rt(epi) if not epi.empty else None,
            "waves": deteksi_gelombang(epi) if not epi.empty else [],
        }

    def ml_train(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return training contracts without mixing them into dashboard state."""
        return {
            "case_severity": train_case_severity(df),
            "klb": train_klb_prediction(df),
            "spatial": train_spatial_outbreak(df),
            "vulnerable": train_vulnerable_population(df),
        }
