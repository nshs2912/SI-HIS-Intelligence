"""Canonical SI-HIS intelligence orchestration facade.

One canonical engine serves Streamlit, FastAPI and future Kemenkes/Dinkes/
Puskesmas consumers. Analytical modules remain independently testable.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd

from .analytics import identifikasi_vulnerable_profile, hitung_effective_rt, deteksi_gelombang
from .ml_engine import (
    train_case_severity, train_klb_prediction, train_spatial_outbreak,
    train_vulnerable_population,
)
from .scope import QueryScope, apply_scope, scope_label
from .epidemiology import analyze_trias, analyze_mortality, analyze_risk
from .surveillance import early_warning
from .forecasting import holt_winters_forecast
from .spatial import run_dbscan, compute_epicenter
from .statistics import hitung_bivariat_lengkap, multivariable_logistic


@dataclass(frozen=True)
class IntelligenceResult:
    scope: QueryScope
    dataframe: pd.DataFrame
    provenance: dict[str, Any]


class IntelligenceEngine:
    """Stateless facade for an isolated SI-HIS intelligence request."""

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

    def analyze(self, df: pd.DataFrame, scope: QueryScope | None = None,
                forecast_days: int = 14, include_ml: bool = False) -> dict[str, Any]:
        """Produce the canonical SI-HIS analytical contract.

        ML is additive and explicitly optional; descriptive, diagnostic and
        surveillance analytics remain available without trained models.
        """
        prepared = self.prepare(df, scope)
        work = prepared.dataframe
        trias = analyze_trias(work)
        epi = trias["time"]
        mortality = analyze_mortality(work)
        risk = analyze_risk(work)
        ews = early_warning(epi)
        forecast = holt_winters_forecast(epi, forecast_days)
        spatial = run_dbscan(work)
        epicenter = compute_epicenter(spatial)
        stats_result = hitung_bivariat_lengkap(work) if len(work) >= 30 else {}

        result = {
            "overview": {
                "total_cases": int(len(work)),
                "provinces": int(work["Provinsi"].nunique()) if "Provinsi" in work else 0,
                "districts": int(work["Kabupaten"].nunique()) if "Kabupaten" in work else 0,
                "subdistricts": int(work["Kecamatan"].nunique()) if "Kecamatan" in work else 0,
                "villages": int(work["Desa/Kelurahan"].nunique()) if "Desa/Kelurahan" in work else 0,
            },
            "person": trias["person"],
            "place": trias["place"],
            "time": epi,
            "mortality": mortality,
            "risk": risk,
            "risk_factors": stats_result,
            "vulnerable": identifikasi_vulnerable_profile(work),
            "ews": ews,
            "rt": hitung_effective_rt(epi) if not epi.empty else None,
            "waves": deteksi_gelombang(epi) if not epi.empty else [],
            "forecast": forecast,
            "spatial": spatial,
            "epicenters": epicenter,
            "provenance": {**prepared.provenance, "ml_enabled": bool(include_ml)},
        }
        if include_ml:
            result["ml"] = self.ml_train(work)
        else:
            result["ml"] = {"enabled": False, "message": "ML layer tidak dijalankan pada request ini."}
        return result

    def ml_train(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "case_severity": train_case_severity(df),
            "klb": train_klb_prediction(df),
            "spatial": train_spatial_outbreak(df),
            "vulnerable": train_vulnerable_population(df),
        }


# Backward-compatible alias for application and API consumers.
SIHISIntelligenceEngine = IntelligenceEngine
