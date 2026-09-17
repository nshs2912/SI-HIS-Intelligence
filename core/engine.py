"""Canonical SI-HIS intelligence engine.

Two analytical modes are intentionally separated:
1. descriptive: national/all-disease distribution; no disease/geography filter required.
2. epidemiology: disease-specific TIME + PERSON + PLACE intelligence; requires
   at least Kabupaten/Kota scope, a disease, and an adequate observation window.

The same engine can be called by Streamlit, FastAPI, scheduled workers, or a
future national server that enumerates disease x district x time windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .analytics import (
    deteksi_bentuk_kurva,
    deteksi_gelombang,
    hitung_effective_rt,
    identifikasi_vulnerable_profile,
)
from .epidemiology import analyze_mortality, analyze_risk, analyze_trias
from .epidemiology.pipeline import (
    classify_temporal_pattern_for_disease,
    resolve_disease_profile,
    validate_scope_for_special_analysis,
)
from .forecasting import holt_winters_forecast
from .ml_engine import (
    train_case_severity,
    train_klb_prediction,
    train_spatial_outbreak,
    train_vulnerable_population,
)
from .scope import QueryScope, apply_scope, scope_label
from .spatial import compute_epicenter, run_dbscan
from .statistics import hitung_bivariat_lengkap
from .surveillance import early_warning


@dataclass(frozen=True)
class IntelligenceResult:
    scope: QueryScope
    dataframe: pd.DataFrame
    provenance: dict[str, Any]


DISEASE_COLUMNS = ["Diagnosis Konfirm", "Diagnosis Probabel", "Diagnosis Suspek"]
NON_DISEASE_VALUES = {"", "nan", "none", "bukan", "tidak ada", "-"}


def _disease_per_case(df: pd.DataFrame) -> pd.Series:
    """Resolve one display disease per case, preferring confirmed diagnosis."""
    if df.empty:
        return pd.Series(index=df.index, dtype="object")
    result = pd.Series("Tidak Teridentifikasi", index=df.index, dtype="object")
    for column in reversed(DISEASE_COLUMNS):
        if column not in df.columns:
            continue
        values = df[column].astype(str).str.strip()
        valid = ~values.str.lower().isin(NON_DISEASE_VALUES)
        result.loc[valid] = values.loc[valid]
    return result


def _filter_disease(df: pd.DataFrame, disease: str | None) -> pd.DataFrame:
    if not disease or disease == "Semua Penyakit":
        return df.copy()
    work = df.copy()
    mask = pd.Series(False, index=work.index)
    for column in DISEASE_COLUMNS:
        if column in work.columns:
            mask |= work[column].astype(str).str.strip().eq(disease)
    return work.loc[mask].copy()


def _apply_period(df: pd.DataFrame, period_days: int) -> pd.DataFrame:
    """Use the most recent N calendar days for epidemiology requests."""
    if df.empty or "Tanggal Sakit" not in df.columns:
        return df.copy()
    work = df.copy()
    work["Tanggal Sakit"] = pd.to_datetime(work["Tanggal Sakit"], errors="coerce")
    dates = work["Tanggal Sakit"].dropna()
    if dates.empty:
        return work.iloc[0:0].copy()
    end = dates.max().normalize()
    start = end - pd.Timedelta(days=max(1, int(period_days)) - 1)
    return work.loc[work["Tanggal Sakit"].between(start, end)].copy()


def _top10_diseases(df: pd.DataFrame) -> pd.DataFrame:
    """National descriptive table requested by the SI-HIS design."""
    if df.empty:
        return pd.DataFrame(columns=["NO.", "Nama Penyakit", "Jumlah Kasus", "CFR", "Kabupaten", "Provinsi"])
    work = df.copy()
    work["_disease"] = _disease_per_case(work)
    death = pd.to_numeric(work.get("Is_Meninggal", 0), errors="coerce").fillna(0)
    if "Status Penderita" in work.columns:
        death = np.maximum(death, work["Status Penderita"].astype(str).str.lower().eq("meninggal").astype(int))
    work["_death"] = death
    rows: list[dict[str, Any]] = []
    for disease, group in work.groupby("_disease", dropna=False):
        if not disease or str(disease).lower() in NON_DISEASE_VALUES or disease == "Tidak Teridentifikasi":
            continue
        n = len(group)
        deaths = int(group["_death"].sum())
        district = group["Kabupaten"].mode().iloc[0] if "Kabupaten" in group and not group["Kabupaten"].mode().empty else "-"
        province = group["Provinsi"].mode().iloc[0] if "Provinsi" in group and not group["Provinsi"].mode().empty else "-"
        # For the national top-10 table, show the district contributing the most cases.
        if "Kabupaten" in group:
            district_counts = group["Kabupaten"].value_counts(dropna=True)
            if not district_counts.empty:
                district = district_counts.index[0]
                province_rows = group.loc[group["Kabupaten"].eq(district), "Provinsi"] if "Provinsi" in group else pd.Series(dtype=object)
                if not province_rows.empty:
                    province = province_rows.mode().iloc[0]
        rows.append({
            "Nama Penyakit": str(disease),
            "Jumlah Kasus": int(n),
            "CFR": round(deaths / n * 100, 2) if n else 0.0,
            "Kabupaten": district,
            "Provinsi": province,
        })
    out = pd.DataFrame(rows).sort_values(["Jumlah Kasus", "Nama Penyakit"], ascending=[False, True]).head(10).reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=["NO.", "Nama Penyakit", "Jumlah Kasus", "CFR", "Kabupaten", "Provinsi"])
    out.insert(0, "NO.", np.arange(1, len(out) + 1))
    return out


class IntelligenceEngine:
    """Stateless facade for SI-HIS analytical requests."""

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

    def descriptive(self, df: pd.DataFrame) -> dict[str, Any]:
        """National descriptive intelligence; deliberately independent of filters."""
        work = df.copy(deep=True)
        disease = _disease_per_case(work)
        work["_disease"] = disease
        total = len(work)
        sex = work["Jenis Kelamin"].value_counts(dropna=False).rename_axis("Jenis Kelamin").reset_index(name="Jumlah Kasus") if "Jenis Kelamin" in work else pd.DataFrame()
        age = pd.to_numeric(work["Umur"], errors="coerce") if "Umur" in work else pd.Series(dtype=float)
        age_bins = pd.cut(age, bins=[-1, 0, 4, 9, 14, 19, 24, 34, 44, 54, 64, 74, 84, np.inf], labels=["<1", "1-4", "5-9", "10-14", "15-19", "20-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "≥85"], include_lowest=True)
        age_table = age_bins.value_counts(sort=False, dropna=False).rename_axis("Kelompok Umur").reset_index(name="Jumlah Kasus")
        province = work.groupby("Provinsi", dropna=False).size().reset_index(name="Jumlah Kasus").sort_values("Jumlah Kasus", ascending=False) if "Provinsi" in work else pd.DataFrame()
        district = work.groupby(["Provinsi", "Kabupaten"], dropna=False).size().reset_index(name="Jumlah Kasus").sort_values("Jumlah Kasus", ascending=False) if "Kabupaten" in work else pd.DataFrame()
        deaths = pd.to_numeric(work.get("Is_Meninggal", 0), errors="coerce").fillna(0).sum()
        return {
            "mode": "descriptive",
            "overview": {"total_cases": int(total), "deaths": int(deaths), "cfr": round(float(deaths / total * 100), 2) if total else 0.0},
            "top10_diseases": _top10_diseases(work),
            "disease_distribution": work["_disease"].value_counts().rename_axis("Nama Penyakit").reset_index(name="Jumlah Kasus"),
            "province_distribution": province,
            "district_distribution": district,
            "sex_distribution": sex,
            "age_distribution": age_table,
            "provenance": {"engine": "SI-HIS Intelligence", "analysis": "national_descriptive", "scope_required": False},
        }

    def analyze(self, df: pd.DataFrame, scope: QueryScope | None = None, forecast_days: int = 14, include_ml: bool = False, mode: str = "epidemiology") -> dict[str, Any]:
        if mode == "descriptive":
            return self.descriptive(df)

        prepared = self.prepare(df, scope)
        base = _filter_disease(prepared.dataframe, prepared.scope.disease)
        work = _apply_period(base, prepared.scope.period_days)
        geographic_level = "Kabupaten" if prepared.scope.district else None
        if prepared.scope.puskesmas:
            geographic_level = "Puskesmas"
        elif prepared.scope.village:
            geographic_level = "Desa/Kelurahan"
        elif prepared.scope.kecamatan:
            geographic_level = "Kecamatan"
        eligibility = validate_scope_for_special_analysis(
            work,
            prepared.scope.disease,
            geographic_level,
            minimum_cases=10,
            minimum_days=14,
        )
        profile = resolve_disease_profile(prepared.scope.disease)
        common = {
            "mode": "epidemiology",
            "eligible": eligibility.eligible,
            "eligibility": {"reasons": eligibility.reasons, "warnings": eligibility.warnings},
            "scope": prepared.scope.to_dict(),
            "overview": {
                "total_cases": int(len(work)),
                "provinces": int(work["Provinsi"].nunique()) if "Provinsi" in work else 0,
                "districts": int(work["Kabupaten"].nunique()) if "Kabupaten" in work else 0,
                "subdistricts": int(work["Kecamatan"].nunique()) if "Kecamatan" in work else 0,
                "villages": int(work["Desa/Kelurahan"].nunique()) if "Desa/Kelurahan" in work else 0,
            },
            "disease_profile": profile.__dict__,
            "provenance": {**prepared.provenance, "analysis_mode": "disease_scoped_epidemiology", "ml_enabled": bool(include_ml)},
        }
        if not eligibility.eligible:
            common["message"] = "Analisis epidemiologi khusus belum dijalankan. Lengkapi penyakit + scope minimal Kabupaten/Kota + periode dan data yang memadai."
            common["analysis_sections"] = {k: None for k in ["person", "place", "time", "mortality", "risk_factors", "vulnerable", "ews", "rt", "waves", "forecast", "spatial", "epicenters", "temporal_interpretation", "ml"]}
            return common

        trias = analyze_trias(work)
        epi = trias["time"]
        waves = deteksi_gelombang(epi) if not epi.empty else []
        temporal = classify_temporal_pattern_for_disease(profile, len(waves))
        try:
            curve = deteksi_bentuk_kurva(epi, profile.name) if len(epi) >= 7 else None
        except Exception:
            curve = None
        common.update({
            "person": trias["person"],
            "place": trias["place"],
            "time": epi,
            "mortality": analyze_mortality(work),
            "risk": analyze_risk(work),
            "risk_factors": hitung_bivariat_lengkap(work) if len(work) >= 30 else {"status": "insufficient_sample", "message": "Minimal 30 kasus untuk modul statistik ini."},
            "vulnerable": identifikasi_vulnerable_profile(work),
            "ews": early_warning(epi),
            "rt": hitung_effective_rt(epi) if not epi.empty else None,
            "waves": waves,
            "forecast": holt_winters_forecast(epi, forecast_days),
            "spatial": run_dbscan(work),
            "epicenters": compute_epicenter(run_dbscan(work)),
            "temporal_interpretation": temporal,
            "epidemic_curve_classification": curve,
        })
        common["ml"] = self.ml_train(work) if include_ml else {"enabled": False, "message": "ML layer tidak dijalankan pada request ini."}
        return common

    def ml_train(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "case_severity": train_case_severity(df),
            "klb": train_klb_prediction(df),
            "spatial": train_spatial_outbreak(df),
            "vulnerable": train_vulnerable_population(df),
        }


SIHISIntelligenceEngine = IntelligenceEngine
