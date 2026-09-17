"""Canonical SI-HIS intelligence engine.

Two analytical modes:
1. descriptive: any geographic scope + all diseases; descriptive only.
2. epidemiology: one disease + any geographic scope (Indonesia/Province/District/
   drill-down) for TIME + PERSON + PLACE intelligence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from .analytics import deteksi_bentuk_kurva, deteksi_gelombang, hitung_effective_rt, identifikasi_vulnerable_profile
from .epidemiology import analyze_mortality, analyze_risk, analyze_trias
from .epidemiology.pipeline import classify_temporal_pattern_for_disease, resolve_disease_profile, validate_scope_for_special_analysis
from .forecasting import holt_winters_forecast
from .ml_engine import train_case_severity, train_klb_prediction, train_spatial_outbreak, train_vulnerable_population
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
    mask = pd.Series(False, index=df.index)
    for column in DISEASE_COLUMNS:
        if column in df.columns:
            mask |= df[column].astype(str).str.strip().eq(disease)
    return df.loc[mask].copy()

def _apply_period(df: pd.DataFrame, period_days: int) -> pd.DataFrame:
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

def _death_series(df: pd.DataFrame) -> pd.Series:
    death = pd.Series(0.0, index=df.index)
    if "Is_Meninggal" in df.columns:
        death = pd.to_numeric(df["Is_Meninggal"], errors="coerce").fillna(0).astype(float)
    if "Status Penderita" in df.columns:
        status = df["Status Penderita"].astype(str).str.strip().str.lower().eq("meninggal").astype(float)
        death = pd.Series(np.maximum(death, status), index=df.index)
    return death

def _top10_diseases(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["NO.", "Nama Penyakit", "Jumlah Kasus", "CFR", "Kabupaten", "Provinsi"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    work = df.copy()
    work["_disease"] = _disease_per_case(work)
    work["_death"] = _death_series(work)
    rows = []
    for disease, group in work.groupby("_disease", dropna=False):
        if not disease or str(disease).lower() in NON_DISEASE_VALUES or disease == "Tidak Teridentifikasi":
            continue
        n = len(group)
        deaths = int(group["_death"].sum())
        district = "-"
        province = "-"
        if "Kabupaten" in group:
            counts = group["Kabupaten"].value_counts(dropna=True)
            if not counts.empty:
                district = counts.index[0]
                if "Provinsi" in group:
                    p = group.loc[group["Kabupaten"].eq(district), "Provinsi"].mode()
                    if not p.empty:
                        province = p.iloc[0]
        rows.append({"Nama Penyakit": str(disease), "Jumlah Kasus": n, "CFR": round(deaths / n * 100, 2) if n else 0.0, "Kabupaten": district, "Provinsi": province})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = out.sort_values(["Jumlah Kasus", "Nama Penyakit"], ascending=[False, True]).head(10).reset_index(drop=True)
    out.insert(0, "NO.", np.arange(1, len(out) + 1))
    return out[columns]

class IntelligenceEngine:
    def prepare(self, df: pd.DataFrame, scope: QueryScope | None = None) -> IntelligenceResult:
        query_scope = (scope or QueryScope()).normalized()
        scoped = apply_scope(df, query_scope)
        return IntelligenceResult(query_scope, scoped, {"engine": "SI-HIS Intelligence", "scope": query_scope.to_dict(), "scope_isolated": True, "source_rows": len(df), "scoped_rows": len(scoped), "area": scope_label(query_scope)})

    def descriptive(self, df: pd.DataFrame) -> dict[str, Any]:
        work = df.copy(deep=True)
        work["_disease"] = _disease_per_case(work)
        total = len(work)
        deaths = int(_death_series(work).sum())
        sex = work["Jenis Kelamin"].value_counts(dropna=False).rename_axis("Jenis Kelamin").reset_index(name="Jumlah Kasus") if "Jenis Kelamin" in work else pd.DataFrame()
        age = pd.to_numeric(work["Umur"], errors="coerce") if "Umur" in work else pd.Series(dtype=float)
        bins = pd.cut(age, bins=[-1, 0, 4, 9, 14, 19, 24, 34, 44, 54, 64, 74, 84, np.inf], labels=["<1", "1-4", "5-9", "10-14", "15-19", "20-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "≥85"], include_lowest=True)
        age_table = bins.value_counts(sort=False, dropna=False).rename_axis("Kelompok Umur").reset_index(name="Jumlah Kasus")
        province = work.groupby("Provinsi", dropna=False).size().reset_index(name="Jumlah Kasus").sort_values("Jumlah Kasus", ascending=False) if "Provinsi" in work else pd.DataFrame()
        district = work.groupby(["Provinsi", "Kabupaten"], dropna=False).size().reset_index(name="Jumlah Kasus").sort_values("Jumlah Kasus", ascending=False) if "Kabupaten" in work else pd.DataFrame()
        return {"mode":"descriptive", "overview":{"total_cases":total,"deaths":deaths}, "top10_diseases":_top10_diseases(work), "disease_distribution":work["_disease"].value_counts().rename_axis("Nama Penyakit").reset_index(name="Jumlah Kasus"), "province_distribution":province, "district_distribution":district, "sex_distribution":sex, "age_distribution":age_table, "provenance":{"engine":"SI-HIS Intelligence","analysis":"descriptive","scope_required":False}}

    def analyze(self, df: pd.DataFrame, scope: QueryScope | None = None, forecast_days: int = 14, include_ml: bool = False, mode: str = "epidemiology") -> dict[str, Any]:
        if mode == "descriptive":
            return self.descriptive(df)
        prepared = self.prepare(df, scope)
        base = _filter_disease(prepared.dataframe, prepared.scope.disease)
        work = _apply_period(base, prepared.scope.period_days)
        if prepared.scope.puskesmas: geographic_level = "Puskesmas"
        elif prepared.scope.village: geographic_level = "Desa/Kelurahan"
        elif prepared.scope.kecamatan: geographic_level = "Kecamatan"
        elif prepared.scope.district: geographic_level = "Kabupaten"
        elif prepared.scope.province: geographic_level = "Provinsi"
        else: geographic_level = "Indonesia"
        eligibility = validate_scope_for_special_analysis(work, prepared.scope.disease, geographic_level, 10, 14)
        profile = resolve_disease_profile(prepared.scope.disease)
        common = {"mode":"epidemiology", "eligible":eligibility.eligible, "eligibility":{"reasons":eligibility.reasons,"warnings":eligibility.warnings}, "scope":prepared.scope.to_dict(), "overview":{"total_cases":len(work),"provinces":work["Provinsi"].nunique() if "Provinsi" in work else 0,"districts":work["Kabupaten"].nunique() if "Kabupaten" in work else 0,"subdistricts":work["Kecamatan"].nunique() if "Kecamatan" in work else 0,"villages":work["Desa/Kelurahan"].nunique() if "Desa/Kelurahan" in work else 0}, "disease_profile":profile.__dict__, "provenance":{**prepared.provenance,"analysis_mode":"disease_scoped_epidemiology","ml_enabled":bool(include_ml)}}
        if not eligibility.eligible:
            common["message"] = "Analisis epidemiologi khusus belum dijalankan karena scope/data belum memenuhi syarat."
            return common
        trias = analyze_trias(work)
        epi = trias["time"]
        waves = deteksi_gelombang(epi) if not epi.empty else []
        try: curve = deteksi_bentuk_kurva(epi, profile.name) if len(epi) >= 7 else None
        except Exception: curve = None
        spatial = run_dbscan(work)
        common.update({"person":trias["person"],"place":trias["place"],"time":epi,"mortality":analyze_mortality(work),"risk":analyze_risk(work),"risk_factors":hitung_bivariat_lengkap(work) if len(work)>=30 else {"status":"insufficient_sample","message":"Minimal 30 kasus untuk modul statistik ini."},"vulnerable":identifikasi_vulnerable_profile(work),"ews":early_warning(epi),"rt":hitung_effective_rt(epi) if not epi.empty else None,"waves":waves,"forecast":holt_winters_forecast(epi,forecast_days),"spatial":spatial,"epicenters":compute_epicenter(spatial),"temporal_interpretation":classify_temporal_pattern_for_disease(profile,len(waves)),"epidemic_curve_classification":curve,"ml":self.ml_train(work) if include_ml else {"enabled":False,"message":"ML layer tidak dijalankan."}})
        return common

    def ml_train(self, df: pd.DataFrame) -> dict[str, Any]:
        return {"case_severity":train_case_severity(df),"klb":train_klb_prediction(df),"spatial":train_spatial_outbreak(df),"vulnerable":train_vulnerable_population(df)}

SIHISIntelligenceEngine = IntelligenceEngine
