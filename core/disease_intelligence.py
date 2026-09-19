"""Disease Intelligence layer for SI-HIS.

Classifies diseases into non-communicable and communicable families and
provides epidemiology-aware temporal features. This layer does not replace
clinical/epidemiological validation; it supplies context to the ML engines.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiseaseIntelligenceProfile:
    name: str
    family: str
    subgroup: str
    transmission: str
    primary_ml_focus: str
    key_dimensions: tuple[str, ...]
    caveats: tuple[str, ...] = ()


DISEASE_INTELLIGENCE = {
    # Non-communicable
    "Diabetes Melitus": DiseaseIntelligenceProfile("Diabetes Melitus","PTM","metabolic","non-communicable","risk/progression","PERSON,TIME,LAB,BEHAVIOR"),
    "Hipertensi": DiseaseIntelligenceProfile("Hipertensi","PTM","cardiometabolic","non-communicable","risk/control/outcome","PERSON,TIME,LAB,BEHAVIOR"),
    "Penyakit Jantung": DiseaseIntelligenceProfile("Penyakit Jantung","PTM","cardiovascular","non-communicable","risk/outcome","PERSON,TIME,LAB"),
    "Stroke": DiseaseIntelligenceProfile("Stroke","PTM","cardiovascular","non-communicable","risk/outcome","PERSON,TIME,LAB"),
    "Kanker": DiseaseIntelligenceProfile("Kanker","PTM","neoplasm","non-communicable","risk/progression/survival","PERSON,TIME,LAB"),
    "Penyakit Ginjal Kronis": DiseaseIntelligenceProfile("Penyakit Ginjal Kronis","PTM","degenerative/chronic","non-communicable","progression/outcome","PERSON,TIME,LAB"),
    "Obesitas": DiseaseIntelligenceProfile("Obesitas","PTM","metabolic","non-communicable","risk/progression","PERSON,TIME,LAB,BEHAVIOR"),
    # Communicable
    "ISPA": DiseaseIntelligenceProfile("ISPA","MENULAR","direct","respiratory/direct","transmission/outbreak/forecast","TIME,PERSON,PLACE"),
    "COVID-like Respiratory Illness": DiseaseIntelligenceProfile("COVID-like Respiratory Illness","MENULAR","direct","respiratory/direct","transmission/outbreak/forecast","TIME,PERSON,PLACE"),
    "TBC": DiseaseIntelligenceProfile("TBC","MENULAR","direct","respiratory/direct","transmission/outbreak/contact","TIME,PERSON,PLACE"),
    "Campak": DiseaseIntelligenceProfile("Campak","MENULAR","direct","respiratory/direct","transmission/outbreak","TIME,PERSON,PLACE,VACCINATION"),
    "Diare Akut": DiseaseIntelligenceProfile("Diare Akut","MENULAR","direct/indirect","fecal-oral/environmental","exposure/outbreak/forecast","TIME,PERSON,PLACE,ENVIRONMENT"),
    "Demam Dengue": DiseaseIntelligenceProfile("Demam Dengue","MENULAR","vector-borne","vector","spatio-temporal/outbreak/forecast","TIME,PERSON,PLACE,ENVIRONMENT"),
    "DBD": DiseaseIntelligenceProfile("DBD","MENULAR","vector-borne","vector","spatio-temporal/outbreak/forecast","TIME,PERSON,PLACE,ENVIRONMENT"),
    "Malaria": DiseaseIntelligenceProfile("Malaria","MENULAR","vector-borne","vector","spatio-temporal/outbreak/forecast","TIME,PERSON,PLACE,ENVIRONMENT"),
    "Chikungunya": DiseaseIntelligenceProfile("Chikungunya","MENULAR","vector-borne","vector","spatio-temporal/outbreak/forecast","TIME,PERSON,PLACE,ENVIRONMENT"),
    "Leptospirosis": DiseaseIntelligenceProfile("Leptospirosis","MENULAR","zoonotic","zoonotic/environmental","one-health/spatio-temporal","TIME,PERSON,PLACE,ANIMAL,ENVIRONMENT"),
    "Rabies": DiseaseIntelligenceProfile("Rabies","MENULAR","zoonotic","zoonotic","one-health/exposure/outcome","TIME,PERSON,PLACE,ANIMAL"),
    "Avian Influenza": DiseaseIntelligenceProfile("Avian Influenza","MENULAR","zoonotic","zoonotic","one-health/outbreak","TIME,PERSON,PLACE,ANIMAL"),
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def classify_disease(disease: str | None) -> DiseaseIntelligenceProfile:
    if not disease or disease == "Semua Penyakit":
        return DiseaseIntelligenceProfile("Semua Penyakit","MIXED","mixed","mixed","descriptive only","TIME,PERSON,PLACE")
    if disease in DISEASE_INTELLIGENCE:
        return DISEASE_INTELLIGENCE[disease]
    n = _norm(disease)
    if any(k in n for k in ("diabetes","hipertensi","obesitas","kolesterol","metabol","jantung","stroke","kanker","neoplas","ginjal kron","osteoarthritis","demens")):
        return DiseaseIntelligenceProfile(disease,"PTM","other","non-communicable","risk/progression/outcome","PERSON,TIME,LAB,BEHAVIOR")
    if any(k in n for k in ("dengue","dbd","malaria","chikungunya","zika")):
        return DiseaseIntelligenceProfile(disease,"MENULAR","vector-borne","vector","spatio-temporal/outbreak/forecast","TIME,PERSON,PLACE,ENVIRONMENT")
    if any(k in n for k in ("leptos","rabies","anthrax","avian","flu burung","zoon")):
        return DiseaseIntelligenceProfile(disease,"MENULAR","zoonotic","zoonotic","one-health/outbreak","TIME,PERSON,PLACE,ANIMAL,ENVIRONMENT")
    if any(k in n for k in ("tbc","tuberk","campak","influenza","covid","ispa","pertusis","difteri","pertussis")):
        return DiseaseIntelligenceProfile(disease,"MENULAR","direct","direct","transmission/outbreak/forecast","TIME,PERSON,PLACE")
    return DiseaseIntelligenceProfile(disease,"UNKNOWN","unknown","unknown","generic/validated-outcome","TIME,PERSON,PLACE")


def disease_intelligence_summary(disease: str | None) -> dict[str, Any]:
    return asdict(classify_disease(disease))


def _date_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def analyze_onset_reporting_bias(df: pd.DataFrame) -> dict[str, Any]:
    """Compare onset time with examination/reporting time and identify temporal data-quality signals."""
    onset_col = _date_column(df, ["Tanggal Onset", "Onset Date", "onset_date", "Tanggal Sakit"])
    exam_col = _date_column(df, ["Tanggal Pemeriksaan", "Examination Date", "examination_date", "Tanggal Kunjungan"])
    report_col = _date_column(df, ["Tanggal Pelaporan", "Reporting Date", "reporting_date", "Tanggal Lapor"])
    if onset_col is None:
        return {"status":"unavailable","message":"Tanggal onset belum tersedia. Epicurve berbasis onset tidak dapat dihitung.","onset_column":None}
    onset = pd.to_datetime(df[onset_col], errors="coerce")
    result: dict[str, Any] = {"status":"ok","onset_column":onset_col,"examination_column":exam_col,"reporting_column":report_col}
    result["onset_completeness_pct"] = round(float(onset.notna().mean()*100),2) if len(df) else 0.0
    if exam_col:
        exam = pd.to_datetime(df[exam_col], errors="coerce")
        delay = (exam-onset).dt.total_seconds()/86400
        valid = delay.dropna()
        result["onset_to_examination"] = {
            "n": int(valid.size),
            "median_days": round(float(valid.median()),2) if len(valid) else None,
            "p90_days": round(float(valid.quantile(.90)),2) if len(valid) else None,
            "max_days": round(float(valid.max()),2) if len(valid) else None,
            "negative_delay_n": int((valid<0).sum()) if len(valid) else 0,
            "negative_delay_pct": round(float((valid<0).mean()*100),2) if len(valid) else 0.0,
        }
    if report_col:
        report = pd.to_datetime(df[report_col], errors="coerce")
        delay = (report-onset).dt.total_seconds()/86400
        valid = delay.dropna()
        result["onset_to_reporting"] = {
            "n": int(valid.size),
            "median_days": round(float(valid.median()),2) if len(valid) else None,
            "p90_days": round(float(valid.quantile(.90)),2) if len(valid) else None,
            "max_days": round(float(valid.max()),2) if len(valid) else None,
            "negative_delay_n": int((valid<0).sum()) if len(valid) else 0,
        }
    if len(onset.dropna()) >= 10:
        counts = onset.dt.normalize().value_counts()
        result["same_onset_spike"] = {
            "max_cases_same_onset_date": int(counts.max()),
            "top_onset_date": counts.idxmax().strftime("%Y-%m-%d"),
            "possible_batch_input": bool(counts.max() >= max(10, int(len(onset.dropna())*.20))),
        }
    warnings: list[str] = []
    if result["onset_completeness_pct"] < 80: warnings.append("Kelengkapan onset <80%; interpretasi temporal berbasis onset harus hati-hati.")
    if "onset_to_examination" in result and result["onset_to_examination"]["negative_delay_n"] > 0:
        warnings.append("Terdapat tanggal pemeriksaan sebelum onset; kemungkinan kesalahan input atau definisi waktu perlu diverifikasi.")
    if result.get("same_onset_spike",{}).get("possible_batch_input"):
        warnings.append("Terdapat konsentrasi onset pada satu tanggal; kemungkinan input batch/default perlu diverifikasi.")
    result["warnings"] = warnings
    return result


def extract_epicurve_features(df: pd.DataFrame) -> dict[str, Any]:
    onset_col = _date_column(df, ["Tanggal Onset", "Onset Date", "onset_date", "Tanggal Sakit"])
    if onset_col is None:
        return {"status":"unavailable","features":{},"message":"Tanggal onset tidak tersedia."}
    dates = pd.to_datetime(df[onset_col], errors="coerce").dropna()
    if dates.empty:
        return {"status":"unavailable","features":{},"message":"Tanggal onset tidak valid."}
    daily = dates.dt.normalize().value_counts().sort_index()
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(),freq="D"),fill_value=0)
    y = full.to_numpy(float)
    peak_i = int(np.argmax(y))
    baseline = float(np.median(y)) if len(y) else 0.0
    return {
        "status":"ok",
        "onset_column":onset_col,
        "daily_cases":pd.DataFrame({"Tanggal Onset":full.index,"Kasus":y.astype(int)}),
        "features":{
            "n_days":int(len(y)),
            "total_cases":int(y.sum()),
            "peak_cases":int(y.max()),
            "peak_date":full.index[peak_i],
            "peak_to_baseline_ratio":round(float(y.max()/max(baseline,1)),3),
            "area_under_curve":float(y.sum()),
            "days_to_peak":int(peak_i),
            "growth_early":round(float((y[min(len(y)-1,6)]/max(y[0],1))-1),3) if len(y)>=2 else None,
            "decline_last_7":round(float((y[-1]/max(y[max(0,len(y)-8)],1))-1),3) if len(y)>=2 else None,
        }
    }


def candidate_index_cases(df: pd.DataFrame, max_candidates: int = 10) -> dict[str, Any]:
    onset_col = _date_column(df, ["Tanggal Onset", "Onset Date", "onset_date", "Tanggal Sakit"])
    if onset_col is None:
        return {"status":"unavailable","message":"Tanggal onset tidak tersedia."}
    onset = pd.to_datetime(df[onset_col], errors="coerce")
    valid = onset.dropna()
    if valid.empty:
        return {"status":"unavailable","message":"Tidak ada onset valid."}
    first = valid.min().normalize()
    idx = valid[valid.dt.normalize().eq(first)].index
    candidates = df.loc[idx].copy().head(max_candidates)
    columns=[c for c in ["Nama","case_id","Case_ID",onset_col,"Tanggal Pemeriksaan","Tanggal Pelaporan","Provinsi","Kabupaten","Kecamatan","Desa/Kelurahan","Puskesmas"] if c in candidates.columns]
    table=candidates[columns].copy() if columns else candidates.copy()
    return {
        "status":"candidate_only",
        "first_onset_date":first,
        "candidate_count":int(len(idx)),
        "candidates":table,
        "interpretation":"Kandidat kasus dengan onset tercatat paling awal pada dataset. Ini bukan penetapan kasus indeks secara epidemiologis; investigasi lapangan, definisi kasus, paparan, kontak, dan kualitas pencatatan tetap diperlukan.",
    }


def build_infectious_intelligence(df: pd.DataFrame, disease: str | None) -> dict[str, Any]:
    profile=classify_disease(disease)
    if profile.family != "MENULAR":
        return {"enabled":False,"reason":"Epicurve/transmission intelligence diprioritaskan untuk penyakit menular.","profile":asdict(profile)}
    return {
        "enabled":True,
        "profile":asdict(profile),
        "onset_quality":analyze_onset_reporting_bias(df),
        "epicurve_features":extract_epicurve_features(df),
        "candidate_index":candidate_index_cases(df),
        "interpretation_guardrail":"Interpretasi berdasarkan onset yang tercatat; perbedaan onset, pemeriksaan, dan pelaporan dapat menghasilkan bias temporal. Model tidak menetapkan kasus indeks atau KLB secara otomatis.",
    }
