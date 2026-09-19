"""Acute poisoning and mass-casualty intelligence for SI-HIS.

This module detects patterns compatible with point-source poisoning or other
acute clustered events. It does not declare a legal KLB or disaster status.
It separates signal detection from attribution and requires human
investigation, case definition, exposure history and laboratory confirmation.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AcuteEventProfile:
    event_family: str
    expected_pattern: str
    onset_window: str
    key_dimensions: tuple[str, ...]
    interpretation: str


POISONING_PROFILES = {
    "chemical": AcuteEventProfile(
        "KERACUNAN_KIMIA", "point_source_or_common_exposure",
        "minutes_to_hours", ("TIME","PLACE","EXPOSURE","PERSON"),
        "Onset dapat sangat cepat setelah paparan; interval onset yang rapat meningkatkan kecurigaan common exposure tetapi tidak membuktikan etiologi."
    ),
    "bacterial_toxin": AcuteEventProfile(
        "KERACUNAN_TOKSIN_BAKTERI", "point_source_or_foodborne",
        "hours_to_days", ("TIME","PLACE","FOOD","EXPOSURE","PERSON"),
        "Onset dapat berkelompok setelah konsumsi/paparan bersama; rentang onset perlu dibandingkan dengan distribusi masa inkubasi yang relevan."
    ),
    "unknown_toxic": AcuteEventProfile(
        "KERACUNAN_AKUT_UNSPECIFIED", "point_source_or_unknown_exposure",
        "minutes_to_days", ("TIME","PLACE","EXPOSURE","PERSON"),
        "Pola klaster akut dapat merupakan common exposure, tetapi sumber dan etiologi tidak dapat ditentukan dari kurva saja."
    ),
}


def _dates(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_datetime(df[c], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def classify_acute_event_type(df: pd.DataFrame, event_type: str | None = None) -> dict[str, Any]:
    key = str(event_type or "unknown_toxic").strip().lower()
    if key not in POISONING_PROFILES:
        key = "unknown_toxic"
    p = POISONING_PROFILES[key]
    return asdict(p)


def detect_point_source_cluster(
    df: pd.DataFrame,
    time_window_minutes: int = 60,
    min_cases: int = 5,
    geographic_columns: tuple[str, ...] = ("Provinsi","Kabupaten","Kecamatan","Desa/Kelurahan","Puskesmas"),
) -> dict[str, Any]:
    """Detect unusually dense onset clusters without assigning etiology."""
    onset = _dates(df, ["Tanggal Onset","Onset Date","onset_date","Tanggal Sakit"])
    valid = onset.notna()
    if valid.sum() < min_cases:
        return {"status":"insufficient_data","point_source_signal":False,"message":f"Kasus onset valid < {min_cases}."}

    work=df.loc[valid].copy()
    work["_onset"]=onset.loc[valid]
    work=work.sort_values("_onset")
    best=None
    window=pd.Timedelta(minutes=time_window_minutes)

    for i,row in work.iterrows():
        end=row["_onset"]+window
        mask=(work["_onset"]>=row["_onset"])&(work["_onset"]<=end)
        n=int(mask.sum())
        if best is None or n>best["cases_in_window"]:
            best={"start":row["_onset"],"end":end,"cases_in_window":n,"indices":work.index[mask].tolist()}

    if best is None:
        return {"status":"no_signal","point_source_signal":False}

    cluster=work.loc[best["indices"]].copy()
    geo_summary={}
    for col in geographic_columns:
        if col in cluster.columns:
            geo_summary[col]=cluster[col].astype(str).value_counts().head(5).to_dict()

    concentration=float(best["cases_in_window"]/max(len(work),1))
    signal=best["cases_in_window"]>=min_cases
    return {
        "status":"signal" if signal else "no_signal",
        "point_source_signal":signal,
        "time_window_minutes":time_window_minutes,
        "cases_in_window":best["cases_in_window"],
        "cluster_start":best["start"],
        "cluster_end":best["end"],
        "concentration_pct":round(concentration*100,2),
        "geography":geo_summary,
        "interpretation":"Klaster onset yang sangat rapat secara waktu konsisten dengan pola point-source/common exposure dan perlu investigasi segera; bukan bukti etiologi atau penetapan KLB.",
    }


def detect_mass_casualty_burst(
    df: pd.DataFrame,
    baseline_days: int = 28,
    burst_window_minutes: int = 60,
    min_cases: int = 10,
) -> dict[str, Any]:
    """Compare an acute time burst with the historical daily baseline."""
    onset=_dates(df,["Tanggal Onset","Onset Date","onset_date","Tanggal Sakit"])
    onset=onset.dropna().sort_values()
    if len(onset)<min_cases:
        return {"status":"insufficient_data","signal":False}

    daily=onset.dt.floor("D").value_counts().sort_index()
    recent_date=daily.index.max()
    recent_window=onset[(onset>=recent_date)&(onset<recent_date+pd.Timedelta(days=1))]
    # Use a conservative historical daily baseline excluding the latest date.
    hist=daily[daily.index<recent_date].tail(baseline_days)
    baseline=float(hist.mean()) if len(hist) else 0.0
    burst=int(len(recent_window))
    ratio=burst/max(baseline,1.0)
    signal=burst>=min_cases and ratio>=3.0
    return {
        "status":"signal" if signal else "no_signal",
        "signal":signal,
        "latest_date":recent_date,
        "latest_day_cases":burst,
        "baseline_daily_mean":round(baseline,2),
        "burst_to_baseline_ratio":round(ratio,2),
        "interpretation":"Lonjakan mendadak dibanding baseline dapat menandakan acute event, mass gathering, outbreak, poisoning/common exposure, atau bencana; klasifikasi penyebab memerlukan data paparan dan investigasi.",
    }


def analyze_exposure_window(df: pd.DataFrame) -> dict[str, Any]:
    onset=_dates(df,["Tanggal Onset","Onset Date","onset_date","Tanggal Sakit"])
    exposure=_dates(df,["Tanggal Paparan","Exposure Date","exposure_datetime"])
    valid=onset.notna()&exposure.notna()
    if valid.sum()==0:
        return {"status":"unavailable","message":"Tanggal paparan belum tersedia."}
    delay=(onset[valid]-exposure[valid]).dt.total_seconds()/3600
    return {
        "status":"ok",
        "n":int(len(delay)),
        "median_hours":round(float(delay.median()),2),
        "p10_hours":round(float(delay.quantile(.10)),2),
        "p90_hours":round(float(delay.quantile(.90)),2),
        "negative_delay_n":int((delay<0).sum()),
        "interpretation":"Distribusi onset-paparan membantu membedakan hipotesis common exposure akut dari pola lain, tetapi harus dibandingkan dengan karakteristik agen dan kualitas waktu paparan.",
    }


def poisoning_vs_disaster_signal(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a neutral differential signal, never a final classification."""
    cluster=detect_point_source_cluster(df)
    burst=detect_mass_casualty_burst(df)
    exposure=analyze_exposure_window(df)
    flags=[]
    if cluster.get("point_source_signal"): flags.append("POINT_SOURCE_COMMON_EXPOSURE")
    if burst.get("signal"): flags.append("MASS_CASUALTY_BURST")
    if exposure.get("status")=="ok" and exposure.get("median_hours") is not None and exposure.get("median_hours")<=6: flags.append("RAPID_ONSET_EXPOSURE_PATTERN")

    if not flags:
        differential=["Tidak ada sinyal kuat acute clustered event dari waktu yang tersedia."]
    else:
        differential=[
            "Keracunan/common exposure",
            "Foodborne/toxin-associated event",
            "Bencana non-infeksi/chemical/environmental exposure",
            "Mass gathering atau kejadian dengan paparan bersama",
            "Outbreak infeksi akut dengan onset yang berdekatan",
        ]
    priority = "URGENT_FIELD_VERIFICATION" if flags else "ROUTINE_SURVEILLANCE"
    return {
        "status":"signal" if flags else "no_signal",
        "priority":priority,
        "signals":flags,
        "differential_hypotheses":differential,
        "point_source":cluster,
        "mass_casualty_burst":burst,
        "exposure_window":exposure,
        "guardrail":"Signal detector tidak membedakan secara definitif keracunan, bencana, atau outbreak. Perlu line list, waktu dan lokasi paparan, gejala, attack rate, pemeriksaan klinis, laboratorium/toksikologi, serta investigasi lapangan.",
    }
