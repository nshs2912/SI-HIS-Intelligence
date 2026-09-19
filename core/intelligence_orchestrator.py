"""SI-HIS Intelligence Orchestrator.

Fuses existing epidemiology/ML signals into an evidence-aware operational
intelligence layer. It does not replace clinical or public-health authority.
"""
from __future__ import annotations
from typing import Any
import math
import numpy as np
import pandas as pd


GEO_HIERARCHY = ("Puskesmas", "Desa/Kelurahan", "Kecamatan", "Kabupaten", "Provinsi")


def _onset(df: pd.DataFrame) -> pd.Series:
    for c in ("Tanggal Onset", "Tanggal Sakit", "Tanggal Pemeriksaan"):
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(pd.NaT, index=df.index)


def _unit_column(df: pd.DataFrame) -> str | None:
    for c in GEO_HIERARCHY:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def _robust_baseline(values: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 3:
        return float(x.mean()) if len(x) else 0.0, 0.0
    med = float(x.median())
    mad = float((x - med).abs().median())
    return med, mad


def multi_window_scan(df: pd.DataFrame, windows_minutes=(15, 30, 60, 180, 360, 720, 1440)) -> dict[str, Any]:
    onset = _onset(df).dropna().sort_values()
    if onset.empty:
        return {"status": "insufficient_data", "windows": []}
    rows = []
    onset_values = onset.astype("int64").to_numpy()
    for minutes in windows_minutes:
        delta_ns = int(pd.Timedelta(minutes=int(minutes)).value)
        right = np.searchsorted(onset_values, onset_values + delta_ns, side="right")
        counts = right - np.arange(len(onset_values))
        best_idx = int(np.argmax(counts)) if len(counts) else 0
        best_n = int(counts[best_idx]) if len(counts) else 0
        best_start = onset.iloc[best_idx] if len(onset) else None
        delta = pd.Timedelta(minutes=int(minutes))
        rows.append({
            "window_minutes": int(minutes),
            "cases": best_n,
            "start": best_start,
            "end": best_start + delta if best_start is not None else None,
            "density_per_hour": round(best_n / max(minutes / 60.0, 1 / 60.0), 2),
        })
    best = max(rows, key=lambda x: x["density_per_hour"]) if rows else None
    return {"status": "ok", "windows": rows, "best_window": best}


def finest_unit_intelligence(df: pd.DataFrame) -> dict[str, Any]:
    """Identify the smallest operational geographic unit with reliable signal."""
    unit = _unit_column(df)
    if unit is None or df.empty:
        return {"status": "insufficient_data", "message": "Unit geografis operasional belum tersedia."}
    onset = _onset(df)
    work = df.copy()
    work["_onset"] = onset
    work = work.dropna(subset=["_onset"])
    if work.empty:
        return {"status": "insufficient_data", "message": "Waktu onset/pelaporan valid belum tersedia."}
    groups = work.groupby(unit, dropna=False)
    rows = []
    for name, g in groups:
        n = len(g)
        span_h = max((g["_onset"].max() - g["_onset"].min()).total_seconds() / 3600.0, 1 / 60.0)
        rows.append({
            "Unit": str(name),
            "Level": unit,
            "Kasus": n,
            "Rentang_Onset_Jam": round(span_h, 2),
            "Kepadatan_per_Jam": round(n / span_h, 2),
        })
    table = pd.DataFrame(rows).sort_values(["Kasus", "Kepadatan_per_Jam"], ascending=False)
    top = table.iloc[0].to_dict() if not table.empty else {}
    return {
        "status": "ok",
        "finest_available_level": unit,
        "units_evaluated": int(len(table)),
        "highest_signal_unit": top,
        "table": table.reset_index(drop=True),
        "interpretation": "Unit terkecil digunakan sebagai lokasi prioritas operasional. Konsentrasi pada unit kecil bukan bukti sumber penularan atau etiologi.",
    }


def unknown_event_detection(df: pd.DataFrame, disease: str | None = None) -> dict[str, Any]:
    """Detect coherent clusters even when the recorded diagnosis is heterogeneous."""
    onset = _onset(df)
    valid = onset.notna()
    if valid.sum() < 5:
        return {"status": "insufficient_data", "signal": False}
    work = df.loc[valid].copy()
    work["_onset"] = onset.loc[valid]
    temporal = multi_window_scan(work)
    best = temporal.get("best_window") or {}
    diagnoses = []
    for c in ("Diagnosis Konfirm", "Diagnosis Probabel", "Diagnosis Suspek"):
        if c in work.columns:
            diagnoses.extend(work[c].dropna().astype(str).str.strip().tolist())
    unique_dx = len(set(x for x in diagnoses if x and x.lower() not in {"nan", "none", "-"}))
    concentrated = int(best.get("cases", 0)) >= max(5, int(math.ceil(len(work) * 0.10)))
    heterogeneous = unique_dx >= 3
    signal = bool(concentrated and heterogeneous)
    return {
        "status": "signal" if signal else "no_signal",
        "signal": signal,
        "cases_evaluated": int(len(work)),
        "best_temporal_window": best,
        "diagnostic_categories_observed": unique_dx,
        "hypotheses": [
            "Acute infectious cluster",
            "Keracunan/common exposure",
            "Paparan lingkungan/kimia",
            "Mass gathering",
            "Bencana/incident non-infeksi",
            "Unknown syndrome/event",
        ] if signal else [],
        "interpretation": "Unknown-event detection menandai pola yang belum cukup dijelaskan oleh satu label diagnosis; diperlukan case definition, line list, gejala, paparan, dan verifikasi lapangan.",
        "disease_context": disease,
    }


def uncertainty_intelligence(df: pd.DataFrame, signals: dict[str, Any]) -> dict[str, Any]:
    n = len(df)
    if n == 0:
        return {"confidence": "LOW", "score": 0, "data_completeness_pct": 0.0}
    required = ["Tanggal Sakit"]
    optional = ["Tanggal Onset", "Provinsi", "Kabupaten", "Kecamatan", "Desa/Kelurahan", "Puskesmas"]
    completeness_parts = []
    for c in required + optional:
        if c in df.columns:
            completeness_parts.append(float(df[c].notna().mean()))
    completeness = float(np.mean(completeness_parts) * 100) if completeness_parts else 0.0
    evidence = int(len(signals.get("active_signals", [])))
    score = min(100.0, 0.55 * completeness + min(evidence, 5) / 5 * 45.0)
    confidence = "HIGH" if score >= 80 else ("MEDIUM" if score >= 55 else "LOW")
    return {
        "confidence": confidence,
        "score": round(score, 1),
        "data_completeness_pct": round(completeness, 1),
        "evidence_count": evidence,
        "interpretation": "Confidence menunjukkan kekuatan bukti operasional yang tersedia, bukan probabilitas diagnosis/etiologi.",
    }


def evidence_fusion(
    df: pd.DataFrame,
    acute_event: dict[str, Any] | None = None,
    infectious: dict[str, Any] | None = None,
    disease: str | None = None,
) -> dict[str, Any]:
    windows = multi_window_scan(df)
    finest = finest_unit_intelligence(df)
    unknown = unknown_event_detection(df, disease)
    active = []
    if acute_event and acute_event.get("signals"):
        active.extend(list(acute_event.get("signals", [])))
    if windows.get("best_window", {}).get("cases", 0) >= 5:
        active.append("TEMPORAL_DENSITY")
    if finest.get("highest_signal_unit", {}).get("Kasus", 0) >= 5:
        active.append("FINEST_UNIT_CONCENTRATION")
    if unknown.get("signal"):
        active.append("UNKNOWN_EVENT_PATTERN")
    if infectious and infectious.get("available"):
        active.append("INFECTIOUS_CONTEXT_AVAILABLE")
    active = list(dict.fromkeys(active))
    if not active:
        priority = "ROUTINE_SURVEILLANCE"
        status = "NO_STRONG_SIGNAL"
    elif "UNKNOWN_EVENT_PATTERN" in active or "MASS_CASUALTY_BURST" in active:
        priority = "URGENT_INVESTIGATION"
        status = "SIGNAL"
    else:
        priority = "TARGETED_INVESTIGATION"
        status = "SIGNAL"
    signals = {"active_signals": active}
    uncertainty = uncertainty_intelligence(df, signals)
    return {
        "status": status,
        "priority": priority,
        "active_signals": active,
        "multi_window": windows,
        "finest_unit": finest,
        "unknown_event": unknown,
        "uncertainty": uncertainty,
        "hypotheses": [
            "Outbreak/cluster infeksi",
            "Keracunan atau common exposure",
            "Paparan lingkungan/kimia",
            "Mass gathering",
            "Bencana/incident non-infeksi",
        ] if active else [],
        "guardrail": "Evidence fusion adalah triage intelligence. Tidak menetapkan diagnosis, sumber paparan, KLB legal, atau status bencana.",
    }


def orchestrate_intelligence(
    df: pd.DataFrame,
    disease: str | None = None,
    acute_event: dict[str, Any] | None = None,
    infectious: dict[str, Any] | None = None,
    ml_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Top-level cognitive layer over the existing SI-HIS engines."""
    fused = evidence_fusion(df, acute_event, infectious, disease)
    model_modules = []
    if isinstance(ml_result, dict):
        model_modules = list(ml_result.get("primary_engines", {}).keys())
    return {
        "engine": "SI-HIS Intelligence Orchestrator",
        "version": "1.0",
        "status": fused["status"],
        "priority": fused["priority"],
        "disease": disease,
        "model_modules_used": model_modules,
        "evidence_fusion": fused,
        "next_actions": [
            "Validasi kualitas data dan definisi kasus",
            "Turunkan analisis ke unit geografis terkecil yang tersedia",
            "Periksa line list dan waktu onset/paparan",
            "Bandingkan exposed vs non-exposed bila data tersedia",
            "Verifikasi dengan petugas epidemiologi/klinisi/laboratorium",
        ] if fused["active_signals"] else [
            "Lanjutkan surveillance dan pembaruan data secara berkala."
        ],
        "continuous_learning": {
            "feedback_required": True,
            "recommended_outcomes": ["confirmed_event_type", "laboratory_result", "intervention", "outcome"],
        },
    }
