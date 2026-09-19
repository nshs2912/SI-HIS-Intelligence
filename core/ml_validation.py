"""Validation utilities for epidemiological ML.

These functions quantify probability quality and dataset drift. They do not
turn model metrics into clinical or legal decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


def calibration_summary(y_true, probability, bins: int = 10) -> dict:
    y = pd.Series(y_true).astype(float)
    p = pd.Series(probability).astype(float).clip(1e-6, 1 - 1e-6)
    valid = y.notna() & p.notna()
    y, p = y[valid], p[valid]
    if len(y) < 30 or y.nunique() < 2:
        return {"status": "error", "message": "Data kalibrasi belum cukup atau outcome hanya satu kelas."}
    brier = float(brier_score_loss(y, p))
    auc = float(roc_auc_score(y, p))
    pr_auc = float(average_precision_score(y, p))
    edges = np.linspace(0, 1, bins + 1)
    bucket = pd.cut(p, bins=edges, include_lowest=True, duplicates="drop")
    cal = pd.DataFrame({"y": y.to_numpy(), "p": p.to_numpy(), "bin": bucket.to_numpy()})
    grouped = cal.groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    ).reset_index()
    grouped["absolute_gap"] = (grouped["predicted"] - grouped["observed"]).abs()
    return {
        "status": "ok",
        "n": int(len(y)),
        "brier": brier,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "mean_absolute_calibration_gap": float(
            np.average(grouped["absolute_gap"], weights=grouped["n"])
        ),
        "calibration_table": grouped,
        "interpretation": (
            "Brier dan calibration gap menggambarkan kualitas probabilitas pada "
            "data evaluasi; keduanya tidak membuktikan validitas epidemiologis."
        ),
    }


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(float)
    if len(ref) < 20 or len(cur) < 20:
        return float("nan")
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    ref_bins = np.clip(np.digitize(ref, quantiles[1:-1], right=True), 0, len(quantiles) - 2)
    cur_bins = np.clip(np.digitize(cur, quantiles[1:-1], right=True), 0, len(quantiles) - 2)
    ref_rate = np.bincount(ref_bins, minlength=len(quantiles)-1).astype(float)
    cur_rate = np.bincount(cur_bins, minlength=len(quantiles)-1).astype(float)
    ref_rate = np.maximum(ref_rate / ref_rate.sum(), 1e-6)
    cur_rate = np.maximum(cur_rate / cur_rate.sum(), 1e-6)
    return float(np.sum((cur_rate - ref_rate) * np.log(cur_rate / ref_rate)))


def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    psi_threshold: float = 0.20,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        psi = population_stability_index(reference[feature], current[feature])
        if np.isnan(psi):
            level = "INSUFFICIENT_DATA"
        elif psi >= psi_threshold:
            level = "DRIFT_REVIEW"
        else:
            level = "STABLE_SCREENING"
        rows.append({"Feature": feature, "PSI": psi, "Drift_Level": level})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def dataset_drift_summary(
    df: pd.DataFrame,
    date_col: str = "Tanggal Sakit",
    features: list[str] | None = None,
) -> dict:
    if df is None or df.empty or date_col not in df.columns:
        return {"status": "error", "message": "Tanggal atau dataset tidak tersedia."}
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col)
    if len(d) < 60:
        return {"status": "error", "message": "Minimal 60 observasi untuk screening drift dataset."}
    cut_ref = int(len(d) * 0.60)
    cut_cur = int(len(d) * 0.80)
    ref, cur = d.iloc[:cut_ref], d.iloc[cut_cur:]
    if features is None:
        features = [
            c for c in d.columns
            if pd.api.types.is_numeric_dtype(d[c])
            and c != date_col
        ][:20]
    table = detect_feature_drift(ref, cur, features)
    review = int((table["Drift_Level"] == "DRIFT_REVIEW").sum()) if not table.empty else 0
    return {
        "status": "ok",
        "reference_start": ref[date_col].min(),
        "reference_end": ref[date_col].max(),
        "current_start": cur[date_col].min(),
        "current_end": cur[date_col].max(),
        "features_checked": list(table["Feature"]) if not table.empty else [],
        "drift_review_count": review,
        "table": table,
        "interpretation": (
            "PSI adalah screening perubahan distribusi data, bukan bukti perubahan "
            "epidemiologi atau penurunan kinerja model."
        ),
    }
