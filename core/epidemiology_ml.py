"""SI-HIS Epidemiological ML extensions.

This module adds ML-oriented epidemiological signal detection on top of the
existing rule/statistical engine. It intentionally keeps derived labels and
scores separate from legal KLB definitions.

Capabilities:
- anomaly detection with Isolation Forest on daily incidence features;
- temporal change-point screening using rolling z-scores;
- outbreak growth/risk modelling at area level;
- spatial neighbour features;
- population vulnerability clustering;
- multi-signal ensemble for continuous epidemiological surveillance.

All outputs are decision-support signals and require epidemiological validation.
"""
from __future__ import annotations

from typing import Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATE_COL = "Tanggal Sakit"
AREA_COL = "Desa/Kelurahan"


def _to_daily(df: pd.DataFrame, area_col: str = AREA_COL) -> pd.DataFrame:
    if df is None or df.empty or DATE_COL not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
    d = d.dropna(subset=[DATE_COL])
    if area_col not in d.columns:
        d[area_col] = "Indonesia"
    d[area_col] = d[area_col].fillna("Tidak Diketahui").astype(str)
    rows = []
    for area, g in d.groupby(area_col, dropna=False):
        s = g.set_index(DATE_COL).resample("D").size()
        dates = pd.date_range(s.index.min().normalize(), s.index.max().normalize(), freq="D")
        x = s.reindex(dates, fill_value=0).rename("Cases").to_frame()
        x[area_col] = area
        x.index.name = DATE_COL
        rows.append(x.reset_index())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_temporal_features(daily: pd.DataFrame, area_col: str = AREA_COL) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame()
    d = daily.copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
    d["Cases"] = pd.to_numeric(d["Cases"], errors="coerce").fillna(0.0)
    out = []
    for area, g in d.sort_values(DATE_COL).groupby(area_col, sort=False):
        x = g.copy()
        x["Rolling7"] = x["Cases"].rolling(7, min_periods=7).sum()
        x["Rolling14"] = x["Cases"].rolling(14, min_periods=14).sum()
        x["Mean7"] = x["Cases"].rolling(7, min_periods=7).mean()
        x["Std7"] = x["Cases"].rolling(7, min_periods=7).std()
        x["Growth7"] = x["Rolling7"] / x["Rolling7"].shift(7).replace(0, np.nan) - 1.0
        x["Lag1"] = x["Cases"].shift(1)
        x["Lag7"] = x["Cases"].shift(7)
        x["Momentum"] = x["Mean7"] - x["Mean7"].shift(7)
        x["Z7"] = (x["Cases"] - x["Mean7"]) / x["Std7"].replace(0, np.nan)
        out.append(x)
    return pd.concat(out, ignore_index=True)


def detect_temporal_anomalies(
    df: pd.DataFrame,
    area_col: str = AREA_COL,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Screen daily area-level anomalies using Isolation Forest.

    The anomaly score is a screening signal, not a disease/outbreak diagnosis.
    """
    daily = _to_daily(df, area_col)
    feat = build_temporal_features(daily, area_col)
    if feat.empty:
        return feat
    features = [c for c in ["Cases", "Rolling7", "Growth7", "Momentum", "Z7"] if c in feat.columns]
    work = feat.dropna(subset=features).copy()
    if len(work) < 30:
        feat["Anomaly_Score"] = np.nan
        feat["Anomaly_Flag"] = False
        feat["Anomaly_Method"] = "IsolationForest_not_run_insufficient_data"
        return feat
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", IsolationForest(
            n_estimators=300,
            contamination=float(np.clip(contamination, 0.01, 0.20)),
            random_state=random_state,
        )),
    ])
    x = work[features]
    prep.fit(x)
    score = prep.decision_function(x)
    pred = prep.predict(x)
    work["Anomaly_Score"] = -score
    work["Anomaly_Flag"] = pred == -1
    work["Anomaly_Method"] = "IsolationForest"
    result = feat.copy()
    result["Anomaly_Score"] = np.nan
    result["Anomaly_Flag"] = False
    result["Anomaly_Method"] = "IsolationForest"
    result.loc[work.index, ["Anomaly_Score", "Anomaly_Flag"]] = work[["Anomaly_Score", "Anomaly_Flag"]]
    return result


def detect_change_points(
    df: pd.DataFrame,
    area_col: str = AREA_COL,
    baseline_days: int = 14,
    z_threshold: float = 2.0,
) -> pd.DataFrame:
    """Rolling baseline change screening with a pre-change window."""
    daily = _to_daily(df, area_col)
    if daily.empty:
        return daily
    out = []
    for area, g in daily.groupby(area_col, sort=False):
        x = g.sort_values(DATE_COL).copy()
        x["Baseline_Mean"] = x["Cases"].shift(1).rolling(baseline_days, min_periods=baseline_days).mean()
        x["Baseline_Std"] = x["Cases"].shift(1).rolling(baseline_days, min_periods=baseline_days).std()
        x["Change_Z"] = (x["Cases"] - x["Baseline_Mean"]) / x["Baseline_Std"].replace(0, np.nan)
        x["Change_Point_Flag"] = x["Change_Z"] >= float(z_threshold)
        x["Change_Signal"] = np.where(
            x["Change_Point_Flag"], "UPWARD_CHANGE", "NO_CHANGE_SIGNAL"
        )
        out.append(x)
    return pd.concat(out, ignore_index=True)


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_spatial_neighbor_features(
    df: pd.DataFrame,
    radius_km: float = 10.0,
    area_col: str = AREA_COL,
) -> pd.DataFrame:
    """Create neighbour case-density features from observed coordinates."""
    if df is None or df.empty or not {"Latitude", "Longitude", area_col}.issubset(df.columns):
        return pd.DataFrame()
    d = df.copy()
    d["Latitude"] = pd.to_numeric(d["Latitude"], errors="coerce")
    d["Longitude"] = pd.to_numeric(d["Longitude"], errors="coerce")
    d = d.dropna(subset=["Latitude", "Longitude"]).copy()
    if d.empty:
        return pd.DataFrame()
    agg = d.groupby(area_col, as_index=False).agg(
        Latitude=("Latitude", "mean"),
        Longitude=("Longitude", "mean"),
        Cases=(area_col, "size"),
    )
    coords = agg[["Latitude", "Longitude"]].to_numpy(float)
    neighbor_count, neighbor_cases, mean_distance = [], [], []
    for i, (lat, lon) in enumerate(coords):
        dist = _haversine_km(lat, lon, coords[:, 0], coords[:, 1])
        mask = (dist <= radius_km) & (np.arange(len(agg)) != i)
        neighbor_count.append(int(mask.sum()))
        neighbor_cases.append(float(agg.loc[mask, "Cases"].sum()))
        vals = dist[mask]
        mean_distance.append(float(vals.mean()) if len(vals) else np.nan)
    agg["Neighbor_Areas"] = neighbor_count
    agg["Neighbor_Cases"] = neighbor_cases
    agg["Mean_Neighbor_Distance_KM"] = mean_distance
    return agg


def cluster_population_vulnerability(
    df: pd.DataFrame,
    features: Optional[Iterable[str]] = None,
    n_clusters: int = 3,
    min_rows: int = 30,
) -> dict:
    """Unsupervised vulnerability segmentation; requires explicit feature data."""
    if df is None or df.empty:
        return {"status": "error", "message": "Data kosong."}
    default = ["Umur"]
    if "Status Komorbid" in df.columns:
        default.append("Status Komorbid")
    if "Is_Meninggal" in df.columns:
        default.append("Is_Meninggal")
    features = list(features or default)
    use = [c for c in features if c in df.columns]
    if len(use) < 1 or len(df) < min_rows:
        return {"status": "error", "message": f"Minimal {min_rows} baris dan feature valid diperlukan."}
    work = df[use].copy()
    for c in use:
        if work[c].dtype == "object":
            work[c] = pd.factorize(work[c].astype(str))[0]
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < min_rows:
        return {"status": "error", "message": "Baris lengkap untuk clustering belum cukup."}
    k = int(np.clip(n_clusters, 2, min(6, len(work) // 10)))
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=42)),
    ])
    labels = model.fit_predict(work)
    prof = work.copy()
    prof["Cluster"] = labels
    summary = prof.groupby("Cluster").agg(["mean", "count"]).round(3)
    return {
        "status": "ok",
        "model": model,
        "features": use,
        "n_clusters": k,
        "labels": pd.Series(labels, index=work.index, name="Vulnerability_Cluster"),
        "profile": summary,
        "note": "Cluster adalah segmentasi pola, bukan label klinis atau kausalitas.",
    }


def train_growth_risk_model(
    df: pd.DataFrame,
    area_col: str = AREA_COL,
    horizon_days: int = 7,
    quantile: float = 0.75,
    min_rows: int = 60,
) -> dict:
    """Predict elevated near-term case burden using historical area-day data.

    Target is explicitly a model-derived burden threshold, not a legal KLB label.
    """
    daily = _to_daily(df, area_col)
    feat = build_temporal_features(daily, area_col)
    if feat.empty:
        return {"status": "error", "message": "Data temporal kosong."}
    rows = []
    for area, g in feat.groupby(area_col, sort=False):
        x = g.sort_values(DATE_COL).copy()
        x["Future_Total"] = x["Cases"].shift(-1).rolling(horizon_days, min_periods=horizon_days).sum().shift(-(horizon_days - 1))
        rows.append(x)
    p = pd.concat(rows, ignore_index=True)
    future_vals = p["Future_Total"].dropna()
    if future_vals.empty:
        return {"status": "error", "message": "Future target belum terbentuk."}
    # Threshold is estimated only from the training-era portion to avoid
    # leaking future test-period information into the target definition.
    cutoff_date = p[DATE_COL].quantile(0.80)
    threshold_source = p.loc[p[DATE_COL] <= cutoff_date, "Future_Total"].dropna()
    if threshold_source.empty:
        threshold_source = future_vals
    threshold = float(max(5.0, threshold_source.quantile(quantile)))
    p["High_Burden_Target"] = np.where(
        p["Future_Total"].notna(),
        (p["Future_Total"] >= threshold).astype(int),
        np.nan,
    )
    feat_cols = ["Cases", "Rolling7", "Rolling14", "Growth7", "Momentum", "Z7"]
    usable = p.dropna(subset=feat_cols + ["High_Burden_Target"]).copy()
    if len(usable) < min_rows or usable["High_Burden_Target"].nunique() < 2:
        return {"status": "error", "message": "Observasi atau variasi target belum cukup untuk training."}
    usable = usable.sort_values(DATE_COL)
    cut = int(len(usable) * 0.8)
    train, test = usable.iloc[:cut], usable.iloc[cut:]
    if train["High_Burden_Target"].nunique() < 2 or test["High_Burden_Target"].nunique() < 2:
        return {"status": "error", "message": "Temporal holdout hanya memiliki satu kelas."}
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(train[feat_cols], train["High_Burden_Target"])
    proba = model.predict_proba(test[feat_cols])[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(test["High_Burden_Target"], proba)),
        "pr_auc": float(average_precision_score(test["High_Burden_Target"], proba)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "threshold_next7_total": threshold,
    }
    return {
        "status": "ok",
        "model": model,
        "features": feat_cols,
        "metrics": metrics,
        "threshold": threshold,
        "horizon_days": horizon_days,
        "target_definition": "Future area case burden >= historical quantile threshold",
        "legal_status": "not_KLB_definition",
    }


def predict_growth_risk(df: pd.DataFrame, model_result: dict, area_col: str = AREA_COL) -> pd.DataFrame:
    if not model_result or model_result.get("status") != "ok":
        return pd.DataFrame()
    daily = _to_daily(df, area_col)
    feat = build_temporal_features(daily, area_col)
    if feat.empty:
        return pd.DataFrame()
    latest = feat.sort_values(DATE_COL).groupby(area_col, as_index=False).tail(1).copy()
    cols = list(model_result["features"])
    usable = latest.dropna(subset=[c for c in cols if c in latest.columns]).copy()
    if usable.empty:
        return usable
    p = model_result["model"].predict_proba(usable[cols])[:, 1]
    usable["Growth_Risk_7D"] = p
    usable["Growth_Risk_Level"] = pd.cut(
        p, bins=[-0.01, 0.33, 0.66, 1.01], labels=["LOW", "MEDIUM", "HIGH"]
    ).astype(str)
    return usable.sort_values("Growth_Risk_7D", ascending=False)


def build_continuous_epidemiology_signals(
    df: pd.DataFrame,
    area_col: str = AREA_COL,
) -> dict:
    """Aggregate anomaly/change/spatial signals for event-driven surveillance."""
    anomalies = detect_temporal_anomalies(df, area_col)
    changes = detect_change_points(df, area_col)
    spatial = build_spatial_neighbor_features(df, area_col=area_col)
    alerts = []
    if not anomalies.empty:
        a = anomalies[anomalies["Anomaly_Flag"]].copy()
        for _, row in a.iterrows():
            alerts.append({
                "signal_type": "TEMPORAL_ANOMALY",
                "area": row.get(area_col),
                "date": row.get(DATE_COL),
                "severity": "HIGH",
                "evidence": {"anomaly_score": float(row["Anomaly_Score"])},
            })
    if not changes.empty:
        c = changes[changes["Change_Point_Flag"]].copy()
        for _, row in c.iterrows():
            alerts.append({
                "signal_type": "UPWARD_CHANGE_POINT",
                "area": row.get(area_col),
                "date": row.get(DATE_COL),
                "severity": "HIGH",
                "evidence": {"change_z": float(row["Change_Z"])},
            })
    return {
        "status": "ok",
        "temporal_anomalies": anomalies,
        "change_points": changes,
        "spatial_neighbors": spatial,
        "signals": alerts,
        "signal_count": len(alerts),
        "note": "Signal detection is not a legal KLB determination and requires validation.",
    }


# --- Additional epidemiological ML helpers ---

def _safe_binary(series):
    if series is None:
        return pd.Series(dtype=int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series.astype(str).str.strip().str.lower().isin({"1","true","ya","yes","y","positif","positif/konfirm","konfirm"}).astype(int)


def build_case_level_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build reusable epidemiological case features without inventing outcomes."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "Umur" in d.columns:
        d["Umur"] = pd.to_numeric(d["Umur"], errors="coerce")
        d["Age_Risk_65Plus"] = (d["Umur"] >= 65).astype(int)
    for source, target in [("Status Komorbid", "Comorbidity_Flag"), ("Riwayat Perjalanan", "Travel_Flag"), ("Status Imunisasi", "Immunization_Incomplete")]:
        if source in d.columns:
            s = d[source].astype(str).str.lower()
            if source == "Status Komorbid": d[target] = s.isin({"ada komorbid", "ada", "ya", "positif"}).astype(int)
            elif source == "Riwayat Perjalanan": d[target] = s.isin({"ya", "yes", "1", "true"}).astype(int)
            else: d[target] = s.isin({"tidak lengkap", "incomplete"}).astype(int)
    if "Is_Meninggal" in d.columns:
        d["Death_Flag"] = _safe_binary(d["Is_Meninggal"]).fillna(0).astype(int)
    return d


def rank_areas_by_continuous_signal(
    df: pd.DataFrame,
    area_col: str = AREA_COL,
    radius_km: float = 10.0,
    anomaly_weight: float = 0.30,
    change_weight: float = 0.25,
    growth_weight: float = 0.25,
    spatial_weight: float = 0.20,
) -> pd.DataFrame:
    """Combine independent signal components into a transparent 0-100 screening score.

    The score is a prioritisation aid, not a probability and not a legal KLB score.
    """
    anomalies = detect_temporal_anomalies(df, area_col)
    changes = detect_change_points(df, area_col)
    spatial = build_spatial_neighbor_features(df, radius_km=radius_km, area_col=area_col)
    daily = build_temporal_features(_to_daily(df, area_col), area_col)
    parts = []
    if not anomalies.empty:
        a = anomalies.groupby(area_col, as_index=False).agg(Anomaly_Score=("Anomaly_Score", "max"), Anomaly_Flag=("Anomaly_Flag", "sum"))
        a["Anomaly_Component"] = a["Anomaly_Score"].rank(pct=True) * 100
        parts.append(a[[area_col, "Anomaly_Component"]])
    if not changes.empty:
        c = changes.groupby(area_col, as_index=False).agg(Change_Z=("Change_Z", "max"), Change_Flag=("Change_Point_Flag", "sum"))
        c["Change_Component"] = pd.to_numeric(c["Change_Z"], errors="coerce").clip(lower=0).rank(pct=True) * 100
        parts.append(c[[area_col, "Change_Component"]])
    if not daily.empty:
        g = daily.sort_values(DATE_COL).groupby(area_col, as_index=False).tail(1).copy()
        g["Growth_Component"] = pd.to_numeric(g["Growth7"], errors="coerce").clip(lower=0).rank(pct=True) * 100
        parts.append(g[[area_col, "Growth_Component"]])
    if not spatial.empty:
        s = spatial[[area_col, "Neighbor_Cases"]].copy()
        s["Spatial_Component"] = pd.to_numeric(s["Neighbor_Cases"], errors="coerce").fillna(0).rank(pct=True) * 100
        parts.append(s[[area_col, "Spatial_Component"]])
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for p in parts[1:]: out = out.merge(p, on=area_col, how="outer")
    for col in ["Anomaly_Component", "Change_Component", "Growth_Component", "Spatial_Component"]:
        if col not in out: out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    total_w = anomaly_weight + change_weight + growth_weight + spatial_weight
    if total_w <= 0: total_w = 1.0
    out["Continuous_Signal_Score"] = (
        out["Anomaly_Component"] * anomaly_weight +
        out["Change_Component"] * change_weight +
        out["Growth_Component"] * growth_weight +
        out["Spatial_Component"] * spatial_weight
    ) / total_w
    out["Signal_Level"] = pd.cut(out["Continuous_Signal_Score"], [-0.01, 33.33, 66.67, 100.01], labels=["LOW","MEDIUM","HIGH"]).astype(str)
    out["Interpretation"] = "Prioritas screening berbasis gabungan signal; bukan probability, diagnosis, atau status KLB."
    return out.sort_values("Continuous_Signal_Score", ascending=False).reset_index(drop=True)


def train_spatiotemporal_risk_model(
    df: pd.DataFrame,
    area_col: str = AREA_COL,
    radius_km: float = 10.0,
    horizon_days: int = 7,
    quantile: float = 0.75,
    min_rows: int = 80,
) -> dict:
    """Train an area-day risk model using TIME + PLACE features.

    The target is elevated future burden, not legal KLB. Spatial inputs are
    derived only from valid observed coordinates.
    """
    daily = _to_daily(df, area_col)
    temporal = build_temporal_features(daily, area_col)
    spatial = build_spatial_neighbor_features(df, radius_km=radius_km, area_col=area_col)
    if temporal.empty:
        return {"status": "error", "message": "Data temporal kosong."}
    if spatial.empty:
        return {"status": "error", "message": "Koordinat valid belum tersedia untuk fitur spasial."}
    p = temporal.merge(
        spatial[[area_col, "Neighbor_Areas", "Neighbor_Cases"]],
        on=area_col, how="left"
    )
    future_parts = []
    for area, g in p.groupby(area_col, sort=False):
        x = g.sort_values(DATE_COL).copy()
        x["Future_Total"] = sum(
            x["Cases"].shift(-i) for i in range(1, horizon_days + 1)
        )
        future_parts.append(x)
    p = pd.concat(future_parts, ignore_index=True)
    cutoff = p[DATE_COL].quantile(0.80)
    train_future = p.loc[p[DATE_COL] <= cutoff, "Future_Total"].dropna()
    if train_future.empty:
        return {"status": "error", "message": "Baseline target training belum terbentuk."}
    threshold = float(max(5.0, train_future.quantile(quantile)))
    p["Spatiotemporal_Target"] = np.where(
        p["Future_Total"].notna(),
        (p["Future_Total"] >= threshold).astype(int),
        np.nan,
    )
    features = [
        "Cases", "Rolling7", "Rolling14", "Growth7", "Momentum", "Z7",
        "Neighbor_Areas", "Neighbor_Cases",
    ]
    usable = p.dropna(subset=features + ["Spatiotemporal_Target"]).sort_values(DATE_COL)
    if len(usable) < min_rows or usable["Spatiotemporal_Target"].nunique() < 2:
        return {"status": "error", "message": "Observasi atau variasi target TIME+PLACE belum cukup."}
    cut = max(int(len(usable) * 0.80), 1)
    train, test = usable.iloc[:cut], usable.iloc[cut:]
    if train["Spatiotemporal_Target"].nunique() < 2 or test["Spatiotemporal_Target"].nunique() < 2:
        return {"status": "error", "message": "Temporal holdout TIME+PLACE hanya memiliki satu kelas."}
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=4, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(train[features], train["Spatiotemporal_Target"])
    proba = model.predict_proba(test[features])[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(test["Spatiotemporal_Target"], proba)),
        "pr_auc": float(average_precision_score(test["Spatiotemporal_Target"], proba)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    return {
        "status": "ok",
        "model": model,
        "features": features,
        "metrics": metrics,
        "threshold_next7_total": threshold,
        "horizon_days": horizon_days,
        "target_definition": "Future area burden >= training-era historical quantile",
        "dimensions": ["TIME", "PLACE"],
        "legal_status": "not_KLB_definition",
    }


def predict_spatiotemporal_risk(
    df: pd.DataFrame, model_result: dict, area_col: str = AREA_COL,
    radius_km: float = 10.0,
) -> pd.DataFrame:
    if not model_result or model_result.get("status") != "ok":
        return pd.DataFrame()
    temporal = build_temporal_features(_to_daily(df, area_col), area_col)
    spatial = build_spatial_neighbor_features(df, radius_km=radius_km, area_col=area_col)
    if temporal.empty or spatial.empty:
        return pd.DataFrame()
    latest = temporal.sort_values(DATE_COL).groupby(area_col, as_index=False).tail(1)
    use = latest.merge(
        spatial[[area_col, "Neighbor_Areas", "Neighbor_Cases"]],
        on=area_col, how="left"
    )
    features = list(model_result["features"])
    if any(c not in use.columns for c in features):
        return pd.DataFrame()
    use = use.dropna(subset=features).copy()
    if use.empty:
        return use
    p = model_result["model"].predict_proba(use[features])[:, 1]
    use["Spatiotemporal_Risk_7D"] = p
    use["Spatiotemporal_Risk_Level"] = pd.cut(
        p, [-0.01, 0.33, 0.66, 1.01],
        labels=["LOW", "MEDIUM", "HIGH"]
    ).astype(str)
    return use.sort_values("Spatiotemporal_Risk_7D", ascending=False)


def train_disease_specific_growth_models(
    df: pd.DataFrame,
    diseases: Optional[Iterable[str]] = None,
    area_col: str = AREA_COL,
) -> dict:
    """Train independent growth-risk models per disease.

    Disease-specific models prevent one disease's epidemic curve from being
    treated as the temporal behaviour of another disease.
    """
    if df is None or df.empty:
        return {}
    disease_cols = ["Diagnosis Konfirm", "Diagnosis Probabel", "Diagnosis Suspek"]
    values = set()
    for col in disease_cols:
        if col in df.columns:
            values.update(
                x for x in df[col].astype(str).str.strip().unique()
                if x and x.lower() not in {"bukan", "tidak ada", "nan", "none"}
            )
    selected = list(diseases) if diseases else sorted(values)
    results = {}
    for disease in selected:
        mask = pd.Series(False, index=df.index)
        for col in disease_cols:
            if col in df.columns:
                mask |= df[col].astype(str).str.strip().eq(disease)
        sub = df.loc[mask].copy()
        if len(sub) >= 60:
            results[str(disease)] = train_growth_risk_model(sub, area_col=area_col)
        else:
            results[str(disease)] = {
                "status": "error",
                "message": "Data penyakit belum mencapai minimal 60 baris untuk model disease-specific."
            }
    return results
