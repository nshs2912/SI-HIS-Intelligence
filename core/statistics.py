"""Epidemiological statistical analysis layer.

Keeps bivariate and multivariable analysis outside the package initializer.
Age is analysed as mutually-exclusive categorical groups rather than raw age.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

AGE_GROUPS = [
    "<1 tahun", "1-4 tahun", "5-9 tahun", "10-14 tahun", "15-19 tahun",
    "20-24 tahun", "25-34 tahun", "35-44 tahun", "45-54 tahun",
    "55-64 tahun", "65-74 tahun", "75-84 tahun", "≥85 tahun",
]

AGE_BINS = [-np.inf, 1, 5, 10, 15, 20, 25, 35, 45, 55, 65, 75, 85, np.inf]

INDEPENDENT_VARS = [
    "Umur", "Jenis Kelamin", "Pekerjaan", "Status Imunisasi",
    "Status Komorbid", "Riwayat Perjalanan", "Faktor Risiko Lain",
]


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy(deep=True)
    if "Umur" not in work.columns:
        return work
    age = pd.to_numeric(work["Umur"], errors="coerce")
    work["Kelompok_Umur"] = pd.cut(
        age, bins=AGE_BINS, labels=AGE_GROUPS, right=False, include_lowest=True
    )
    work["Kategori_Umur"] = work["Kelompok_Umur"].map(
        {label: i + 1 for i, label in enumerate(AGE_GROUPS)}
    ).astype("Int64")
    return work


def prepare_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    work = add_age_groups(df)
    if "Is_Konfirm" not in work.columns:
        work["Is_Konfirm"] = np.nan
    target = pd.to_numeric(work["Is_Konfirm"], errors="coerce")
    if target.notna().sum() == 0 and "Diagnosis Konfirm" in work.columns:
        target = work["Diagnosis Konfirm"].astype(str).str.strip().ne("Bukan").astype(int)
    work["Is_Konfirm"] = target
    return work


def _binary_table(work: pd.DataFrame, variable: str, target: str = "Is_Konfirm") -> dict:
    d = work.dropna(subset=[variable, target]).copy()
    ct = pd.crosstab(d[variable], d[target])
    try:
        chi2, p_value, _, _ = stats.chi2_contingency(ct)
    except Exception:
        chi2, p_value = np.nan, np.nan
    return {"crosstab": ct, "chi2": chi2, "p_value": p_value}


def _age_analysis(work: pd.DataFrame) -> dict:
    d = work.dropna(subset=["Kelompok_Umur", "Is_Konfirm"]).copy()
    if d.empty:
        return {"crosstab": pd.DataFrame(), "chi2": np.nan, "p_value": np.nan, "or_by_group": pd.DataFrame()}
    ct = pd.crosstab(d["Kelompok_Umur"], d["Is_Konfirm"]).reindex(AGE_GROUPS, fill_value=0)
    try:
        chi2, p_value, _, _ = stats.chi2_contingency(ct)
    except Exception:
        chi2, p_value = np.nan, np.nan
    levels = [g for g in AGE_GROUPS if ct.loc[g].sum() > 0]
    rows = []
    if levels:
        ref_name = levels[0]
        ref = ct.loc[ref_name]
        for level in levels:
            if level == ref_name:
                rows.append({"Kelompok Umur": level, "Kategori": AGE_GROUPS.index(level) + 1, "Referensi": "Ya", "OR": 1.0, "OR_Lower_95%": np.nan, "OR_Upper_95%": np.nan})
                continue
            row = ct.loc[level]
            a, b, c, e = float(row.get(1, 0)) + .5, float(row.get(0, 0)) + .5, float(ref.get(1, 0)) + .5, float(ref.get(0, 0)) + .5
            or_value = (a * e) / (b * c)
            se = math.sqrt(1/a + 1/b + 1/c + 1/e)
            rows.append({"Kelompok Umur": level, "Kategori": AGE_GROUPS.index(level) + 1, "Referensi": ref_name, "OR": or_value, "OR_Lower_95%": math.exp(math.log(or_value) - 1.96 * se), "OR_Upper_95%": math.exp(math.log(or_value) + 1.96 * se)})
    return {"crosstab": ct, "chi2": chi2, "p_value": p_value, "or_by_group": pd.DataFrame(rows).round(4)}


def multivariable_logistic(df: pd.DataFrame) -> pd.DataFrame:
    work = prepare_binary_target(df)
    variables = ["Kelompok_Umur", "Jenis Kelamin", "Pekerjaan", "Status Imunisasi", "Status Komorbid", "Riwayat Perjalanan", "Faktor Risiko Lain"]
    available = [v for v in variables if v in work.columns]
    if not available:
        return pd.DataFrame()
    model_df = work[available + ["Is_Konfirm"]].dropna().copy()
    if model_df.empty or model_df["Is_Konfirm"].nunique() < 2 or len(model_df) < 30:
        return pd.DataFrame()
    for col in available:
        model_df[col] = model_df[col].astype(str)
    X = pd.get_dummies(model_df[available], columns=available, drop_first=True, dtype=float)
    y = pd.to_numeric(model_df["Is_Konfirm"], errors="coerce").astype(int)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    keep = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
    X = X[keep]
    if X.empty or y.nunique() < 2:
        return pd.DataFrame()
    try:
        model = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=False, maxiter=300)
        conf = model.conf_int()
        out = pd.DataFrame({"Variabel": model.params.index, "Koefisien (β)": model.params.values, "OR Adjusted": np.exp(model.params.values), "OR_Lower_95%": np.exp(conf[0].values), "OR_Upper_95%": np.exp(conf[1].values), "p_value": model.pvalues.values})
        out = out[out["Variabel"] != "const"].copy()
        out["Signifikan (p<0.05)"] = out["p_value"] < .05
        return out.round(4)
    except Exception as exc:
        return pd.DataFrame([{"Variabel": "MODEL_ERROR", "Koefisien (β)": np.nan, "OR Adjusted": np.nan, "OR_Lower_95%": np.nan, "OR_Upper_95%": np.nan, "p_value": np.nan, "Signifikan (p<0.05)": False, "Keterangan": str(exc)}])


def hitung_bivariat_lengkap(df: pd.DataFrame, var_indep=None, var_dep_binary=None):
    work = prepare_binary_target(df)
    if var_indep is not None and var_dep_binary is not None:
        return _binary_table(work, var_indep, var_dep_binary)
    results = {"Umur": _age_analysis(work)}
    for variable in INDEPENDENT_VARS[1:]:
        if variable in work.columns:
            results[variable] = _binary_table(work, variable)
    results["MULTIVARIAT — Logistic Regression"] = multivariable_logistic(work)
    return results
