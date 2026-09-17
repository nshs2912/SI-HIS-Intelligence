"""SI-HIS core compatibility, synthetic national data, and presentation helpers."""

import math
import numpy as np
import pandas as pd

from . import analytics as _analytics

try:
    from .national_dummy import generate_national_dummy
    _analytics.generate_data_simulasi = generate_national_dummy
except Exception:
    generate_national_dummy = None

_original_hitung_bivariat_lengkap = _analytics.hitung_bivariat_lengkap

AGE_GROUPS = [
    "<1 tahun", "1-4 tahun", "5-9 tahun", "10-14 tahun", "15-19 tahun",
    "20-24 tahun", "25-34 tahun", "35-44 tahun", "45-54 tahun",
    "55-64 tahun", "65-74 tahun", "75-84 tahun", "≥85 tahun",
]


def _add_age_group(df):
    work = df.copy()
    if "Umur" in work.columns:
        age = pd.to_numeric(work["Umur"], errors="coerce")
        bins = [-np.inf, 1, 5, 10, 15, 20, 25, 35, 45, 55, 65, 75, 85, np.inf]
        # Intervals are left-open/right-closed for the infant boundary, then
        # converted to explicit mutually-exclusive labels.
        work["Kelompok_Umur"] = pd.cut(
            age,
            bins=bins,
            labels=AGE_GROUPS,
            right=False,
            include_lowest=True,
        )
        # With right=False the first interval is [0,1), then 1-4, etc.
        # This keeps every age in exactly one group and avoids overlap.
        work["Kategori_Umur"] = work["Kelompok_Umur"].map(
            {label: i + 1 for i, label in enumerate(AGE_GROUPS)}
        ).astype("Int64")
    return work


def _prepare_binary_target(df):
    work = _add_age_group(df)
    if "Is_Konfirm" not in work.columns:
        work["Is_Konfirm"] = np.nan
    target = pd.to_numeric(work["Is_Konfirm"], errors="coerce")
    if target.notna().sum() == 0 and "Diagnosis Konfirm" in work.columns:
        target = work["Diagnosis Konfirm"].astype(str).str.strip().ne("Bukan").astype(int)
    work["Is_Konfirm"] = target
    return work


def _age_bivariate(df):
    work = _prepare_binary_target(df).dropna(subset=["Kelompok_Umur", "Is_Konfirm"]).copy()
    if work.empty:
        return {"crosstab": pd.DataFrame(), "chi2": np.nan, "p_value": np.nan, "or_by_group": pd.DataFrame()}

    ct = pd.crosstab(work["Kelompok_Umur"], work["Is_Konfirm"])
    ct = ct.reindex(AGE_GROUPS, fill_value=0)
    try:
        import scipy.stats as stats
        chi2, p_value, _, _ = stats.chi2_contingency(ct)
    except Exception:
        chi2, p_value = np.nan, np.nan

    # OR is reported for each age category against the first available group,
    # rather than forcing a single OR across 13 categories.
    rows = []
    levels = [x for x in AGE_GROUPS if x in ct.index and ct.loc[x].sum() > 0]
    if levels:
        reference = levels[0]
        ref = ct.loc[reference]
        for level in levels:
            if level == reference:
                rows.append({"Kelompok Umur": level, "Kategori": AGE_GROUPS.index(level) + 1,
                             "Referensi": "Ya", "OR": 1.0, "OR_Lower_95%": np.nan,
                             "OR_Upper_95%": np.nan})
                continue
            row = ct.loc[level]
            # Target columns are 0/1; add continuity correction for zero cells.
            a = float(row.get(1, 0)); b = float(row.get(0, 0))
            c = float(ref.get(1, 0)); d = float(ref.get(0, 0))
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            or_val = (a * d) / (b * c)
            se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            rows.append({
                "Kelompok Umur": level,
                "Kategori": AGE_GROUPS.index(level) + 1,
                "Referensi": f"{reference}",
                "OR": or_val,
                "OR_Lower_95%": math.exp(math.log(or_val) - 1.96 * se),
                "OR_Upper_95%": math.exp(math.log(or_val) + 1.96 * se),
            })
    return {
        "crosstab": ct,
        "chi2": chi2,
        "p_value": p_value,
        "or_by_group": pd.DataFrame(rows).round(4),
    }


def _hitung_multivariat_logistik(df):
    """Multivariable logistic regression with age treated as categorical groups."""
    try:
        import statsmodels.api as sm
        work = _prepare_binary_target(df)
        variables = [
            "Kelompok_Umur", "Jenis Kelamin", "Pekerjaan", "Status Imunisasi",
            "Status Komorbid", "Riwayat Perjalanan", "Faktor Risiko Lain",
        ]
        available = [v for v in variables if v in work.columns]
        model_df = work[available + ["Is_Konfirm"]].dropna().copy()
        if model_df.empty or model_df["Is_Konfirm"].nunique() < 2:
            return pd.DataFrame()
        for col in available:
            if col != "Kelompok_Umur":
                model_df[col] = model_df[col].astype(str)
        cat_cols = [c for c in available if c != "Kelompok_Umur"]
        X = pd.get_dummies(model_df[available], columns=["Kelompok_Umur"] + cat_cols, drop_first=True, dtype=float)
        y = pd.to_numeric(model_df["Is_Konfirm"], errors="coerce").astype(int)
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]
        if len(X) < 30 or y.nunique() < 2 or X.shape[1] == 0:
            return pd.DataFrame()
        X = sm.add_constant(X, has_constant="add")
        keep = [c for c in X.columns if c == "const" or X[c].nunique(dropna=False) > 1]
        X = X[keep]
        model = sm.Logit(y, X).fit(disp=False, maxiter=300)
        conf = model.conf_int()
        result = pd.DataFrame({
            "Variabel": model.params.index,
            "Koefisien (β)": model.params.values,
            "OR Adjusted": np.exp(model.params.values),
            "OR_Lower_95%": np.exp(conf[0].values),
            "OR_Upper_95%": np.exp(conf[1].values),
            "p_value": model.pvalues.values,
        })
        result = result[result["Variabel"] != "const"].copy()
        result["Signifikan (p<0.05)"] = result["p_value"] < 0.05
        return result.round(4)
    except Exception as exc:
        return pd.DataFrame([{
            "Variabel": "MODEL_ERROR", "Koefisien (β)": np.nan,
            "OR Adjusted": np.nan, "OR_Lower_95%": np.nan,
            "OR_Upper_95%": np.nan, "p_value": np.nan,
            "Signifikan (p<0.05)": False, "Keterangan": str(exc),
        }])


def _hitung_bivariat_compat(df, var_indep=None, var_dep_binary=None):
    work = _prepare_binary_target(df)
    if var_indep is not None and var_dep_binary is not None:
        return _original_hitung_bivariat_lengkap(work.dropna(subset=[var_indep, var_dep_binary]), var_indep, var_dep_binary)

    results = {}
    dependent = "Is_Konfirm"
    independent_vars = [
        "Jenis Kelamin", "Pekerjaan", "Status Imunisasi",
        "Status Komorbid", "Riwayat Perjalanan", "Faktor Risiko Lain",
    ]
    results["Umur"] = _age_bivariate(work)
    for variable in independent_vars:
        if variable not in work.columns:
            continue
        try:
            results[variable] = _original_hitung_bivariat_lengkap(
                work.dropna(subset=[variable, dependent]), variable, dependent
            )
        except Exception as exc:
            results[variable] = {"error": str(exc)}
    results["MULTIVARIAT — Logistic Regression"] = _hitung_multivariat_logistik(work)
    return results


_analytics.hitung_bivariat_lengkap = _hitung_bivariat_compat

try:
    import folium
    if not getattr(folium.Map, "_sihis_epicenter_patch", False):
        _OriginalMap = folium.Map
        class SIHISEpicenterMap(_OriginalMap):
            _sihis_epicenter_patch = True
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                try:
                    lat, lon = float(self.location[0]), float(self.location[1])
                    folium.Circle(location=[lat, lon], radius=500, weight=2, fill=False,
                                  tooltip="Zona episentrum / centroid wilayah terfilter").add_to(self)
                    folium.Marker([lat, lon], tooltip="📍 Episentrum",
                                  popup=("<b>📍 Episentrum / Centroid</b><br>"
                                         "Titik pusat spasial dari koordinat kasus yang sedang difilter.<br>"
                                         "Bukan bukti sumber atau arah penularan."),
                                  icon=folium.Icon(color="red", icon="star")).add_to(self)
                except Exception:
                    pass
        folium.Map = SIHISEpicenterMap
except Exception:
    pass

hitung_bivariat_lengkap = _hitung_bivariat_compat
