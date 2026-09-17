"""SI-HIS core compatibility, synthetic national data, and presentation helpers."""

import numpy as np
import pandas as pd

from . import analytics as _analytics

# National synthetic dataset becomes the canonical demo source for the existing
# app.py API without rewriting the validated analytics implementation.
try:
    from .national_dummy import generate_national_dummy
    _analytics.generate_data_simulasi = generate_national_dummy
except Exception:
    generate_national_dummy = None

_original_hitung_bivariat_lengkap = _analytics.hitung_bivariat_lengkap


def _prepare_binary_target(df):
    work = df.copy()
    if "Is_Konfirm" not in work.columns:
        work["Is_Konfirm"] = np.nan
    target = pd.to_numeric(work["Is_Konfirm"], errors="coerce")
    if target.notna().sum() == 0 and "Diagnosis Konfirm" in work.columns:
        target = work["Diagnosis Konfirm"].astype(str).str.strip().ne("Bukan").astype(int)
    work["Is_Konfirm"] = target
    return work


def _hitung_multivariat_logistik(df):
    """Restore the app16-style multivariable logistic regression output."""
    try:
        import statsmodels.api as sm

        work = _prepare_binary_target(df)
        variables = [
            "Umur", "Jenis Kelamin", "Pekerjaan", "Status Imunisasi",
            "Status Komorbid", "Riwayat Perjalanan", "Faktor Risiko Lain",
        ]
        available = [v for v in variables if v in work.columns]
        if not available:
            return pd.DataFrame()

        model_df = work[available + ["Is_Konfirm"]].copy()
        model_df["Umur"] = pd.to_numeric(model_df["Umur"], errors="coerce")
        model_df = model_df.dropna(subset=["Is_Konfirm"])
        if model_df.empty or model_df["Is_Konfirm"].nunique() < 2:
            return pd.DataFrame()

        for col in available:
            if col != "Umur":
                model_df[col] = model_df[col].astype(str)
        model_df = model_df.dropna()
        y = pd.to_numeric(model_df["Is_Konfirm"], errors="coerce").astype(int)
        X = pd.get_dummies(
            model_df[available],
            columns=[c for c in available if c != "Umur"],
            drop_first=True,
            dtype=float,
        )
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]
        if len(X) < 30 or y.nunique() < 2 or X.shape[1] == 0:
            return pd.DataFrame()

        X = sm.add_constant(X, has_constant="add")
        keep = [c for c in X.columns if c == "const" or X[c].nunique(dropna=False) > 1]
        X = X[keep]
        model = sm.Logit(y, X).fit(disp=False, maxiter=200)
        conf = model.conf_int()
        result = pd.DataFrame({
            "Variabel": model.params.index,
            "Koefisien (β)": model.params.values,
            "OR": np.exp(model.params.values),
            "OR_Lower_95%": np.exp(conf[0].values),
            "OR_Upper_95%": np.exp(conf[1].values),
            "p_value": model.pvalues.values,
        })
        result = result[result["Variabel"] != "const"].copy()
        result["Signifikan (p<0.05)"] = result["p_value"] < 0.05
        return result.round({
            "Koefisien (β)": 4, "OR": 4, "OR_Lower_95%": 4,
            "OR_Upper_95%": 4, "p_value": 4,
        })
    except Exception as exc:
        return pd.DataFrame([{
            "Variabel": "MODEL_ERROR",
            "Koefisien (β)": np.nan,
            "OR": np.nan,
            "OR_Lower_95%": np.nan,
            "OR_Upper_95%": np.nan,
            "p_value": np.nan,
            "Signifikan (p<0.05)": False,
            "Keterangan": str(exc),
        }])


def _hitung_bivariat_compat(df, var_indep=None, var_dep_binary=None):
    work = _prepare_binary_target(df)
    if var_indep is not None and var_dep_binary is not None:
        return _original_hitung_bivariat_lengkap(
            work.dropna(subset=[var_indep, var_dep_binary]), var_indep, var_dep_binary
        )

    results = {}
    dependent = "Is_Konfirm"
    independent_vars = [
        "Umur", "Jenis Kelamin", "Pekerjaan", "Status Imunisasi",
        "Status Komorbid", "Riwayat Perjalanan", "Faktor Risiko Lain",
    ]
    if dependent not in work.columns:
        return results
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
                    folium.Circle(
                        location=[lat, lon], radius=500, weight=2, fill=False,
                        tooltip="Zona episentrum / centroid wilayah terfilter",
                    ).add_to(self)
                    folium.Marker(
                        [lat, lon], tooltip="📍 Episentrum",
                        popup=(
                            "<b>📍 Episentrum / Centroid</b><br>"
                            "Titik pusat spasial dari koordinat kasus yang sedang difilter.<br>"
                            "Bukan bukti sumber atau arah penularan."
                        ),
                        icon=folium.Icon(color="red", icon="star"),
                    ).add_to(self)
                except Exception:
                    pass

        folium.Map = SIHISEpicenterMap
except Exception:
    pass

hitung_bivariat_lengkap = _hitung_bivariat_compat
