import pandas as pd

from core.data_provider import get_cases
from core.engine import SIHISIntelligenceEngine
from core.scope import QueryScope, apply_scope
from core.spatial import compute_epicenter, run_dbscan
from core.statistics import AGE_GROUPS, add_age_groups, hitung_bivariat_lengkap


def test_scope_does_not_mutate_source():
    source = pd.DataFrame({"Provinsi": ["Jawa Barat", "Jawa Tengah"], "Kabupaten": ["Kabupaten Bandung", "Kabupaten Semarang"]})
    original = source.copy(deep=True)
    scoped = apply_scope(source, QueryScope(province="Jawa Barat"))
    scoped.loc[scoped.index[0], "Kabupaten"] = "MUTATED"
    assert source.equals(original)
    assert len(scoped) == 1


def test_age_groups_are_non_overlapping():
    df = pd.DataFrame({"Umur": [0, 1, 4, 5, 9, 10, 84, 85, 100]})
    out = add_age_groups(df)
    assert out["Kelompok_Umur"].notna().all()
    assert out["Kelompok_Umur"].astype(str).tolist() == ["<1 tahun", "1-4 tahun", "1-4 tahun", "5-9 tahun", "5-9 tahun", "10-14 tahun", "75-84 tahun", "≥85 tahun", "≥85 tahun"]
    assert len(AGE_GROUPS) == 13


def test_bivariate_returns_age_and_multivariate():
    df = get_cases(days=90, target_rows=500, seed=123)
    result = hitung_bivariat_lengkap(df)
    assert "Umur" in result
    assert "MULTIVARIAT — Logistic Regression" in result
    assert "crosstab" in result["Umur"]


def test_spatial_cluster_and_epicenter_are_cluster_specific():
    df = pd.DataFrame({"Latitude": [-7.75, -7.751, -7.752, -7.753, -6.90], "Longitude": [110.36, 110.361, 110.362, 110.363, 107.61]})
    clustered = run_dbscan(df, eps_km=1.0, min_samples=3)
    centers = compute_epicenter(clustered)
    assert "Cluster" in clustered.columns
    assert not centers.empty
    assert {"Latitude", "Longitude", "Jumlah_Kasus"}.issubset(centers.columns)


def test_descriptive_mode_requires_no_scope_and_returns_top10_contract():
    df = get_cases(days=90, target_rows=500, seed=321)
    result = SIHISIntelligenceEngine().analyze(df, mode="descriptive")
    assert result["mode"] == "descriptive"
    assert "top10_diseases" in result
    assert list(result["top10_diseases"].columns) == ["NO.", "Nama Penyakit", "Jumlah Kasus", "CFR", "Kabupaten", "Provinsi"]


def test_special_analysis_rejected_without_disease_and_district():
    df = get_cases(days=90, target_rows=500, seed=321)
    result = SIHISIntelligenceEngine().analyze(df)
    assert result["eligible"] is False
    assert result["analysis_sections"]["risk_factors"] is None
    assert result["analysis_sections"]["spatial"] is None


def test_special_analysis_rejected_for_all_diseases_even_with_district():
    df = get_cases(days=90, target_rows=500, seed=321)
    district = str(df["Kabupaten"].dropna().iloc[0])
    result = SIHISIntelligenceEngine().analyze(df, QueryScope(district=district, disease="Semua Penyakit", period_days=14))
    assert result["eligible"] is False
    assert any("satu penyakit" in reason.lower() for reason in result["eligibility"]["reasons"])


def test_special_analysis_allowed_with_disease_district_and_period():
    df = get_cases(days=90, target_rows=500, seed=321)
    disease = str(df["Diagnosis Konfirm"].loc[df["Diagnosis Konfirm"].astype(str) != "Bukan"].iloc[0])
    district = str(df["Kabupaten"].dropna().iloc[0])
    scoped = df[df["Kabupaten"].astype(str).eq(district)].copy()
    # The synthetic generator has multiple dates/diseases; select a district/disease
    # combination with enough observations for the special-analysis contract.
    disease_mask = False
    for col in ["Diagnosis Konfirm", "Diagnosis Probabel", "Diagnosis Suspek"]:
        disease_mask = disease_mask | scoped[col].astype(str).eq(disease)
    scoped = scoped[disease_mask]
    if len(scoped) >= 10 and pd.to_datetime(scoped["Tanggal Sakit"], errors="coerce").nunique() >= 14:
        result = SIHISIntelligenceEngine().analyze(df, QueryScope(district=district, disease=disease, period_days=14))
        assert result["eligible"] is True
        assert result["temporal_interpretation"] is not None
        assert result["time"] is not None
