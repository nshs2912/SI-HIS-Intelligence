import pandas as pd

from core.data_provider import get_cases
from core.scope import QueryScope, apply_scope
from core.statistics import AGE_GROUPS, add_age_groups, hitung_bivariat_lengkap
from core.engine import SIHISIntelligenceEngine
from core.spatial import run_dbscan, compute_epicenter


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


def test_engine_returns_canonical_sections():
    df = get_cases(days=90, target_rows=500, seed=321)
    result = SIHISIntelligenceEngine().analyze(df)
    for key in ["overview", "person", "place", "time", "mortality", "risk", "risk_factors", "vulnerable", "ews", "forecast", "spatial", "epicenters", "ml", "provenance"]:
        assert key in result
    assert result["provenance"]["ml_enabled"] is False
