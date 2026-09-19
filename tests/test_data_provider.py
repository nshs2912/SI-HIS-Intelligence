import pandas as pd

from core.data_provider import canonicalize_factual_data, get_source_catalog, load_source, validate_case_schema

def test_source_catalog_contains_demo_and_operational_sources():
    ids = {item["source_id"] for item in get_source_catalog()}
    assert "dummy" in ids
    assert {"simpus", "fktp", "hospital", "nutrimed_mobile", "laboratory", "pharmacy"}.issubset(ids)

def test_factual_aliases_are_canonicalized():
    df = pd.DataFrame({
        "patient_name": ["P1"], "age": [40], "gender": ["Laki-laki"],
        "tanggal_kunjungan": ["2026-09-01"], "province": ["Jawa Barat"],
        "district": ["Kabupaten Bandung"], "diagnosis": ["ISPA"], "confirmed": [1],
    })
    out = canonicalize_factual_data(df)
    assert out.loc[0, "Nama"] == "P1"
    assert out.loc[0, "Umur"] == 40
    assert out.loc[0, "Jenis Kelamin"] == "Laki-laki"
    assert out.loc[0, "Tanggal Sakit"] == "2026-09-01"
    assert out.loc[0, "Provinsi"] == "Jawa Barat"
    assert out.loc[0, "Diagnosis Konfirm"] == "ISPA"
    assert out.loc[0, "Is_Konfirm"] == 1

def test_factual_source_uses_same_canonical_boundary():
    payload = b"Nama,Umur,Jenis Kelamin,Tanggal Sakit,Provinsi,Kabupaten\nP1,40,Laki-laki,2026-09-01,Jawa Barat,Kabupaten Bandung\n"
    class Upload:
        name = "factual.csv"
        def getvalue(self): return payload
    df, metadata = load_source("simpus", factual_file=Upload())
    assert metadata["source_mode"] == "FACTUAL"
    assert metadata["integration_status"] == "adapter_ready"
    assert df.loc[0, "Nama"] == "P1"
    assert "Diagnosis Konfirm" in validate_case_schema(df)

def test_dummy_source_is_explicit():
    df, metadata = load_source("dummy", target_rows=25, days=30)
    assert len(df) == 25
    assert metadata["source_mode"] == "DEMO"
