import pandas as pd

from core.intelligence_orchestrator import (
    multi_window_scan,
    finest_unit_intelligence,
    unknown_event_detection,
    evidence_fusion,
)


def _df():
    base = pd.Timestamp("2026-09-19 08:00")
    return pd.DataFrame({
        "Tanggal Onset": [base + pd.Timedelta(minutes=i*5) for i in range(12)],
        "Provinsi": ["Jawa Tengah"] * 12,
        "Kabupaten": ["Kabupaten X"] * 12,
        "Kecamatan": ["Kecamatan A"] * 12,
        "Desa/Kelurahan": ["Desa 1"] * 8 + ["Desa 2"] * 4,
        "Puskesmas": ["Puskesmas A"] * 8 + ["Puskesmas B"] * 4,
        "Diagnosis Konfirm": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
    })


def test_multi_window_finds_dense_burst():
    result = multi_window_scan(_df())
    assert result["status"] == "ok"
    assert result["best_window"]["cases"] >= 5


def test_finest_unit_identifies_puskesmas():
    result = finest_unit_intelligence(_df())
    assert result["status"] == "ok"
    assert result["finest_available_level"] == "Puskesmas"
    assert result["highest_signal_unit"]["Kasus"] == 8


def test_unknown_event_detects_heterogeneous_cluster():
    result = unknown_event_detection(_df())
    assert result["signal"] is True


def test_evidence_fusion_keeps_guardrail():
    result = evidence_fusion(_df(), acute_event={"signals": ["MASS_CASUALTY_BURST"]})
    assert result["priority"] == "URGENT_INVESTIGATION"
    assert "KLB legal" in result["guardrail"]
