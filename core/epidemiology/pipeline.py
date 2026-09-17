"""Context-aware epidemiological analysis pipeline.

The pipeline enforces the SI-HIS analytical rule:
TIME + PERSON + PLACE are the epidemiological core. Disease and geographic
scope are resolved before disease-specific temporal/spatial interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DiseaseProfile:
    name: str
    transmission: str = "unknown"
    human_to_human: bool | None = None
    temporal_interpretation: str = "generic"
    relevant_exposures: tuple[str, ...] = ()


DISEASE_PROFILES = {
    "Leptospirosis": DiseaseProfile(
        "Leptospirosis", "zoonotic/environmental", False,
        "exposure_episode", ("Pekerjaan", "Riwayat Perjalanan", "Faktor Risiko Lain")
    ),
    "Demam Dengue": DiseaseProfile(
        "Demam Dengue", "vector-borne", False,
        "vector_environment", ("Faktor Risiko Lain", "Pekerjaan", "Riwayat Perjalanan")
    ),
    "ISPA Berat": DiseaseProfile(
        "ISPA Berat", "respiratory", True,
        "transmission_wave", ("Pekerjaan", "Riwayat Perjalanan", "Status Komorbid")
    ),
    "Diare Akut": DiseaseProfile(
        "Diare Akut", "fecal-oral/environmental", None,
        "exposure_episode", ("Faktor Risiko Lain", "Riwayat Perjalanan")
    ),
}


@dataclass
class AnalysisEligibility:
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def resolve_disease_profile(disease: str | None) -> DiseaseProfile:
    if not disease or disease == "Semua Penyakit":
        return DiseaseProfile("Semua Penyakit", "mixed/unknown", None, "overview")
    return DISEASE_PROFILES.get(disease, DiseaseProfile(disease))


def validate_scope_for_special_analysis(
    df: pd.DataFrame,
    disease: str | None,
    geographic_level: str | None,
    minimum_cases: int = 10,
    minimum_days: int = 14,
) -> AnalysisEligibility:
    reasons: list[str] = []
    warnings: list[str] = []
    profile = resolve_disease_profile(disease)

    if profile.name == "Semua Penyakit":
        reasons.append("Analisis khusus penyakit memerlukan satu penyakit, bukan Semua Penyakit.")
    if geographic_level not in {"Kabupaten", "Kecamatan", "Desa/Kelurahan", "Puskesmas"}:
        reasons.append("Analisis khusus harus memiliki scope geografis spesifik minimal Kabupaten/Kota.")
    if len(df) < minimum_cases:
        reasons.append(f"Kasus belum mencukupi: {len(df)} < {minimum_cases}.")
    if "Tanggal Sakit" in df.columns:
        dates = pd.to_datetime(df["Tanggal Sakit"], errors="coerce").dropna()
        if not dates.empty and (dates.max() - dates.min()).days + 1 < minimum_days:
            warnings.append("Rentang observasi kurang dari periode minimum yang disarankan.")
    else:
        reasons.append("Tanggal Sakit tidak tersedia.")
    return AnalysisEligibility(not reasons, reasons, warnings)


def classify_temporal_pattern_for_disease(
    profile: DiseaseProfile,
    detected_peaks: int,
) -> dict[str, Any]:
    """Translate mathematical peaks into disease-aware language.

    A peak detector alone never proves human-to-human transmission.
    """
    if profile.temporal_interpretation == "transmission_wave":
        return {
            "pattern_type": "POTENTIAL_TRANSMISSION_WAVE" if detected_peaks >= 2 else "TEMPORAL_PEAK",
            "interpretation": "Pola temporal konsisten dengan kemungkinan gelombang transmisi; perlu dikonfirmasi dengan data epidemiologis lain.",
            "transmission_claim_allowed": False,
        }
    if detected_peaks:
        return {
            "pattern_type": "TEMPORAL_EXPOSURE_EPISODES",
            "interpretation": "Terdapat beberapa puncak temporal. Untuk penyakit ini, puncak tidak boleh ditafsirkan sebagai penularan antar-manusia tanpa bukti tambahan.",
            "transmission_claim_allowed": False,
        }
    return {
        "pattern_type": "NO_CLEAR_TEMPORAL_PEAK",
        "interpretation": "Tidak ditemukan puncak temporal yang cukup kuat pada scope ini.",
        "transmission_claim_allowed": False,
    }
