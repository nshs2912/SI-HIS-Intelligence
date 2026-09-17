"""Disease-aware epidemiology rules used by the SI-HIS engine."""
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
    "Leptospirosis": DiseaseProfile("Leptospirosis", "zoonotic/environmental", False, "exposure_episode", ("Pekerjaan", "Riwayat Perjalanan", "Faktor Risiko Lain")),
    "Demam Dengue": DiseaseProfile("Demam Dengue", "vector-borne", False, "vector_environment", ("Faktor Risiko Lain", "Pekerjaan", "Riwayat Perjalanan")),
    "ISPA Berat": DiseaseProfile("ISPA Berat", "respiratory", True, "transmission_wave", ("Pekerjaan", "Riwayat Perjalanan", "Status Komorbid")),
    "Diare Akut": DiseaseProfile("Diare Akut", "fecal-oral/environmental", None, "exposure_episode", ("Faktor Risiko Lain", "Riwayat Perjalanan")),
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

def validate_scope_for_special_analysis(df: pd.DataFrame, disease: str | None, geographic_level: str | None, minimum_cases: int = 10, minimum_days: int = 14) -> AnalysisEligibility:
    reasons, warnings = [], []
    if resolve_disease_profile(disease).name == "Semua Penyakit":
        reasons.append("Analisis epidemiologi memerlukan satu penyakit. Pilih penyakit tertentu; Semua Penyakit tetap bersifat deskriptif.")
    if len(df) < minimum_cases:
        reasons.append(f"Data kasus belum mencukupi untuk analisis statistik utama: {len(df)} < {minimum_cases}.")
    if "Tanggal Sakit" not in df.columns:
        reasons.append("Tanggal Sakit tidak tersedia.")
    else:
        dates = pd.to_datetime(df["Tanggal Sakit"], errors="coerce").dropna()
        if dates.empty:
            reasons.append("Tidak ada Tanggal Sakit yang valid.")
        elif (dates.max() - dates.min()).days + 1 < minimum_days:
            warnings.append(f"Rentang waktu observasi < {minimum_days} hari; modul temporal tertentu mungkin tidak tersedia.")
    return AnalysisEligibility(not reasons, reasons, warnings)

def classify_temporal_pattern_for_disease(profile: DiseaseProfile, detected_peaks: int) -> dict[str, Any]:
    if profile.temporal_interpretation == "transmission_wave":
        return {"pattern_type": "POTENTIAL_TRANSMISSION_WAVE" if detected_peaks >= 2 else "TEMPORAL_PEAK", "interpretation": "Pola temporal dapat konsisten dengan kemungkinan gelombang transmisi, tetapi peak detector saja tidak membuktikan transmisi.", "transmission_claim_allowed": False}
    if detected_peaks:
        return {"pattern_type": "TEMPORAL_EXPOSURE_EPISODES", "interpretation": "Terdapat puncak temporal; untuk penyakit ini puncak tidak boleh otomatis ditafsirkan sebagai transmisi antar-manusia.", "transmission_claim_allowed": False}
    return {"pattern_type": "NO_CLEAR_TEMPORAL_PEAK", "interpretation": "Tidak ditemukan puncak temporal yang cukup kuat pada scope ini.", "transmission_claim_allowed": False}
