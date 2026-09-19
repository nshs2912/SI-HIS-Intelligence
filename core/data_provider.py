"""Canonical SI-HIS data access and source registry.

Two runtime modes are supported:
1. DEMO / synthetic national data for presentations and development.
2. FACTUAL / operational data loaded through a canonical ingestion contract.

Operational connectors (SIMPUS, FKTP, hospital/RME, NutriMed Mobile,
clinical professionals, laboratory, pharmacy and supporting services) should
feed the same canonical schema. The provider deliberately keeps ingestion
separate from intelligence so IT teams can replace adapters without changing
the analytics engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import pandas as pd

from .national_dummy import generate_national_dummy

DATASET_TYPE = "synthetic_national"

SOURCE_CATALOG = {
    "dummy": {"label": "Data Dummy / Synthetic Nasional", "category": "demo", "status": "ready", "description": "Dataset sintetis untuk presentasi, pengembangan, dan pengujian."},
    "simpus": {"label": "SIMPUS / Sistem Informasi Puskesmas", "category": "factual", "status": "adapter_ready", "description": "Kunjungan, diagnosis, layanan, rujukan, dan data operasional Puskesmas."},
    "fktp": {"label": "FKTP / Puskesmas / Klinik", "category": "factual", "status": "adapter_ready", "description": "Data pelayanan primer dan jejaring FKTP."},
    "hospital": {"label": "Rumah Sakit / RME", "category": "factual", "status": "adapter_ready", "description": "Rawat jalan, IGD, rawat inap, diagnosis, tindakan, outcome dan rujukan."},
    "nutrimed_mobile": {"label": "NutriMed MyLab Mobile", "category": "factual", "status": "adapter_ready", "description": "Patient-generated health data, skrining, konsultasi, diet, aktivitas dan perjalanan kesehatan."},
    "clinical_network": {"label": "Jejaring Dokter & Tenaga Kesehatan", "category": "factual", "status": "adapter_ready", "description": "Dokter, dietisien, fisioterapis, radiologis dan tenaga kesehatan/penunjang lain."},
    "laboratory": {"label": "Laboratorium / LIS", "category": "factual", "status": "adapter_ready", "description": "Hasil pemeriksaan laboratorium, specimen dan diagnostic report."},
    "pharmacy": {"label": "Farmasi / Apotek", "category": "factual", "status": "adapter_ready", "description": "Resep, dispensing, medication administration dan data obat."},
    "supporting_services": {"label": "Instalasi Penunjang Medis", "category": "factual", "status": "adapter_ready", "description": "Radiologi, patologi, rehabilitasi dan layanan penunjang lain."},
}

CANONICAL_ALIASES = {
    "Nama": ["nama", "patient_name", "patient_name_local", "nama_pasien"],
    "Umur": ["umur", "age", "usia"],
    "Jenis Kelamin": ["jenis_kelamin", "gender", "sex"],
    "Tanggal Sakit": ["tanggal_sakit", "tanggal_kunjungan", "encounter_date", "date"],
    "Provinsi": ["provinsi", "province"],
    "Kabupaten": ["kabupaten", "kabupaten_kota", "district"],
    "Kecamatan": ["kecamatan", "subdistrict"],
    "Desa/Kelurahan": ["desa", "kelurahan", "village"],
    "Puskesmas": ["puskesmas", "faskes", "facility_name"],
    "Diagnosis Konfirm": ["diagnosis_konfirm", "diagnosis", "icd10", "condition"],
    "Is_Konfirm": ["is_konfirm", "confirmed", "confirmed_case"],
    "Is_Meninggal": ["is_meninggal", "death", "deceased"],
    "Status Penderita": ["status_penderita", "patient_status", "status"],
    "Latitude": ["latitude", "lat"],
    "Longitude": ["longitude", "lon", "lng"],
}

@dataclass(frozen=True)
class SourceDescriptor:
    source_id: str
    label: str
    category: str
    status: str
    description: str

def get_source_catalog() -> list[dict[str, Any]]:
    return [{"source_id": source_id, **dict(metadata)} for source_id, metadata in SOURCE_CATALOG.items()]

def get_source_descriptor(source_id: str) -> SourceDescriptor:
    if source_id not in SOURCE_CATALOG:
        raise ValueError(f"Unknown SI-HIS data source: {source_id}")
    return SourceDescriptor(source_id=source_id, **SOURCE_CATALOG[source_id])

def _normalize_column_name(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")

def canonicalize_factual_data(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy(deep=True)
    normalized = {_normalize_column_name(c): c for c in work.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        if canonical in work.columns:
            continue
        for alias in [_normalize_column_name(canonical), *aliases]:
            source_col = normalized.get(_normalize_column_name(alias))
            if source_col is not None:
                rename[source_col] = canonical
                break
    return work.rename(columns=rename)

def validate_case_schema(df: pd.DataFrame, required: Optional[list[str]] = None) -> list[str]:
    if required is None:
        required = [
            "Nama", "Umur", "Jenis Kelamin", "Pekerjaan", "Status Imunisasi",
            "Status Komorbid", "Riwayat Perjalanan", "Faktor Risiko Lain",
            "Tanggal Sakit", "Provinsi", "Kabupaten", "Kecamatan",
            "Desa/Kelurahan", "Puskesmas", "Latitude", "Longitude",
            "Diagnosis Konfirm", "Is_Konfirm", "Is_Meninggal", "Status Penderita",
        ]
    return [column for column in required if column not in df.columns]

def load_factual_file(uploaded_file: Any) -> pd.DataFrame:
    name = str(getattr(uploaded_file, "name", "")).lower()
    payload = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
    if name.endswith(".csv"):
        df = pd.read_csv(BytesIO(payload) if isinstance(payload, (bytes, bytearray)) else payload)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(BytesIO(payload) if isinstance(payload, (bytes, bytearray)) else payload)
    else:
        raise ValueError("Format data faktual harus CSV atau Excel.")
    return canonicalize_factual_data(df)

def load_source(source_id: str, *, factual_file: Any = None, days: int = 365, target_rows: int = 60000, seed: int = 20260917) -> tuple[pd.DataFrame, dict[str, Any]]:
    descriptor = get_source_descriptor(source_id)
    if source_id == "dummy":
        df = generate_national_dummy(days=days, target_rows=target_rows, seed=seed)
        return df.copy(deep=True), {
            "source_id": source_id, "source_type": DATASET_TYPE, "source_mode": "DEMO",
            "integration_status": descriptor.status, "record_count": len(df), "description": descriptor.description,
        }
    if factual_file is None:
        raise ValueError(f"Sumber faktual '{descriptor.label}' membutuhkan file/data adapter.")
    df = load_factual_file(factual_file)
    missing = validate_case_schema(df)
    return df.copy(deep=True), {
        "source_id": source_id, "source_type": "factual", "source_mode": "FACTUAL",
        "integration_status": descriptor.status, "record_count": len(df),
        "description": descriptor.description, "missing_canonical_columns": missing,
    }

def get_cases(*, days: int = 365, target_rows: int = 60000, seed: int = 20260917, source_id: str = "dummy", factual_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if source_id == "dummy":
        df = generate_national_dummy(days=days, target_rows=target_rows, seed=seed)
    else:
        if factual_df is None:
            raise ValueError("factual_df wajib diisi untuk source_id selain dummy.")
        df = canonicalize_factual_data(factual_df)
    return df.copy(deep=True)

def get_metadata() -> dict:
    from .national_dummy import national_metadata
    return dict(national_metadata())

def get_cases_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)
