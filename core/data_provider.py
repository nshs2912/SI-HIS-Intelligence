"""Canonical SI-HIS data access layer.

The application and API must obtain demo/production data through this module.
It deliberately returns copies so consumers cannot mutate the canonical source.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .national_dummy import generate_national_dummy


DATASET_TYPE = "synthetic_national"


def get_cases(*, days: int = 365, target_rows: int = 60_000, seed: int = 20260917) -> pd.DataFrame:
    """Return the canonical SI-HIS case dataset for the requested demo period."""
    df = generate_national_dummy(days=days, target_rows=target_rows, seed=seed)
    return df.copy(deep=True)


def get_metadata() -> dict:
    """Return metadata without exposing mutable dataframe state."""
    from .national_dummy import national_metadata
    return dict(national_metadata())


def validate_case_schema(df: pd.DataFrame, required: Optional[list[str]] = None) -> list[str]:
    """Return missing columns; no mutation is performed."""
    if required is None:
        required = [
            "Nama", "Umur", "Jenis Kelamin", "Pekerjaan", "Status Imunisasi",
            "Status Komorbid", "Riwayat Perjalanan", "Faktor Risiko Lain",
            "Tanggal Sakit", "Provinsi", "Kabupaten", "Kecamatan",
            "Desa/Kelurahan", "Puskesmas", "Latitude", "Longitude",
            "Diagnosis Konfirm", "Is_Konfirm", "Is_Meninggal", "Status Penderita",
        ]
    return [column for column in required if column not in df.columns]


def get_cases_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Create an isolated query/analysis dataframe from any canonical frame."""
    return df.copy(deep=True)
