"""Request scope and isolated filtering for SI-HIS Intelligence."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd


FIELD_MAP = {
    "province": "Provinsi",
    "district": "Kabupaten",
    "kecamatan": "Kecamatan",
    "village": "Desa/Kelurahan",
    "puskesmas": "Puskesmas",
    "disease": "Diagnosis Konfirm",
}


@dataclass(frozen=True)
class QueryScope:
    province: Optional[str] = None
    district: Optional[str] = None
    kecamatan: Optional[str] = None
    village: Optional[str] = None
    puskesmas: Optional[str] = None
    disease: Optional[str] = None
    period_days: int = 14

    def normalized(self) -> "QueryScope":
        values = {}
        for key, value in asdict(self).items():
            if key == "period_days":
                values[key] = max(1, int(value))
            else:
                text = None if value is None else str(value).strip()
                values[key] = None if text in (None, "", "Semua", "Semua Provinsi", "Semua Kabupaten/Kota") else text
        return QueryScope(**values)

    def to_dict(self) -> dict:
        return asdict(self.normalized())


def apply_scope(df: pd.DataFrame, scope: QueryScope) -> pd.DataFrame:
    """Return an isolated copy filtered by scope; never mutate ``df``."""
    work = df.copy(deep=True)
    clean = scope.normalized()
    for parameter, column in FIELD_MAP.items():
        value = getattr(clean, parameter)
        if value and column in work.columns:
            work = work[work[column].astype(str).str.strip().eq(value)].copy()
    return work


def scope_label(scope: QueryScope) -> str:
    clean = scope.normalized()
    for key in ("puskesmas", "village", "kecamatan", "district", "province"):
        value = getattr(clean, key)
        if value:
            return value
    return "Indonesia"
