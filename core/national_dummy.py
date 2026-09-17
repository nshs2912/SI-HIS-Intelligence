"""National synthetic epidemiology dataset for SI-HIS development/testing.

This module intentionally generates synthetic data only. It is designed to exercise
SI-HIS analytics across the full Indonesia administrative hierarchy without
embedding real patient information.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd


# Province names follow the current 38-province structure. Administrative counts
# are synthetic allocations used only to create 514 dummy kabupaten/kota records.
PROVINCES = [
    ("11", "Aceh", 23),
    ("12", "Sumatera Utara", 33),
    ("13", "Sumatera Barat", 19),
    ("14", "Riau", 12),
    ("15", "Jambi", 11),
    ("16", "Sumatera Selatan", 17),
    ("17", "Bengkulu", 10),
    ("18", "Lampung", 15),
    ("19", "Kepulauan Bangka Belitung", 7),
    ("21", "Kepulauan Riau", 7),
    ("31", "DKI Jakarta", 6),
    ("32", "Jawa Barat", 27),
    ("33", "Jawa Tengah", 35),
    ("34", "DI Yogyakarta", 5),
    ("35", "Jawa Timur", 38),
    ("36", "Banten", 8),
    ("51", "Bali", 9),
    ("52", "Nusa Tenggara Barat", 10),
    ("53", "Nusa Tenggara Timur", 22),
    ("61", "Kalimantan Barat", 14),
    ("62", "Kalimantan Tengah", 14),
    ("63", "Kalimantan Selatan", 13),
    ("64", "Kalimantan Timur", 10),
    ("65", "Kalimantan Utara", 5),
    ("71", "Sulawesi Utara", 15),
    ("72", "Sulawesi Tengah", 13),
    ("73", "Sulawesi Selatan", 24),
    ("74", "Sulawesi Tenggara", 17),
    ("75", "Gorontalo", 6),
    ("76", "Sulawesi Barat", 6),
    ("81", "Maluku", 11),
    ("82", "Maluku Utara", 10),
    ("91", "Papua", 9),
    ("92", "Papua Barat", 7),
    ("93", "Papua Selatan", 4),
    ("94", "Papua Tengah", 8),
    ("95", "Papua Pegunungan", 8),
    ("96", "Papua Barat Daya", 6),
]

DISEASES = [
    "ISPA",
    "Diare Akut",
    "Demam Dengue",
    "Leptospirosis",
    "TBC",
    "Pneumonia",
    "Campak",
    "Malaria",
    "Hipertensi",
    "Diabetes Melitus",
    "Hepatitis",
    "Tifoid",
    "Penyakit Kulit",
    "COVID-like Respiratory Illness",
    "DBD",
]

DISEASE_BASE_RATE = {
    "ISPA": 0.24,
    "Diare Akut": 0.11,
    "Demam Dengue": 0.09,
    "Leptospirosis": 0.035,
    "TBC": 0.075,
    "Pneumonia": 0.06,
    "Campak": 0.035,
    "Malaria": 0.055,
    "Hipertensi": 0.075,
    "Diabetes Melitus": 0.055,
    "Hepatitis": 0.025,
    "Tifoid": 0.045,
    "Penyakit Kulit": 0.06,
    "COVID-like Respiratory Illness": 0.065,
    "DBD": 0.085,
}

# Broad synthetic regional multipliers. These are deliberately not claims about
# real disease prevalence; they merely make the demo spatially non-uniform.
REGION_MULTIPLIER = {
    "Java": 1.55,
    "Sumatra": 1.15,
    "Kalimantan": 0.88,
    "Sulawesi": 0.82,
    "Bali-Nusa Tenggara": 0.78,
    "Maluku": 0.62,
    "Papua": 0.52,
}


def _island(province: str) -> str:
    if province in {"DKI Jakarta", "Jawa Barat", "Jawa Tengah", "DI Yogyakarta", "Jawa Timur", "Banten"}:
        return "Java"
    if province.startswith("Sumatera") or province in {"Aceh", "Kepulauan Bangka Belitung", "Kepulauan Riau"}:
        return "Sumatra"
    if province.startswith("Kalimantan"):
        return "Kalimantan"
    if province.startswith("Sulawesi") or province == "Gorontalo":
        return "Sulawesi"
    if province in {"Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur"}:
        return "Bali-Nusa Tenggara"
    if province in {"Maluku", "Maluku Utara"}:
        return "Maluku"
    return "Papua"


def _province_weight(province: str) -> float:
    # Population/service-volume proxy for synthetic load, not population data.
    return {
        "Jawa Barat": 1.85,
        "Jawa Timur": 1.55,
        "Jawa Tengah": 1.45,
        "DKI Jakarta": 1.30,
        "Banten": 1.05,
        "Sumatera Utara": 1.00,
        "Sulawesi Selatan": 0.78,
        "Sumatera Selatan": 0.74,
        "Lampung": 0.68,
        "Riau": 0.62,
        "Kalimantan Timur": 0.58,
        "Bali": 0.55,
    }.get(province, 0.38)


def build_national_regions() -> pd.DataFrame:
    """Create 38 provinces and exactly 514 synthetic kabupaten/kota records."""
    rows = []
    for p_idx, (pcode, province, district_count) in enumerate(PROVINCES, start=1):
        for d_idx in range(1, district_count + 1):
            # Stable synthetic administrative identifiers; no real address is implied.
            dcode = f"{pcode}.{d_idx:02d}"
            kind = "Kota" if d_idx <= max(1, round(district_count * 0.18)) else "Kabupaten"
            district = f"{kind} Dummy {p_idx:02d}-{d_idx:02d}"
            rows.append(
                {
                    "Province_Code": pcode,
                    "Provinsi": province,
                    "District_Code": dcode,
                    "Kabupaten": district,
                    "Is_Kota": kind == "Kota",
                    "Island_Group": _island(province),
                    "Province_Weight": _province_weight(province),
                }
            )
    out = pd.DataFrame(rows)
    assert len(out) == 514, f"Expected 514 dummy districts, got {len(out)}"
    return out


def _make_locations(regions: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic synthetic kecamatan, desa and puskesmas rows per district."""
    rows = []
    for _, r in regions.iterrows():
        # Keep the hierarchy large enough for drill-down without creating millions
        # of rows before cases are generated.
        n_kec = int(5 + (int(r["District_Code"].split(".")[-1]) % 8))
        for k in range(1, n_kec + 1):
            kec = f"Kecamatan Dummy {k:02d}"
            n_village = 4 + (k % 4)
            for v in range(1, n_village + 1):
                village = f"Desa/Kelurahan Dummy {k:02d}-{v:02d}"
                facility = f"Puskesmas Dummy {r['District_Code']}-{k:02d}-{v:02d}"
                # Synthetic coordinate envelope; intentionally approximate and not a
                # representation of an actual village/facility location.
                base_lat = -6.2 + (int(r["Province_Code"]) % 100) * 0.11 - (k * 0.012)
                base_lon = 106.8 + (int(r["Province_Code"]) % 100) * 0.17 + (v * 0.014)
                rows.append(
                    {
                        **r.to_dict(),
                        "Kecamatan": kec,
                        "Desa/Kelurahan": village,
                        "Puskesmas": facility,
                        "Latitude": round(base_lat, 5),
                        "Longitude": round(base_lon, 5),
                    }
                )
    return pd.DataFrame(rows)


def _disease_probability(disease: str, province: str, month: int, rng: np.random.Generator) -> float:
    p = DISEASE_BASE_RATE[disease]
    p *= REGION_MULTIPLIER[_island(province)]
    # Synthetic seasonality to ensure different temporal patterns.
    if disease in {"DBD", "Demam Dengue"}:
        p *= 1.0 + 0.55 * np.sin((month - 1) / 12 * 2 * np.pi)
    elif disease == "Leptospirosis":
        p *= 1.0 + 0.65 * max(0.0, np.sin((month - 2) / 12 * 2 * np.pi))
    elif disease in {"ISPA", "Pneumonia", "COVID-like Respiratory Illness"}:
        p *= 1.0 + 0.35 * np.cos((month - 1) / 12 * 2 * np.pi)
    elif disease == "Malaria" and _island(province) == "Papua":
        p *= 2.4
    elif disease == "Malaria" and _island(province) != "Papua":
        p *= 0.55
    # Small deterministic noise avoids identical rates across districts.
    return max(0.001, p * float(rng.uniform(0.78, 1.24)))


def _generate_cases(locations: pd.DataFrame, days: int, target_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tz = timezone(timedelta(hours=7))
    end = datetime.now(tz).date()
    start = end - timedelta(days=days - 1)

    # Sample facilities according to a synthetic service-volume weight.
    loc = locations.copy()
    loc["_weight"] = loc["Province_Weight"] * rng.lognormal(mean=0.0, sigma=0.55, size=len(loc))
    loc["_weight"] /= loc["_weight"].sum()
    chosen = rng.choice(len(loc), size=target_rows, replace=True, p=loc["_weight"].to_numpy())
    selected = loc.iloc[chosen].reset_index(drop=True)

    dates = [start + timedelta(days=int(x)) for x in rng.integers(0, days, size=target_rows)]
    diseases = []
    for _, row in selected.iterrows():
        weights = np.array([_disease_probability(d, row["Provinsi"], dates[len(diseases)].month, rng) for d in DISEASES])
        weights /= weights.sum()
        diseases.append(rng.choice(DISEASES, p=weights))

    age = rng.integers(1, 86, size=target_rows)
    sex = rng.choice(["Laki-laki", "Perempuan"], size=target_rows, p=[0.51, 0.49])
    occupation = rng.choice(
        ["Petani", "Nelayan", "PNS/ASN", "Wiraswasta", "Ibu Rumah Tangga", "Pelajar/Mahasiswa", "Buruh", "Pekerja Formal"],
        size=target_rows,
        p=[0.12, 0.03, 0.12, 0.18, 0.17, 0.16, 0.10, 0.12],
    )
    comorbidity = np.where(age >= 50, rng.choice(["Ada Komorbid", "Tidak Ada"], size=target_rows, p=[0.42, 0.58]), rng.choice(["Ada Komorbid", "Tidak Ada"], size=target_rows, p=[0.18, 0.82]))
    immunization = rng.choice(["Lengkap", "Tidak Lengkap", "Tidak Ada/Belum"], size=target_rows, p=[0.62, 0.25, 0.13])
    travel = rng.choice(["Ya", "Tidak"], size=target_rows, p=[0.18, 0.82])
    other_risk = rng.choice(["Tidak Ada", "Paparan Lingkungan", "Kontak Erat", "Air/Sanitasi", "Paparan Vektor"], size=target_rows, p=[0.45, 0.17, 0.15, 0.12, 0.11])

    diagnosis = np.array(diseases)
    confirm_prob = np.where(travel == "Ya", 0.62, 0.48)
    confirm_prob += np.where(comorbidity == "Ada Komorbid", 0.04, 0.0)
    confirmed = rng.random(target_rows) < np.clip(confirm_prob, 0.2, 0.85)
    suspect = np.where(confirmed, "Konfirm", "Suspek")

    severity = (
        0.012
        + np.where(age >= 60, 0.045, 0.0)
        + np.where(comorbidity == "Ada Komorbid", 0.035, 0.0)
        + np.where(np.isin(diagnosis, ["Malaria", "Pneumonia", "TBC"]), 0.025, 0.0)
        + np.where(np.isin(diagnosis, ["DBD", "Demam Dengue"]), 0.012, 0.0)
    )
    deaths = rng.random(target_rows) < np.clip(severity, 0.003, 0.18)
    status = np.where(deaths, "Meninggal", np.where(confirmed, "Sembuh/Rawat", "Dalam Pemantauan"))

    # Add outbreak pulses to a subset of districts/diseases. This is what makes
    # EWS, epidemic curves, Rt, wave detection and DBSCAN visibly useful.
    outbreak_mask = (
        selected["District_Code"].astype(str).str.endswith(("03", "07"), na=False).to_numpy()
        & np.isin(diagnosis, ["DBD", "Demam Dengue", "Leptospirosis"])
    )
    pulse = rng.random(target_rows) < np.where(outbreak_mask, 0.22, 0.015)
    outbreak_multiplier = np.where(pulse, 2.8, 1.0)

    out = pd.DataFrame(
        {
            "Nama": [f"National_Dummy_{i:07d}" for i in range(1, target_rows + 1)],
            "Umur": age,
            "Jenis Kelamin": sex,
            "Pekerjaan": occupation,
            "Status Imunisasi": immunization,
            "Status Komorbid": comorbidity,
            "Riwayat Perjalanan": travel,
            "Faktor Risiko Lain": other_risk,
            "Tanggal Sakit": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "Provinsi": selected["Provinsi"].to_numpy(),
            "Kabupaten": selected["Kabupaten"].to_numpy(),
            "Kecamatan": selected["Kecamatan"].to_numpy(),
            "Desa/Kelurahan": selected["Desa/Kelurahan"].to_numpy(),
            "Puskesmas": selected["Puskesmas"].to_numpy(),
            "Latitude": selected["Latitude"].to_numpy(),
            "Longitude": selected["Longitude"].to_numpy(),
            "Diagnosis Suspek": diagnosis,
            "Diagnosis Probabel": np.where(confirmed, diagnosis, "Bukan"),
            "Diagnosis Konfirm": np.where(confirmed, diagnosis, "Bukan"),
            "Is_Konfirm": confirmed.astype(int),
            "Is_Meninggal": deaths.astype(int),
            "Status Penderita": status,
            "Province_Code": selected["Province_Code"].to_numpy(),
            "District_Code": selected["District_Code"].to_numpy(),
            "Island_Group": selected["Island_Group"].to_numpy(),
            "Synthetic_Outbreak_Pulse": outbreak_multiplier.astype(float),
        }
    )
    return out


@lru_cache(maxsize=4)
def generate_national_dummy(days: int = 365, target_rows: int = 60000, seed: int = 20260917) -> pd.DataFrame:
    """Return a cached national synthetic patient-level dataset.

    Defaults to 38 provinces, exactly 514 synthetic kabupaten/kota, a 365-day
    observation window and 60,000 synthetic case records. Increase target_rows
    for stress testing after the dashboard/API is stable.
    """
    regions = build_national_regions()
    locations = _make_locations(regions)
    return _generate_cases(locations, int(days), int(target_rows), int(seed))


def national_metadata() -> dict:
    regions = build_national_regions()
    return {
        "dataset_type": "synthetic_national",
        "provinces": int(regions["Provinsi"].nunique()),
        "districts": int(regions[["Provinsi", "Kabupaten"]].drop_duplicates().shape[0]),
        "diseases": len(DISEASES),
        "default_days": 365,
        "default_cases": 60000,
        "source": "SI-HIS synthetic data generator",
        "real_patient_data": False,
    }
