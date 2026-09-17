"""Regulatory KLB early-warning engine for SI-HIS.

Legal basis: Permenkes No. 1 Tahun 2026 tentang Kejadian Luar Biasa,
Wabah, dan Krisis Kesehatan (status: berlaku). The regulation requires
KLB vigilance through early detection, epidemiological assessment and
early warning. The algorithm is not a legal KLB declaration tool.

Quantitative criterion families are versioned technical surveillance rules.
They must be validated against applicable current disease-specific guidance.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd

REGULATION = {
    "name": "Permenkes No. 1 Tahun 2026",
    "title": "Kejadian Luar Biasa, Wabah, dan Krisis Kesehatan",
    "status": "BERLAKU",
    "effective_date": "2026-01-21",
    "basis_role": "Deteksi dini, kajian epidemiologis, peringatan kewaspadaan dini KLB",
    "legal_note": "Hasil SI-HIS adalah sinyal kewaspadaan/kajian dan bukan penetapan status KLB otomatis.",
}

RULESET = {
    "id": "SIHIS-KLB-LEGACY-8-CRITERIA-V2",
    "version": "2.0",
    "description": "Versioned operational KLB surveillance rules. The legacy eight criterion families are retained, with an additional absolute-CFR severity signal so a very high case-fatality proportion is never hidden by low case volume. Absolute CFR signal is an investigation trigger, not by itself a legal declaration of KLB.",
    "absolute_cfr_signal_threshold_pct": 50.0,
}

@dataclass
class CriterionResult:
    code: str
    name: str
    triggered: bool
    value: Any = None
    threshold: Any = None
    evidence: str = ""
    data_sufficient: bool = True
    note: str = ""


def _date_series(df):
    if "Tanggal Sakit" not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df["Tanggal Sakit"], errors="coerce").dropna()


def _period_counts(df, freq="D"):
    dates = _date_series(df)
    if dates.empty:
        return pd.Series(dtype=float)
    return dates.dt.to_period(freq).value_counts().sort_index().astype(float)


def _death_series(df):
    death = pd.Series(0, index=df.index, dtype=int)
    if "Is_Meninggal" in df.columns:
        death = pd.to_numeric(df["Is_Meninggal"], errors="coerce").fillna(0).astype(int)
    if "Status Penderita" in df.columns:
        status = df["Status Penderita"].astype(str).str.strip().str.lower().eq("meninggal").astype(int)
        death = pd.Series(np.maximum(death, status), index=df.index)
    return death


def _cfr(df):
    if df.empty:
        return np.nan
    return float(_death_series(df).sum() / len(df) * 100)


def _criterion_1(df):
    cols = ["Penyakit_Baru", "New_Disease", "Emerging_Disease"]
    for c in cols:
        if c in df.columns:
            v = df[c].astype(str).str.strip().str.lower().isin({"1", "true", "ya", "yes", "baru"})
            if v.any():
                return CriterionResult("C1", "Penyakit baru/tidak pernah ada sebelumnya", True, int(v.sum()), ">0", f"{int(v.sum())} kasus memiliki penanda penyakit baru.")
            return CriterionResult("C1", "Penyakit baru/tidak pernah ada sebelumnya", False, 0, ">0", "Tidak ada penanda penyakit baru pada dataset.")
    return CriterionResult("C1", "Penyakit baru/tidak pernah ada sebelumnya", False, None, ">0", "Field pembanding penyakit baru belum tersedia.", False, "Tidak boleh menginfer penyakit baru tanpa baseline/reference yang sah.")


def _criterion_2(df):
    c = _period_counts(df, "D")
    if len(c) < 3:
        c = _period_counts(df, "W")
    if len(c) < 3:
        return CriterionResult("C2", "Peningkatan morbiditas 3 periode berturut-turut", False, None, "3 periode naik", "Belum tersedia ≥3 periode observasi.", False)
    last = c.iloc[-3:]
    triggered = bool(last.iloc[0] < last.iloc[1] < last.iloc[2])
    return CriterionResult("C2", "Peningkatan morbiditas 3 periode berturut-turut", triggered, last.tolist(), "p1 < p2 < p3", f"Tiga periode terakhir: {last.iloc[0]:.0f}, {last.iloc[1]:.0f}, {last.iloc[2]:.0f}.")


def _criterion_3(df):
    c = _period_counts(df, "D")
    if len(c) < 2:
        return CriterionResult("C3", "Peningkatan morbiditas ≥2× periode sebelumnya", False, None, "≥2×", "Belum tersedia dua periode pembanding.", False)
    ratio = float(c.iloc[-1] / c.iloc[-2]) if c.iloc[-2] else (np.inf if c.iloc[-1] > 0 else 1.0)
    return CriterionResult("C3", "Peningkatan morbiditas ≥2× periode sebelumnya", ratio >= 2, ratio, "≥2.00×", f"Rasio periode terakhir terhadap periode sebelumnya = {ratio:.2f}×.")


def _criterion_4(df):
    dates = _date_series(df)
    if dates.empty or dates.max() - dates.min() < pd.Timedelta(days=365):
        return CriterionResult("C4", "Kasus bulanan ≥2× rata-rata bulan yang sama tahun sebelumnya", False, None, "≥2×", "Baseline satu tahun sebelumnya belum tersedia.", False)
    current_month = dates.max().to_period("M")
    prior_same = dates[(dates.dt.month == current_month.month) & (dates.dt.year < current_month.year)]
    if prior_same.empty:
        return CriterionResult("C4", "Kasus bulanan ≥2× rata-rata bulan yang sama tahun sebelumnya", False, None, "≥2×", "Tidak ada baseline bulan yang sama tahun sebelumnya.", False)
    current_n = int(((dates.dt.year == current_month.year) & (dates.dt.month == current_month.month)).sum())
    baseline = float(prior_same.dt.to_period("M").value_counts().mean())
    ratio = current_n / baseline if baseline else (np.inf if current_n else 1.0)
    return CriterionResult("C4", "Kasus bulanan ≥2× rata-rata bulan yang sama tahun sebelumnya", ratio >= 2, ratio, "≥2.00×", f"Kasus bulan berjalan={current_n}; baseline={baseline:.2f}; rasio={ratio:.2f}×.")


def _criterion_5(df):
    dates = _date_series(df)
    if dates.empty or dates.max() - dates.min() < pd.Timedelta(days=365):
        return CriterionResult("C5", "Rata-rata morbiditas bulanan tahun berjalan ≥2× tahun sebelumnya", False, None, "≥2×", "Baseline satu tahun sebelumnya belum tersedia.", False)
    max_year = dates.max().year
    cur = dates[dates.dt.year == max_year].dt.to_period("M").value_counts().mean()
    prev = dates[dates.dt.year == max_year - 1].dt.to_period("M").value_counts().mean()
    ratio = cur / prev if prev else (np.inf if cur else 1.0)
    return CriterionResult("C5", "Rata-rata morbiditas bulanan tahun berjalan ≥2× tahun sebelumnya", ratio >= 2, ratio, "≥2.00×", f"Rata-rata bulanan tahun berjalan={cur:.2f}; tahun sebelumnya={prev:.2f}; rasio={ratio:.2f}×.")


def _criterion_6(df):
    dates = _date_series(df)
    if dates.empty or dates.max() - dates.min() < pd.Timedelta(days=365):
        return CriterionResult("C6", "CFR meningkat ≥50% dibanding periode sama sebelumnya", False, None, "≥50%", "Baseline satu tahun sebelumnya belum tersedia.", False)
    latest = dates.max().to_period("M")
    date_col = pd.to_datetime(df["Tanggal Sakit"], errors="coerce")
    cur_df = df[date_col.dt.to_period("M") == latest]
    prev_df = df[date_col.dt.to_period("M") == (latest - 12)]
    cur_cfr, prev_cfr = _cfr(cur_df), _cfr(prev_df)
    if np.isnan(prev_cfr):
        return CriterionResult("C6", "CFR meningkat ≥50% dibanding periode sama sebelumnya", False, None, "≥50%", "CFR baseline tidak tersedia.", False)
    increase = ((cur_cfr - prev_cfr) / prev_cfr * 100) if prev_cfr else (np.inf if cur_cfr > 0 else 0)
    return CriterionResult("C6", "CFR meningkat ≥50% dibanding periode sama sebelumnya", increase >= 50, increase, "≥50%", f"CFR berjalan={cur_cfr:.2f}%; baseline={prev_cfr:.2f}%; perubahan={increase:.2f}%.")


def _criterion_7(df):
    for col in ["Total_Kasus_Baru", "Total_New_Cases"]:
        if col in df.columns:
            numerator = len(df)
            denom = pd.to_numeric(df[col], errors="coerce").sum()
            if denom <= 0:
                break
            return CriterionResult("C7", "Proporsi penyakit pada kasus baru meningkat ≥2×", False, numerator / denom, "≥2× baseline", "Denominator tersedia, tetapi baseline periode pembanding belum disediakan.", False, "Memerlukan proporsi penyakit pada periode pembanding yang sah.")
    return CriterionResult("C7", "Proporsi penyakit pada kasus baru meningkat ≥2×", False, None, "≥2× baseline", "Denominator dan baseline proporsi kasus baru belum tersedia.", False)


def _criterion_8(df):
    exposure_cols = [c for c in ["Sumber_Makanan", "Food_Source", "Paparan_Makanan"] if c in df.columns]
    if not exposure_cols:
        return CriterionResult("C8", "≥2 kasus dengan gejala sama/setara terkait sumber makanan", False, None, ">=2", "Field sumber makanan/paparan belum tersedia.", False)
    col = exposure_cols[0]
    counts = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}).dropna().value_counts()
    if counts.empty:
        return CriterionResult("C8", "≥2 kasus dengan gejala sama/setara terkait sumber makanan", False, 0, ">=2", "Tidak ada sumber makanan yang dapat dianalisis.")
    top = counts.iloc[0]
    return CriterionResult("C8", "≥2 kasus dengan gejala sama/setara terkait sumber makanan", top >= 2, int(top), ">=2", f"Sumber makanan terbanyak memiliki {int(top)} kasus; hubungan epidemiologis masih memerlukan investigasi.")


def _criterion_9_absolute_cfr(df):
    """High CFR is an immediate severity/mortality signal, separate from C6 trend."""
    if df.empty:
        return CriterionResult("C9", "CFR absolut ≥50% — sinyal kematian bermakna", False, None, "≥50%", "Tidak ada kasus untuk menghitung CFR.", False)
    cases = len(df)
    deaths = int(_death_series(df).sum())
    cfr = _cfr(df)
    triggered = bool(cfr >= RULESET["absolute_cfr_signal_threshold_pct"])
    note = "CFR absolut sangat tinggi; lakukan verifikasi definisi kasus/outcome dan investigasi segera. Sinyal ini tidak otomatis menetapkan status KLB."
    if cases < 20:
        note += " Denominator kecil; interpretasi harus hati-hati karena satu kematian dapat mengubah CFR secara besar."
    evidence = f"{deaths} meninggal dari {cases} kasus; CFR={cfr:.2f}%."
    return CriterionResult("C9", "CFR absolut ≥50% — sinyal kematian bermakna", triggered, cfr, "≥50%", evidence, True, note)


def evaluate_klb(df: pd.DataFrame, disease: str | None = None, geographic_scope: str = "Indonesia") -> dict[str, Any]:
    """Evaluate KLB early-warning criterion families without declaring KLB."""
    work = df.copy()
    if disease and disease != "Semua Penyakit":
        masks = []
        for c in ["Diagnosis Konfirm", "Diagnosis Probabel", "Diagnosis Suspek"]:
            if c in work.columns:
                masks.append(work[c].astype(str).str.strip().eq(disease))
        if masks:
            mask = masks[0]
            for m in masks[1:]:
                mask |= m
            work = work.loc[mask].copy()
    results = [_criterion_1(work), _criterion_2(work), _criterion_3(work), _criterion_4(work), _criterion_5(work), _criterion_6(work), _criterion_7(work), _criterion_8(work), _criterion_9_absolute_cfr(work)]
    triggered = [r for r in results if r.triggered]
    insufficient = [r for r in results if not r.data_sufficient]
    if not triggered:
        status = "NO_SIGNAL" if not insufficient else "EARLY_WARNING_DATA_INCOMPLETE"
    elif any(r.code == "C9" for r in triggered):
        status = "INVESTIGATION_REQUIRED"
    elif len(triggered) == 1:
        status = "EARLY_WARNING"
    else:
        status = "POTENTIAL_KLB"
    return {
        "status": status,
        "criteria_triggered": len(triggered),
        "criteria_evaluated": len(results),
        "criteria": [asdict(r) for r in results],
        "regulatory_basis": REGULATION,
        "ruleset": RULESET,
        "scope": geographic_scope,
        "disease": disease or "Semua Penyakit",
        "authority_note": "SI-HIS tidak menetapkan status KLB secara otomatis. Sinyal kematian/CFR tinggi harus segera diverifikasi dan ditindaklanjuti dengan kajian/investigasi epidemiologis serta validasi otoritas sesuai kewenangan.",
    }
