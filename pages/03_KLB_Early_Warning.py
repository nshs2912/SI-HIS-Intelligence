"""Regulatory KLB / outbreak early-warning view."""
import pandas as pd
import streamlit as st

from core.analytics import generate_data_simulasi
from core.engine import SIHISIntelligenceEngine
from core.scope import QueryScope
from core.surveillance import evaluate_klb

st.set_page_config(page_title="SI-HIS — KLB Early Warning", page_icon="🚨", layout="wide")
st.title("🚨 SI-HIS — Kewaspadaan Dini KLB")
st.caption("Regulatory Early Warning • TIME + PERSON + PLACE")

engine = SIHISIntelligenceEngine()
df = generate_data_simulasi().copy()

diseases = set()
for col in ["Diagnosis Konfirm", "Diagnosis Probabel", "Diagnosis Suspek"]:
    if col in df:
        diseases.update(str(x).strip() for x in df[col].dropna().unique() if str(x).strip().lower() not in {"", "nan", "none", "bukan", "tidak ada", "-"})

disease = st.selectbox("Penyakit", ["Semua Penyakit"] + sorted(diseases))
province = st.selectbox("Wilayah", ["Indonesia"] + sorted(df["Provinsi"].dropna().astype(str).unique()) if "Provinsi" in df else ["Indonesia"])

st.info(
    "**Dasar regulasi:** Permenkes No. 1 Tahun 2026 tentang Kejadian Luar Biasa, "
    "Wabah, dan Krisis Kesehatan — status **BERLAKU**. SI-HIS menggunakan regulasi "
    "ini sebagai dasar kerangka kewaspadaan dini: deteksi dini, kajian epidemiologis, "
    "peringatan kewaspadaan dini, serta peningkatan kewaspadaan dan kesiapsiagaan."
)
st.warning(
    "**Catatan kewenangan:** hasil di bawah adalah sinyal kewaspadaan/kajian untuk "
    "mendukung investigasi epidemiologis. SI-HIS **tidak menetapkan status KLB secara otomatis**."
)

work = df.copy()
if province != "Indonesia" and "Provinsi" in work:
    work = work.loc[work["Provinsi"].astype(str).eq(province)].copy()

klb = evaluate_klb(work, disease=None if disease == "Semua Penyakit" else disease, geographic_scope=province)
status = klb["status"]
status_label = {
    "NO_SIGNAL": "🟢 Tidak ditemukan sinyal berdasarkan data tersedia",
    "EARLY_WARNING": "🟡 Sinyal kewaspadaan dini",
    "POTENTIAL_KLB": "🟠 Potensi KLB — kajian/investigasi diperlukan",
    "EARLY_WARNING_DATA_INCOMPLETE": "🟡 Data belum cukup — kewaspadaan dipertahankan",
}.get(status, status)

st.subheader(status_label)
a, b, c = st.columns(3)
a.metric("Kriteria terpicu", klb["criteria_triggered"])
b.metric("Kriteria dievaluasi", klb["criteria_evaluated"])
c.metric("Scope", klb["scope"])

st.markdown("### Dasar Regulasi & Ruleset")
reg = klb["regulatory_basis"]
rules = klb["ruleset"]
st.write(f"**Regulasi:** {reg['name']} — {reg['title']}")
st.write(f"**Status:** {reg['status']} • **Tanggal berlaku:** {reg['effective_date']}")
st.write(f"**Ruleset analitik:** `{rules['id']}` • versi `{rules['version']}`")
st.caption(rules["description"])

st.markdown("### Evaluasi Kriteria KLB")
rows = []
for item in klb["criteria"]:
    rows.append({
        "Kode": item["code"],
        "Kriteria": item["name"],
        "Status": "TERPICU" if item["triggered"] else "Tidak terpicu",
        "Data cukup": "Ya" if item["data_sufficient"] else "Tidak",
        "Nilai": item["value"],
        "Ambang": item["threshold"],
        "Bukti/Data": item["evidence"],
        "Catatan": item["note"],
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("### Interpretasi TIME + PERSON + PLACE")
if disease != "Semua Penyakit":
    result = engine.analyze(work, scope=QueryScope(disease=disease, period_days=3650), mode="epidemiology")
    tri = result.get("trias_summary", {})
    if tri.get("narrative"):
        st.info(tri["narrative"])
else:
    st.info("Pilih penyakit tertentu untuk interpretasi epidemiologis disease-specific TIME + PERSON + PLACE.")

st.markdown("### Tindak Lanjut")
st.write(
    "Sinyal yang terpicu perlu ditelusuri dengan definisi kasus, kelengkapan dan kualitas "
    "pelaporan, baseline pembanding, distribusi TIME + PERSON + PLACE, faktor risiko, "
    "serta investigasi lapangan sebelum keputusan status KLB."
)
st.caption("SI-HIS Intelligence • Regulatory KLB Early Warning • Human/Authority Validation Required")
