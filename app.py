import warnings

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.analytics import (
    REQUIRED_COLUMNS,
    generate_excel_template,
    generate_data_simulasi,
    get_wib_time,
)
from core.engine import SIHISIntelligenceEngine
from core.scope import QueryScope

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SI-HIS Intelligence", page_icon="🧠", layout="wide")

st.markdown(
    """<div style="background:#f0e68c;padding:18px 22px;border-radius:12px;border:2px solid #d4c886;">
    <h1 style="margin:0;color:#1e3a8a;font-size:1.7rem;">SI-HIS — Smart Integrated Health Intelligence System</h1>
    <p style="margin:6px 0 0;color:#475569;font-weight:600;">Early Detection, Smarter Intervention</p>
    </div>""",
    unsafe_allow_html=True,
)
st.caption(f"🕒 Waktu Sistem: {get_wib_time()['full']}")
engine = SIHISIntelligenceEngine()


def render_value(value, title=None):
    """Render engine output without exposing Streamlit DeltaGenerator objects."""
    if title:
        st.markdown(f"#### {title}")
    if isinstance(value, pd.DataFrame):
        if value.empty:
            st.info("Belum ada data untuk ditampilkan.")
        else:
            st.dataframe(value, use_container_width=True, hide_index=True)
    elif isinstance(value, dict):
        if not value:
            st.info("Belum ada hasil.")
            return
        for key, item in value.items():
            label = str(key).replace("_", " ").title()
            if isinstance(item, (pd.DataFrame, dict, list)):
                render_value(item, label)
            else:
                st.write(f"**{label}:** {item}")
    elif isinstance(value, list):
        if not value:
            st.info("Belum ada hasil.")
        elif all(isinstance(x, dict) for x in value):
            render_value(pd.DataFrame(value))
        else:
            for item in value:
                st.write(item)
    elif value is None:
        st.info("Belum tersedia.")
    else:
        st.write(value)


def disease_values_from(df):
    values = set()
    for col in ["Diagnosis Konfirm", "Diagnosis Probabel", "Diagnosis Suspek"]:
        if col in df.columns:
            values.update(
                str(x).strip()
                for x in df[col].dropna().unique()
                if str(x).strip().lower()
                not in {"", "nan", "bukan", "none", "tidak ada", "-"}
            )
    return sorted(values)


# DATA SOURCE
st.sidebar.header("⚙️ Panel Kontrol & Filter")
source = st.sidebar.radio(
    "Sumber Data",
    ["Gunakan Data Simulasi (AI-Ready)", "Upload File Excel/CSV Custom"],
)

if source.startswith("Gunakan"):
    df_raw = generate_data_simulasi().copy()
    st.sidebar.success(f"✅ {len(df_raw):,} data dimuat.")
else:
    uploaded = st.sidebar.file_uploader("Upload File Kasus", type=["xlsx", "csv"])
    if uploaded is None:
        st.info("Upload file kasus untuk memulai.")
        st.stop()
    try:
        df_raw = (
            pd.read_csv(uploaded)
            if uploaded.name.lower().endswith(".csv")
            else pd.read_excel(uploaded)
        )
    except Exception as exc:
        st.error(f"Error membaca file: {exc}")
        st.stop()
    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        st.error(f"Kolom wajib kurang: {missing}")
        st.stop()
    st.sidebar.success(f"✅ {len(df_raw):,} baris data dimuat.")

try:
    st.sidebar.download_button(
        "📥 Download Template Excel Standard",
        generate_excel_template(),
        "Template_Data_Surveilans.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
except Exception:
    pass

# THREE PRIMARY FILTERS — ALWAYS PRESENT
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Filter Analisis Epidemiologi")

provinces = (
    sorted(df_raw["Provinsi"].dropna().astype(str).unique())
    if "Provinsi" in df_raw
    else []
)
sel_prov = st.sidebar.selectbox("1. Provinsi", ["Semua Provinsi"] + provinces)

dfp = (
    df_raw
    if sel_prov == "Semua Provinsi"
    else df_raw[df_raw["Provinsi"].astype(str).eq(sel_prov)]
)

districts = (
    sorted(dfp["Kabupaten"].dropna().astype(str).unique())
    if "Kabupaten" in dfp
    else []
)
sel_kab = st.sidebar.selectbox(
    "2. Kabupaten/Kota", ["Semua Kabupaten/Kota"] + districts
)

dfk = (
    dfp
    if sel_kab == "Semua Kabupaten/Kota"
    else dfp[dfp["Kabupaten"].astype(str).eq(sel_kab)]
)

diseases = disease_values_from(df_raw)
sel_disease = st.sidebar.selectbox(
    "3. Diagnosis Penyakit", ["Semua Penyakit"] + diseases
)

# Optional geographic drill-down. It narrows the selected geographic scope.
with st.sidebar.expander("Drill-down wilayah (opsional)"):
    kecs = (
        sorted(dfk["Kecamatan"].dropna().astype(str).unique())
        if "Kecamatan" in dfk
        else []
    )
    sel_kec = st.selectbox("Kecamatan", ["Semua Kecamatan"] + kecs)
    dbase = (
        dfk
        if sel_kec == "Semua Kecamatan"
        else dfk[dfk["Kecamatan"].astype(str).eq(sel_kec)]
    )

    villages = (
        sorted(dbase["Desa/Kelurahan"].dropna().astype(str).unique())
        if "Desa/Kelurahan" in dbase
        else []
    )
    sel_desa = st.selectbox("Desa/Kelurahan", ["Semua Desa/Kelurahan"] + villages)
    vbase = (
        dbase
        if sel_desa == "Semua Desa/Kelurahan"
        else dbase[dbase["Desa/Kelurahan"].astype(str).eq(sel_desa)]
    )

    pusk = (
        sorted(vbase["Puskesmas"].dropna().astype(str).unique())
        if "Puskesmas" in vbase
        else []
    )
    sel_pusk = st.selectbox("Puskesmas", ["Semua Puskesmas"] + pusk)

include_ml = st.sidebar.checkbox("Aktifkan ML layer", False)

scope = QueryScope(
    province=None if sel_prov == "Semua Provinsi" else sel_prov,
    district=None if sel_kab == "Semua Kabupaten/Kota" else sel_kab,
    kecamatan=None if sel_kec == "Semua Kecamatan" else sel_kec,
    village=None if sel_desa == "Semua Desa/Kelurahan" else sel_desa,
    puskesmas=None if sel_pusk == "Semua Puskesmas" else sel_pusk,
    disease=None if sel_disease == "Semua Penyakit" else sel_disease,
    period_days=3650,
)

# -----------------------------------------------------------------------------
# ALL DISEASES = DESCRIPTIVE ONLY
# -----------------------------------------------------------------------------
if sel_disease == "Semua Penyakit":
    # Use the geographic selection, but never calculate a combined CFR.
    result = engine.descriptive(dfk)
    geographic_label = (
        sel_kab
        if sel_kab != "Semua Kabupaten/Kota"
        else (sel_prov if sel_prov != "Semua Provinsi" else "Indonesia")
    )

    st.markdown(f"## 📊 Analisis Deskriptif — {geographic_label}")
    st.caption(
        "Semua Penyakit → analisis deskriptif. Jumlah kematian merupakan gabungan seluruh penyakit "
        "dan tidak disebut CFR."
    )

    overview = result["overview"]
    col1, col2 = st.columns(2)
    col1.metric("Total Kunjungan Pasien", f"{overview['total_cases']:,}")
    col2.metric("Kasus Meninggal", f"{overview['deaths']:,}")

    st.markdown("### 🏆 10 Besar Penyakit")
    st.dataframe(
        result["top10_diseases"],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🦠 Distribusi Penyakit")
    st.dataframe(result["disease_distribution"], use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 👤 Distribusi Jenis Kelamin")
        st.dataframe(result["sex_distribution"], use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### 🎂 Distribusi Kelompok Umur")
        st.dataframe(result["age_distribution"], use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🗺️ Distribusi Provinsi")
        st.dataframe(result["province_distribution"], use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### 🏘️ Distribusi Kabupaten/Kota")
        st.dataframe(result["district_distribution"].head(50), use_container_width=True, hide_index=True)

    st.info(
        "Mode deskriptif tidak menjalankan bivariat, multivariat, faktor risiko, EWS/KLB, "
        "DBSCAN, episentrum, kurva epidemik, forecast, atau prediksi penyakit."
    )

# -----------------------------------------------------------------------------
# SPECIFIC DISEASE = DESCRIPTIVE + EPIDEMIOLOGICAL ANALYSIS
# -----------------------------------------------------------------------------
else:
    result = engine.analyze(
        df_raw,
        scope=scope,
        include_ml=include_ml,
        mode="epidemiology",
    )

    geographic_label = (
        sel_kab
        if sel_kab != "Semua Kabupaten/Kota"
        else (sel_prov if sel_prov != "Semua Provinsi" else "Indonesia")
    )

    st.markdown(f"## 🧬 Analisis Epidemiologi — {sel_disease}")
    st.caption(f"Scope: **{sel_disease} — {geographic_label}** | TIME + PERSON + PLACE")

    if not result.get("eligible", False):
        st.warning("Analisis epidemiologi belum dapat dijalankan.")
        for reason in result.get("eligibility", {}).get("reasons", []):
            st.write(f"• {reason}")
        st.stop()

    overview = result["overview"]
    mortality = result.get("mortality")
    deaths = 0
    if isinstance(mortality, dict):
        deaths = mortality.get("deaths", mortality.get("meninggal", 0)) or 0
    elif isinstance(mortality, pd.DataFrame) and "Meninggal" in mortality.columns:
        deaths = int(mortality["Meninggal"].sum())

    total = int(overview["total_cases"])
    cfr = deaths / total * 100 if total else 0

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Total {sel_disease}", f"{total:,}")
    col2.metric(f"Meninggal {sel_disease}", f"{deaths:,}")
    col3.metric("CFR", f"{cfr:.2f}%")

    for warning in result.get("eligibility", {}).get("warnings", []):
        st.warning(warning)

    tabs = st.tabs(
        [
            "📊 TIME + PERSON + PLACE",
            "🧪 Faktor Risiko",
            "🚨 EWS/KLB",
            "🗺️ Spatial/DBSCAN",
            "📈 Kurva & Forecast",
            "👥 Vulnerable",
            "🧠 AI/ML",
        ]
    )

    with tabs[0]:
        render_value(result.get("place"), "PLACE")
        render_value(result.get("person"), "PERSON")

        st.markdown("#### TIME")
        time_df = result.get("time")
        if isinstance(time_df, pd.DataFrame) and not time_df.empty and "Tanggal Sakit" in time_df.columns:
            chart_df = time_df.copy()
            chart_df["Tanggal Sakit"] = pd.to_datetime(chart_df["Tanggal Sakit"], errors="coerce")
            chart_df = chart_df.dropna(subset=["Tanggal Sakit"]).set_index("Tanggal Sakit")
            if "Jumlah Kasus" in chart_df:
                st.line_chart(chart_df["Jumlah Kasus"])
            st.dataframe(time_df, use_container_width=True, hide_index=True)
        else:
            st.info("Data TIME belum tersedia.")

        st.markdown("#### MORTALITY / CFR")
        render_value(mortality)

    with tabs[1]:
        st.markdown("### Faktor Risiko — Bivariat & Multivariat")
        render_value(result.get("risk_factors"))
        st.caption(
            "OR/association menunjukkan asosiasi statistik pada data yang dianalisis; bukan bukti kausalitas."
        )

    with tabs[2]:
        st.markdown("### Early Warning / KLB")
        render_value(result.get("ews"))
        st.markdown("### Interpretasi Temporal Disease-Aware")
        temporal = result.get("temporal_interpretation", {})
        if isinstance(temporal, dict):
            st.info(temporal.get("interpretation", "Interpretasi temporal tersedia."))
            st.caption("Puncak matematis tidak otomatis membuktikan transmisi antar-manusia.")
        st.markdown("### Rₜ")
        render_value(result.get("rt"))

    with tabs[3]:
        st.markdown("### Spatial / DBSCAN")
        spatial = result.get("spatial")
        if isinstance(spatial, pd.DataFrame) and not spatial.empty:
            st.dataframe(spatial, use_container_width=True, hide_index=True)
            if {"Latitude", "Longitude"}.issubset(spatial.columns):
                geo = spatial.dropna(subset=["Latitude", "Longitude"]).copy()
                if not geo.empty:
                    m = folium.Map(
                        location=[float(geo["Latitude"].mean()), float(geo["Longitude"].mean())],
                        zoom_start=9,
                    )
                    for _, row in geo.head(500).iterrows():
                        folium.CircleMarker(
                            [float(row["Latitude"]), float(row["Longitude"])],
                            radius=4,
                            popup=f"{row.get('Desa/Kelurahan', '')} | Cluster {row.get('Cluster', '')}",
                        ).add_to(m)
                    st_folium(m, width=None, height=500)
        else:
            st.info("Data spasial belum tersedia.")

        st.markdown("### Episentrum")
        render_value(result.get("epicenters"))

    with tabs[4]:
        st.markdown("### Kurva Epidemik")
        render_value(result.get("epidemic_curve_classification"))

        time_df = result.get("time")
        if isinstance(time_df, pd.DataFrame) and not time_df.empty and "Tanggal Sakit" in time_df.columns:
            chart_df = time_df.copy()
            chart_df["Tanggal Sakit"] = pd.to_datetime(chart_df["Tanggal Sakit"], errors="coerce")
            chart_df = chart_df.dropna(subset=["Tanggal Sakit"]).set_index("Tanggal Sakit")
            if "Jumlah Kasus" in chart_df:
                st.line_chart(chart_df["Jumlah Kasus"])

        st.markdown("### Forecast")
        render_value(result.get("forecast"))

        waves = result.get("waves", [])
        if waves:
            st.markdown("### Deteksi Puncak/Gelombang Temporal")
            render_value(waves)

    with tabs[5]:
        st.markdown("### Vulnerable Population")
        render_value(result.get("vulnerable"))

    with tabs[6]:
        if include_ml:
            st.markdown("### ML Intelligence")
            render_value(result.get("ml"))
        else:
            st.info("ML layer tidak diaktifkan.")

st.caption(
    "SI-HIS Intelligence — Semua Penyakit = descriptive; penyakit spesifik = descriptive + epidemiological intelligence."
)
