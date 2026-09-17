import warnings

import folium
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.analytics import REQUIRED_COLUMNS, generate_excel_template, get_wib_time, generate_data_simulasi
from core.engine import SIHISIntelligenceEngine

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SI-HIS Intelligence", page_icon="🧠", layout="wide")

st.markdown("""
<div style="background:#f0e68c;padding:18px 22px;border-radius:12px;border:2px solid #d4c886;">
<h1 style="margin:0;color:#1e3a8a;font-size:1.7rem;">SI-HIS — Smart Integrated Health Intelligence System</h1>
<p style="margin:6px 0 0;color:#475569;font-weight:600;">Early Detection, Smarter Intervention</p>
</div>
""", unsafe_allow_html=True)
st.caption(f"🕒 Waktu Sistem: {get_wib_time()['full']}")

engine = SIHISIntelligenceEngine()

# -----------------------------------------------------------------------------
# DATA SOURCE
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Panel Kontrol")
mode_data = st.sidebar.radio(
    "Sumber Data",
    ["Gunakan Data Simulasi (AI-Ready)", "Upload File Excel/CSV Custom"],
    key="data_source_mode",
)

if mode_data == "Gunakan Data Simulasi (AI-Ready)":
    df_raw = generate_data_simulasi().copy()
    st.sidebar.success(f"✅ {len(df_raw):,} kasus simulasi dimuat.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload File Kasus (.xlsx / .csv)", type=["xlsx", "csv"])
    if uploaded_file is None:
        st.info("Silakan upload file kasus untuk memulai.")
        st.stop()
    try:
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"❌ Error membaca file: {exc}")
        st.stop()
    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        st.error(f"❌ Kolom wajib kurang: {missing}")
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

# -----------------------------------------------------------------------------
# TWO-MODE ANALYTICAL FLOW
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Mode Analisis")
analysis_mode = st.sidebar.radio(
    "Pilih mode",
    [
        "Deskriptif Nasional",
        "Analisis Epidemiologi Terarah",
    ],
    key="analysis_mode",
)

if analysis_mode == "Deskriptif Nasional":
    st.markdown("## 🇮🇩 SI-HIS — Analisis Deskriptif Nasional")
    st.caption(
        "Mode deskriptif tidak memerlukan filter penyakit atau wilayah. "
        "Sistem memaparkan distribusi kasus berdasarkan penyakit, wilayah, jenis kelamin, dan kelompok umur."
    )
    result = engine.analyze(df_raw, mode="descriptive")

    ov = result["overview"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Kasus", f"{ov['total_cases']:,}")
    c2.metric("Meninggal", f"{ov['deaths']:,}")
    c3.metric("CFR", f"{ov['cfr']:.2f}%")

    st.markdown("### 🏆 10 Besar Penyakit di Seluruh Indonesia")
    top10 = result["top10_diseases"].copy()
    if not top10.empty:
        # Keep the requested national table contract exactly.
        st.dataframe(
            top10[["NO.", "Nama Penyakit", "Jumlah Kasus", "CFR", "Kabupaten", "Provinsi"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Kabupaten pada tabel menunjukkan kabupaten/kota dengan kontribusi kasus terbanyak untuk penyakit tersebut.")
    else:
        st.info("Belum ada distribusi penyakit yang dapat ditampilkan.")

    st.markdown("### 🦠 Distribusi Semua Penyakit")
    st.dataframe(result["disease_distribution"], use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🗺️ Distribusi Provinsi")
        st.dataframe(result["province_distribution"], use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### 🏘️ Distribusi Kabupaten/Kota")
        st.dataframe(result["district_distribution"].head(50), use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 👤 Distribusi Jenis Kelamin")
        st.dataframe(result["sex_distribution"], use_container_width=True, hide_index=True)
    with col4:
        st.markdown("### 🎂 Distribusi Kelompok Umur")
        st.dataframe(result["age_distribution"], use_container_width=True, hide_index=True)

    st.info(
        "ℹ️ Mode ini bersifat deskriptif. Tidak menjalankan bivariat, multivariat, faktor risiko, "
        "episentrum, DBSCAN, kurva epidemik, forecast, atau analisis KLB khusus."
    )

else:
    # -------------------------------------------------------------------------
    # SCOPED EPIDEMIOLOGY FILTERS
    # -------------------------------------------------------------------------
    st.markdown("## 🧬 SI-HIS — Analisis Epidemiologi Terarah")
    st.caption("TIME + PERSON + PLACE adalah inti analisis epidemiologi.")

    provinces = sorted(df_raw.get("Provinsi", pd.Series(dtype=str)).dropna().astype(str).unique())
    sel_province = st.sidebar.selectbox("Provinsi", ["Pilih Provinsi"] + provinces, index=0)

    if sel_province != "Pilih Provinsi":
        df_province = df_raw[df_raw["Provinsi"].astype(str).eq(sel_province)]
    else:
        df_province = df_raw.iloc[0:0]

    districts = sorted(df_province.get("Kabupaten", pd.Series(dtype=str)).dropna().astype(str).unique())
    sel_district = st.sidebar.selectbox("Kabupaten/Kota — WAJIB", ["Pilih Kabupaten/Kota"] + districts, index=0)

    if sel_district != "Pilih Kabupaten/Kota":
        df_district = df_province[df_province["Kabupaten"].astype(str).eq(sel_district)]
    else:
        df_district = df_province.iloc[0:0]

    subdistricts = sorted(df_district.get("Kecamatan", pd.Series(dtype=str)).dropna().astype(str).unique())
    sel_kecamatan = st.sidebar.selectbox("Kecamatan — opsional", ["Semua Kecamatan"] + subdistricts)
    villages_base = df_district if sel_kecamatan == "Semua Kecamatan" else df_district[df_district["Kecamatan"].astype(str).eq(sel_kecamatan)]
    villages = sorted(villages_base.get("Desa/Kelurahan", pd.Series(dtype=str)).dropna().astype(str).unique())
    sel_village = st.sidebar.selectbox("Desa/Kelurahan — opsional", ["Semua Desa/Kelurahan"] + villages)

    if sel_village == "Semua Desa/Kelurahan":
        facilities_base = villages_base
    else:
        facilities_base = villages_base[villages_base["Desa/Kelurahan"].astype(str).eq(sel_village)]
    facilities = sorted(facilities_base.get("Puskesmas", pd.Series(dtype=str)).dropna().astype(str).unique())
    sel_puskesmas = st.sidebar.selectbox("Puskesmas — opsional", ["Semua Puskesmas"] + facilities)

    disease_values = set()
    for column in ["Diagnosis Konfirm", "Diagnosis Probabel", "Diagnosis Suspek"]:
        if column in df_raw.columns:
            disease_values.update(
                str(x).strip() for x in df_raw[column].dropna().unique()
                if str(x).strip().lower() not in {"", "nan", "bukan", "none"}
            )
    diseases = sorted(disease_values)
    sel_disease = st.sidebar.selectbox("Penyakit — WAJIB", ["Pilih Penyakit"] + diseases, index=0)
    period_days = st.sidebar.number_input("Periode observasi (hari)", min_value=14, max_value=365, value=14, step=1)
    include_ml = st.sidebar.checkbox("Aktifkan ML layer", value=False)

    # Hard gate: no special epidemiology until disease + minimum geographic unit are selected.
    if sel_disease == "Pilih Penyakit" or sel_district == "Pilih Kabupaten/Kota":
        st.warning(
            "🔒 Analisis epidemiologi khusus dikunci. Pilih **1 penyakit** dan minimal **1 Kabupaten/Kota** "
            "serta periode observasi ≥14 hari. Setelah scope lengkap, SI-HIS akan menjalankan TIME + PERSON + PLACE "
            "dan analisis lanjutan."
        )
        st.markdown("### Alur analisis")
        st.code(
            "Penyakit + Kabupaten/Kota + Periode\n"
            "↓\nTIME + PERSON + PLACE\n"
            "↓\nFaktor Risiko → Bivariat → Multivariat\n"
            "↓\nEWS/KLB → DBSCAN → Episentrum\n"
            "↓\nKurva Epidemik → Forecast → Vulnerable Population\n"
            "↓\nAI/ML → Intelligence & Recommendation"
        )
        st.stop()

    scope = {
        "province": None if sel_province == "Pilih Provinsi" else sel_province,
        "district": sel_district,
        "kecamatan": None if sel_kecamatan == "Semua Kecamatan" else sel_kecamatan,
        "village": None if sel_village == "Semua Desa/Kelurahan" else sel_village,
        "puskesmas": None if sel_puskesmas == "Semua Puskesmas" else sel_puskesmas,
        "disease": sel_disease,
        "period_days": int(period_days),
    }
    from core.scope import QueryScope
    result = engine.analyze(df_raw, scope=QueryScope(**scope), include_ml=include_ml, mode="epidemiology")

    if not result["eligible"]:
        st.error("🔒 Scope belum memenuhi syarat analisis epidemiologi.")
        for reason in result["eligibility"]["reasons"]:
            st.write(f"• {reason}")
        if result["eligibility"]["warnings"]:
            for warning in result["eligibility"]["warnings"]:
                st.warning(warning)
        st.stop()

    st.success(
        f"✅ Scope aktif: **{sel_disease}** | **{sel_district}** | **{period_days} hari**"
    )
    if result["eligibility"]["warnings"]:
        for warning in result["eligibility"]["warnings"]:
            st.warning(warning)

    overview = result["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kasus", f"{overview['total_cases']:,}")
    c2.metric("Kabupaten", f"{overview['districts']:,}")
    c3.metric("Kecamatan", f"{overview['subdistricts']:,}")
    c4.metric("Desa/Kelurahan", f"{overview['villages']:,}")

    tabs = st.tabs([
        "📊 TIME + PERSON + PLACE",
        "🧪 Faktor Risiko",
        "🚨 EWS / KLB",
        "🗺️ Spatial / DBSCAN",
        "📈 Kurva & Forecast",
        "👥 Vulnerable",
        "🧠 AI / ML",
    ])

    with tabs[0]:
        st.markdown("### TIME + PERSON + PLACE")
        st.markdown("#### PLACE")
        place = result.get("place")
        if isinstance(place, pd.DataFrame):
            st.dataframe(place, use_container_width=True, hide_index=True)
        else:
            st.write(place)

        st.markdown("#### PERSON")
        person = result.get("person")
        if isinstance(person, dict):
            for key, value in person.items():
                st.markdown(f"**{key}**")
                if isinstance(value, pd.DataFrame):
                    st.dataframe(value, use_container_width=True, hide_index=True)
                else:
                    st.write(value)
        elif isinstance(person, pd.DataFrame):
            st.dataframe(person, use_container_width=True, hide_index=True)
        else:
            st.write(person)

        st.markdown("#### TIME")
        time_df = result.get("time")
        if isinstance(time_df, pd.DataFrame) and not time_df.empty:
            st.line_chart(time_df.set_index("Tanggal Sakit")["Jumlah Kasus"])
            st.dataframe(time_df, use_container_width=True, hide_index=True)
        else:
            st.info("Kurva waktu belum tersedia.")

        st.markdown("#### MORTALITY / CFR")
        mortality = result.get("mortality")
        if isinstance(mortality, pd.DataFrame):
            st.dataframe(mortality, use_container_width=True, hide_index=True)
        else:
            st.write(mortality)

    with tabs[1]:
        st.markdown("### 🧪 Faktor Risiko — Bivariat & Multivariat")
        st.caption("Analisis ini hanya dijalankan setelah scope penyakit + Kabupaten/Kota + periode memenuhi syarat.")
        risk = result.get("risk_factors")
        if isinstance(risk, dict):
            for key, value in risk.items():
                st.markdown(f"#### {key}")
                if isinstance(value, pd.DataFrame):
                    st.dataframe(value, use_container_width=True, hide_index=True)
                else:
                    st.write(value)
        elif isinstance(risk, pd.DataFrame):
            st.dataframe(risk, use_container_width=True, hide_index=True)
        else:
            st.info("Hasil faktor risiko belum tersedia.")
        st.caption("OR/association tidak ditafsirkan sebagai hubungan kausal tanpa desain dan validasi epidemiologis yang sesuai.")

    with tabs[2]:
        st.markdown("### 🚨 Early Warning / KLB")
        ews = result.get("ews")
        if isinstance(ews, dict):
            st.json(ews)
        else:
            st.write(ews)
        st.markdown("### Disease-aware temporal interpretation")
        temporal = result.get("temporal_interpretation")
        if temporal:
            st.info(temporal.get("interpretation", "Interpretasi temporal tersedia."))
            st.caption("Puncak matematis tidak otomatis membuktikan transmisi antar-manusia.")
        st.markdown("### Effective Rt")
        st.write(result.get("rt"))

    with tabs[3]:
        st.markdown("### 🗺️ Spatial / DBSCAN & Episentrum")
        spatial = result.get("spatial")
        if isinstance(spatial, pd.DataFrame):
            st.dataframe(spatial, use_container_width=True, hide_index=True)
            geo = spatial.dropna(subset=["Latitude", "Longitude"]).copy() if {"Latitude", "Longitude"}.issubset(spatial.columns) else pd.DataFrame()
            if not geo.empty:
                center = [float(geo["Latitude"].mean()), float(geo["Longitude"].mean())]
                m = folium.Map(location=center, zoom_start=10)
                for _, row in geo.head(500).iterrows():
                    folium.CircleMarker(
                        [float(row["Latitude"]), float(row["Longitude"])],
                        radius=4,
                        popup=f"{row.get('Desa/Kelurahan', '')} | Cluster {row.get('Cluster', '')}",
                    ).add_to(m)
                st_folium(m, width=None, height=550)
        else:
            st.write(spatial)
        st.markdown("### Episentrum")
        epicenter = result.get("epicenters")
        if isinstance(epicenter, pd.DataFrame):
            st.dataframe(epicenter, use_container_width=True, hide_index=True)
        else:
            st.write(epicenter)

    with tabs[4]:
        st.markdown("### 📈 Kurva Epidemik & Forecast")
        curve = result.get("epidemic_curve_classification")
        if curve:
            st.write(curve)
        time_df = result.get("time")
        if isinstance(time_df, pd.DataFrame) and not time_df.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(time_df["Tanggal Sakit"], time_df["Jumlah Kasus"], marker="o", label="Kasus")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Jumlah Kasus")
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig)
        forecast = result.get("forecast")
        if isinstance(forecast, dict):
            st.json(forecast)
        else:
            st.write(forecast)
        waves = result.get("waves")
        if waves:
            st.markdown("### Pola Gelombang")
            st.dataframe(pd.DataFrame(waves), use_container_width=True, hide_index=True)

    with tabs[5]:
        st.markdown("### 👥 Vulnerable Population")
        vulnerable = result.get("vulnerable")
        if isinstance(vulnerable, list) and vulnerable:
            st.dataframe(pd.DataFrame(vulnerable), use_container_width=True, hide_index=True)
        elif isinstance(vulnerable, pd.DataFrame):
            st.dataframe(vulnerable, use_container_width=True, hide_index=True)
        else:
            st.info("Profil populasi rentan belum tersedia pada scope ini.")

    with tabs[6]:
        st.markdown("### 🧠 AI / ML Intelligence")
        if not include_ml:
            st.info("ML tidak dijalankan. Aktifkan **ML layer** pada sidebar jika ingin menambahkan modul prediktif.")
        else:
            ml = result.get("ml", {})
            st.json({k: v for k, v in ml.items() if not isinstance(v, pd.DataFrame)})
            st.warning(
                "Output AI/ML adalah decision-support. Model tidak menggantikan validasi epidemiologis, "
                "investigasi lapangan, keputusan klinis, atau kewenangan penetapan KLB/wabah."
            )

st.caption("SI-HIS Intelligence — canonical intelligence engine: descriptive mode + disease-scoped epidemiological intelligence.")
