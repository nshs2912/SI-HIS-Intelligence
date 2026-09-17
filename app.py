import io
import math
import warnings
from datetime import datetime, timedelta, timezone

import statsmodels.api as sm
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components

from core.analytics import (
    REQUIRED_COLUMNS, DEPENDENT_VARS, INDEPENDENT_VARS, KLB_THRESHOLD,
    get_wib_time, hitung_risk_stratification, identifikasi_vulnerable_profile,
    hitung_early_warning_score, deteksi_bentuk_kurva, prediksi_kurva_holt_winters,
    hitung_effective_rt, deteksi_gelombang, hitung_jarak_km, get_geolocator,
    lengkapi_koordinat_otomatis, generate_data_simulasi, generate_excel_template,
    hitung_bivariat_lengkap
)
from core.ml_engine import (
    train_case_severity, predict_case_severity, train_klb_prediction, predict_klb,
    train_spatial_outbreak, predict_spatial_outbreak, train_vulnerable_population,
    predict_vulnerable_population, robust_forecast
)

warnings.filterwarnings('ignore')
st.set_page_config(page_title='SI-HIS Intelligence', page_icon='🧠', layout='wide')

# ================================================================
# HEADER
# ================================================================
components.html("""
<div style="font-family:Arial,sans-serif;background:#f0e68c;padding:18px 22px;border-radius:12px;border:2px solid #d4c886;">
<h1 style="margin:0;color:#1e3a8a;font-size:1.7rem;">SI-HIS — Smart Integrated Health Intelligence System</h1>
<p style="margin:6px 0 0;color:#475569;font-weight:600;">Early Detection, Smarter Intervention</p>
</div>
""", height=115)

st.caption(f"🕒 Waktu Sistem: {get_wib_time()['full']}")

# ================================================================
# SIDEBAR — RESTORED FROM ORIGINAL APPLICATION
# ================================================================
st.sidebar.header("⚙️ Panel Kontrol & Filter")
mode_data = st.sidebar.radio(
    "Pilih Sumber Data:",
    ["Gunakan Data Simulasi (AI-Ready)", "Upload File Excel/CSV Custom"],
    key="data_source_mode"
)

if mode_data == "Gunakan Data Simulasi (AI-Ready)":
    df_raw = generate_data_simulasi().copy()
    st.sidebar.success(f"✅ Data simulasi {len(df_raw):,} kasus dimuat.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload File Kasus (.xlsx / .csv)", type=["xlsx", "csv"]
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
            if missing_cols:
                st.sidebar.error(f"❌ Kolom wajib kurang: {missing_cols}")
                st.stop()
            st.sidebar.success(f"✅ Berhasil memuat {len(df_raw):,} baris data.")
        except Exception as e:
            st.sidebar.error(f"❌ Error membaca file: {e}")
            st.stop()
    else:
        st.sidebar.info("👈 Silakan upload file untuk memulai.")
        st.stop()

st.sidebar.markdown("---")
try:
    st.sidebar.download_button(
        label="📥 Download Template Excel Standard",
        data=generate_excel_template(),
        file_name="Template_Data_Surveilans.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
except Exception:
    pass

st.sidebar.markdown("""
<div style="background:linear-gradient(135deg,#1e3a8a 0%,#0f4c81 100%);padding:16px;border-radius:10px;border:2px solid #fbbf24;margin-top:12px;margin-bottom:12px;">
<p style="color:#fbbf24;font-weight:700;margin:0 0 8px;">🚀 Enterprise Edition</p>
<p style="color:#e0f2fe;font-size:.8rem;line-height:1.5;margin:0 0 8px;">Analisis berkelanjutan untuk institusi/organisasi, big data, multi-user, integrasi dan keamanan data.</p>
<p style="color:#e0f2fe;font-size:.78rem;line-height:1.7;margin:0;">✅ Penyimpanan permanen<br>✅ Multi-user & kontrol akses<br>✅ Backup otomatis<br>✅ Integrasi big data<br>✅ Proteksi keamanan data</p>
</div>
""", unsafe_allow_html=True)

# ================================================================
# FILTER WILAYAH & DIAGNOSIS
# ================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Filter Wilayah & Diagnosis")

list_prov = sorted(df_raw['Provinsi'].dropna().astype(str).unique().tolist())
sel_prov = st.sidebar.multiselect("Pilih Provinsi:", list_prov, default=list_prov)
df_filtered_prov = df_raw[df_raw['Provinsi'].astype(str).isin(sel_prov)] if sel_prov else df_raw.iloc[0:0]

list_kab = sorted(df_filtered_prov['Kabupaten'].dropna().astype(str).unique().tolist())
sel_kab = st.sidebar.multiselect("Pilih Kabupaten:", list_kab, default=list_kab)

sel_tingkat_diag = st.sidebar.multiselect(
    "Pilih Tingkat Diagnosis:",
    ["Suspek", "Probabel", "Konfirm"],
    default=["Suspek", "Probabel", "Konfirm"]
)

daftar_penyakit_tetap = ["Semua Penyakit", "Leptospirosis", "ISPA Berat", "Diare Akut", "Demam Dengue"]
all_diseases_dynamic = []
for col in ["Diagnosis Suspek", "Diagnosis Probabel", "Diagnosis Konfirm"]:
    if col in df_raw.columns:
        for val in df_raw[col].dropna().unique():
            val_str = str(val).strip()
            if val_str not in ["Bukan", "nan", ""] and val_str not in daftar_penyakit_tetap:
                all_diseases_dynamic.append(val_str)
final_disease_list = daftar_penyakit_tetap + sorted(set(all_diseases_dynamic))
sel_disease = st.sidebar.selectbox("Pilih Target Penyakit:", final_disease_list, index=0)

# ================================================================
# APPLY FILTERS
# ================================================================
df_active = df_raw.copy()
if sel_prov:
    df_active = df_active[df_active['Provinsi'].astype(str).isin(sel_prov)]
else:
    df_active = df_active.iloc[0:0]
if sel_kab:
    df_active = df_active[df_active['Kabupaten'].astype(str).isin(sel_kab)]
else:
    df_active = df_active.iloc[0:0]

# Diagnosis-level filter: keep records that have at least one selected level.
diag_mask = pd.Series(False, index=df_active.index)
for level in sel_tingkat_diag:
    col = {"Suspek":"Diagnosis Suspek", "Probabel":"Diagnosis Probabel", "Konfirm":"Diagnosis Konfirm"}[level]
    if col in df_active.columns:
        diag_mask = diag_mask | df_active[col].notna() & (df_active[col].astype(str) != "Bukan")
df_active = df_active[diag_mask] if sel_tingkat_diag else df_active.iloc[0:0]

if sel_disease != "Semua Penyakit":
    disease_mask = pd.Series(False, index=df_active.index)
    for col in ["Diagnosis Suspek", "Diagnosis Probabel", "Diagnosis Konfirm"]:
        if col in df_active.columns:
            disease_mask = disease_mask | (df_active[col].astype(str) == sel_disease)
    df_active = df_active[disease_mask]

if 'Tanggal Sakit' in df_active.columns:
    df_active['Tanggal Sakit'] = pd.to_datetime(df_active['Tanggal Sakit'], errors='coerce')

st.sidebar.markdown("---")
st.sidebar.metric("Kasus Aktif", f"{len(df_active):,}")
st.sidebar.caption(f"Penyakit: **{sel_disease}**")

# ================================================================
# MAIN NAVIGATION
# ================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    '🧠 ML Intelligence Center',
    '📊 Trias Epidemiologi',
    '📈 Kurva & Forecast',
    '🗺️ Spatial Intelligence',
    '⚠️ Faktor Risiko',
    '📁 Dataset & Export'
])

# ================================================================
# TAB 1 — ML
# ================================================================
with tab1:
    st.markdown('## 🧠 ML Intelligence Center')
    st.info('Pipeline ML: Case Severity, KLB Prediction, Spatial Outbreak, Vulnerable Population, dan Robust Forecasting.')
    if len(df_active) < 10:
        st.warning('Data aktif terlalu sedikit untuk sebagian model ML. Perluas filter atau gunakan data yang lebih besar.')
    ml1, ml2, ml3, ml4, ml5 = st.tabs(['Severity', 'KLB', 'Spatial', 'Vulnerable', 'Forecasting'])

    with ml1:
        st.markdown('### Case Severity Prediction')
        try:
            result = train_case_severity(df_active)
            st.write(result.get('metrics', {}))
        except Exception as e:
            st.warning(f'Model severity belum dapat dijalankan: {e}')

    with ml2:
        st.markdown('### KLB Prediction')
        try:
            result = train_klb_prediction(df_active)
            st.write(result.get('metrics', {}))
        except Exception as e:
            st.warning(f'Model KLB belum dapat dijalankan: {e}')

    with ml3:
        st.markdown('### Spatial Outbreak Prediction')
        try:
            result = train_spatial_outbreak(df_active)
            st.write(result.get('metrics', {}))
        except Exception as e:
            st.warning(f'Model spatial belum dapat dijalankan: {e}')

    with ml4:
        st.markdown('### Vulnerable Population Prediction')
        try:
            result = train_vulnerable_population(df_active)
            st.write(result.get('metrics', {}))
        except Exception as e:
            st.warning(f'Model vulnerable population belum dapat dijalankan: {e}')

    with ml5:
        st.markdown('### Robust Forecasting')
        try:
            if 'Tanggal Sakit' in df_active.columns:
                daily = df_active.dropna(subset=['Tanggal Sakit']).groupby(df_active.dropna(subset=['Tanggal Sakit'])['Tanggal Sakit'].dt.date).size().reset_index(name='Jumlah Kasus')
                daily.columns = ['Tanggal Sakit', 'Jumlah Kasus']
                daily['Tanggal Sakit'] = pd.to_datetime(daily['Tanggal Sakit'])
                forecast = robust_forecast(daily, horizon=14)
                if isinstance(forecast, dict) and 'forecast' in forecast:
                    st.dataframe(pd.DataFrame({'Tanggal': forecast['dates'], 'Prediksi': forecast['forecast']}), use_container_width=True, hide_index=True)
                else:
                    st.info('Forecast belum tersedia.')
        except Exception as e:
            st.warning(f'Forecast belum dapat dijalankan: {e}')

# ================================================================
# TAB 2 — TRIAS
# ================================================================
with tab2:
    st.markdown('### 📊 Trias Epidemiologi — Person, Place, Time')
    vulnerable_profiles = identifikasi_vulnerable_profile(df_active)
    if vulnerable_profiles:
        st.markdown('#### Profil Populasi Rentan')
        st.dataframe(pd.DataFrame(vulnerable_profiles), use_container_width=True, hide_index=True)
    if len(df_active):
        st.markdown('#### Distribusi Wilayah')
        st.dataframe(
            df_active.groupby(['Provinsi','Kabupaten','Kecamatan','Desa/Kelurahan'])
            .size().reset_index(name='Jumlah Kasus')
            .sort_values('Jumlah Kasus', ascending=False),
            use_container_width=True, hide_index=True
        )

        st.markdown('#### Distribusi Diagnosis & Status Penderita')
        rows = []
        for level, col in [('Suspek','Diagnosis Suspek'),('Probabel','Diagnosis Probabel'),('Konfirm','Diagnosis Konfirm')]:
            if col in df_active.columns:
                tmp = df_active[df_active[col].astype(str) != 'Bukan']
                if not tmp.empty:
                    rows.append(pd.DataFrame({'Tingkat Diagnosis':[level], 'Jumlah Kasus':[len(tmp)]}))
        if rows:
            st.dataframe(pd.concat(rows, ignore_index=True), use_container_width=True, hide_index=True)
    else:
        st.warning('Tidak ada data yang sesuai filter.')

# ================================================================
# TAB 3 — CURVE & FORECAST
# ================================================================
with tab3:
    st.markdown('### 📈 Kurva Epidemik & Forecast')
    if 'Tanggal Sakit' in df_active.columns and not df_active['Tanggal Sakit'].isna().all():
        valid_dates = df_active.dropna(subset=['Tanggal Sakit'])
        df_epi = valid_dates.groupby(valid_dates['Tanggal Sakit'].dt.date).size().reset_index(name='Jumlah Kasus')
        df_epi.columns = ['Tanggal Sakit', 'Jumlah Kasus']
        df_epi['Tanggal Sakit'] = pd.to_datetime(df_epi['Tanggal Sakit'])

        # Compatibility fix: function requires disease_name and returns a tuple.
        curve = deteksi_bentuk_kurva(df_epi, sel_disease)
        if curve:
            shape, description, interpretation, prediction, metrics = curve
            st.info(f'**Bentuk kurva:** {shape}\n\n{description}')
            with st.expander('Interpretasi & prediksi'):
                st.write(interpretation)
                st.write(prediction)
                if metrics:
                    st.json(metrics)

        forecast_result = prediksi_kurva_holt_winters(df_epi, forecast_days=14)
        if isinstance(forecast_result, dict):
            fig, ax = plt.subplots(figsize=(14,5))
            ax.bar(df_epi['Tanggal Sakit'], df_epi['Jumlah Kasus'], alpha=.7, label='Historis')
            ax.plot(forecast_result['dates'], forecast_result['forecast'], '--', marker='o', label='Holt-Winters 14 Hari')
            ax.fill_between(forecast_result['dates'], forecast_result['lower'], forecast_result['upper'], alpha=.15, label='Interval')
            ax.legend(); ax.grid(alpha=.3); st.pyplot(fig)
            c1,c2,c3=st.columns(3)
            c1.metric('Tren', forecast_result['trend'])
            c2.metric('Prediksi Puncak', forecast_result['peak_date'].strftime('%d %b %Y'))
            c3.metric('Kasus Puncak', f"{forecast_result['peak_value']:.0f}")
        elif forecast_result == 'KURANG_DATA':
            st.info('Data historis kurang untuk forecast 14 hari.')

        rt_result = hitung_effective_rt(df_epi)
        if rt_result:
            st.metric('Rₜ rata-rata', f"{rt_result['rt_avg']:.2f}")
            st.info(rt_result['interpretation'])

        waves = deteksi_gelombang(df_epi)
        if waves:
            st.markdown('#### 🌊 Deteksi Gelombang Epidemi')
            st.dataframe(pd.DataFrame(waves), use_container_width=True, hide_index=True)
    else:
        st.warning('Data tanggal tidak tersedia.')

# ================================================================
# TAB 4 — SPATIAL
# ================================================================
with tab4:
    st.markdown('### 🗺️ Spatial Intelligence — AI DBSCAN')
    if {'Latitude','Longitude'}.issubset(df_active.columns):
        geo = df_active.dropna(subset=['Latitude','Longitude']).copy()
        if len(geo) >= 4:
            with st.expander('📚 Apa itu DBSCAN?', expanded=False):
                st.markdown('DBSCAN mengelompokkan kasus berdasarkan kedekatan geografis tanpa harus menentukan jumlah klaster terlebih dahulu.')
            coords_rad = np.radians(geo[['Latitude','Longitude']])
            db = DBSCAN(eps=3.0/6371.0088, min_samples=4, metric='haversine').fit(coords_rad)
            geo['Cluster'] = db.labels_
            st.dataframe(geo[['Provinsi','Kabupaten','Kecamatan','Desa/Kelurahan','Latitude','Longitude','Cluster']], use_container_width=True, hide_index=True)
            unique_clusters = sorted(set(db.labels_) - {-1})
            st.metric('Jumlah Klaster', len(unique_clusters))
            st.metric('Outlier / Noise', int((geo['Cluster'] == -1).sum()))

            m = folium.Map(location=[geo['Latitude'].mean(), geo['Longitude'].mean()], zoom_start=10)
            for _, r in geo.iterrows():
                folium.CircleMarker(
                    location=[r['Latitude'], r['Longitude']], radius=5,
                    popup=f"{r['Kabupaten']} | Cluster {r['Cluster']}"
                ).add_to(m)
            st_folium(m, width=None, height=500)
        else:
            st.warning('Minimal 4 kasus dengan koordinat valid diperlukan untuk DBSCAN.')
    else:
        st.warning('Kolom Latitude/Longitude tidak tersedia.')

# ================================================================
# TAB 5 — RISK FACTORS
# ================================================================
with tab5:
    st.markdown('### ⚠️ Analisis Faktor Risiko')
    if len(df_active) < 10:
        st.warning('Data aktif terlalu sedikit untuk analisis faktor risiko.')
    else:
        tab_bivar, tab_multivar = st.tabs(['🔬 Analisis Bivariat', '🧠 Analisis Multivariat'])
        with tab_bivar:
            col1, col2 = st.columns(2)
            with col1:
                var_indep = st.selectbox('Variabel Independen:', INDEPENDENT_VARS, key='bivar_indep')
            with col2:
                var_dep = st.selectbox('Variabel Dependen:', DEPENDENT_VARS, key='bivar_dep')
            try:
                df_biv = df_active.copy()
                if sel_disease == 'Semua Penyakit':
                    df_biv['Outcome_Binary'] = np.where(
                        (df_biv['Diagnosis Suspek'].astype(str) != 'Bukan') |
                        (df_biv['Diagnosis Probabel'].astype(str) != 'Bukan') |
                        (df_biv['Diagnosis Konfirm'].astype(str) != 'Bukan'), 1, 0
                    )
                else:
                    df_biv['Outcome_Binary'] = np.where(
                        (df_biv['Diagnosis Suspek'].astype(str) == sel_disease) |
                        (df_biv['Diagnosis Probabel'].astype(str) == sel_disease) |
                        (df_biv['Diagnosis Konfirm'].astype(str) == sel_disease), 1, 0
                    )
                analysis_var = 'Umur_Kategori' if var_indep == 'Umur' else var_indep
                if var_indep == 'Umur':
                    df_biv['Umur_Kategori'] = pd.cut(df_biv['Umur'], bins=[0,18,45,60,100], labels=['<=18','19-45','46-60','>60'])
                res = hitung_bivariat_lengkap(df_biv, analysis_var, 'Outcome_Binary')
                if isinstance(res, dict):
                    st.dataframe(res.get('crosstab'), use_container_width=True)
                    st.write({k: v for k,v in res.items() if k != 'crosstab'})
            except Exception as e:
                st.warning(f'Analisis bivariat belum dapat dijalankan: {e}')

        with tab_multivar:
            st.info('Analisis multivariat menggunakan regresi logistik dan perlu ditafsirkan oleh epidemiolog/statistikawan.')
            try:
                cols = [c for c in INDEPENDENT_VARS + ['Is_Konfirm'] if c in df_active.columns]
                dm = df_active[cols].dropna().copy()
                if len(dm) >= 30 and dm['Is_Konfirm'].nunique() > 1:
                    X = pd.get_dummies(dm[INDEPENDENT_VARS], drop_first=True, dtype=float)
                    X = sm.add_constant(X, has_constant='add')
                    y = pd.to_numeric(dm['Is_Konfirm'], errors='coerce').fillna(0).astype(int)
                    model = sm.Logit(y, X).fit(disp=False)
                    summary = pd.DataFrame({'OR': np.exp(model.params), 'p_value': model.pvalues, 'CI_low': np.exp(model.conf_int()[0]), 'CI_high': np.exp(model.conf_int()[1])})
                    st.dataframe(summary, use_container_width=True)
                else:
                    st.info('Data tidak cukup atau outcome hanya memiliki satu kelas untuk regresi logistik.')
            except Exception as e:
                st.warning(f'Analisis multivariat belum dapat dijalankan: {e}')

# ================================================================
# TAB 6 — DATA & EXPORT
# ================================================================
with tab6:
    st.markdown('### 📁 Dataset & Export')
    st.dataframe(df_active, use_container_width=True, height=450, hide_index=True)
    csv = df_active.to_csv(index=False).encode('utf-8')
    st.download_button('⬇️ Download CSV', csv, 'sihis_dataset_filtered.csv', 'text/csv', use_container_width=True)
    try:
        excel_bytes = io.BytesIO()
        with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
            df_active.to_excel(writer, index=False, sheet_name='Data')
        st.download_button(
            '⬇️ Download Excel', excel_bytes.getvalue(), 'sihis_dataset_filtered.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )
    except Exception as e:
        st.warning(f'Export Excel belum tersedia: {e}')
