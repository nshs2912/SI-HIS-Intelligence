import io
import math
import warnings
import html
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

components.html("""
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; }
        .header-khaki { background-color: #f0e68c; padding: 20px 25px; border-radius: 12px; border: 2px solid #d4c886; display: flex; align-items: center; box-sizing: border-box; }
        .header-khaki img { height: 70px; width: auto; margin-right: 20px; }
        .header-text h1 { margin: 0; color: #1e3a8a; font-size: 1.8rem; font-weight: 800; line-height: 1.2; }
        .header-text h3 { margin: 5px 0 0 0; color: #475569; font-size: 1.1rem; font-weight: 600; letter-spacing: 1px; }
        .clock-text { margin: 8px 0 0 0; color: #1e3a8a; font-size: 0.95rem; font-weight: 700; letter-spacing: 0.5px; }
    </style>
</head>
<body>
    <div class="header-khaki">
        <img src="https://cdn-icons-png.flaticon.com/512/2966/2966334.png" alt="Logo Surveilans">
        <div class="header-text">
            <h1>SIHIS-Smart Integrated Health Intelligence System</h1>
            <h3>Early Detection, Smarter Intervention - Created by Nur Subagyo HS</h3>
            <p class="clock-text">🕒 Waktu Saat Ini (WIB / GMT+7): <span id="live-wib-clock" style="font-variant-numeric: tabular-nums;">Memuat waktu...</span></p>
        </div>
    </div>
    <script>
        function updateWIBClock() {
            const now = new Date(); const utc = now.getTime() + (now.getTimezoneOffset() * 60000); const wibTime = new Date(utc + (7 * 3600000));
            const days = ["Minggu", "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"];
            const months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"];
            const dayName = days[wibTime.getDay()], date = wibTime.getDate(), monthName = months[wibTime.getMonth()], year = wibTime.getFullYear();
            const hours = String(wibTime.getHours()).padStart(2, '0'), minutes = String(wibTime.getMinutes()).padStart(2, '0'), seconds = String(wibTime.getSeconds()).padStart(2, '0');
            const clockElement = document.getElementById('live-wib-clock'); if (clockElement) clockElement.textContent = `${dayName}, ${date} ${monthName} ${year}, ${hours}:${minutes}:${seconds} WIB`;
        }
        updateWIBClock(); setInterval(updateWIBClock, 1000);
    </script>
</body>
</html>
""", height=140)

st.markdown("""
<div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 18px 22px; border-radius: 10px; border-left: 5px solid #d97706; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(217, 119, 6, 0.15);">
<p style="color: #78350f; font-size: 0.95rem; font-weight: 700; margin: 0 0 12px 0;">⚠️ DISCLAIMER PENTING</p>
<ol style="color: #92400e; font-size: 0.9rem; margin: 0; padding-left: 20px; line-height: 1.7;">
<li style="margin-bottom: 8px;">Hasil analisis berfungsi sebagai alat bantu pengolahan data dipergunakan pribadi, bukan untuk umum.<strong>Decision Support System (DSS)</strong>. Keputusan operasional tetap berada di bawah wewenang otoritas kesehatan.</li>
<li>Anda sedang menggunakan aplikasi <strong>AI Analysis Free Version</strong>, sehingga data yang diupload <strong>tidak tersimpan di sistem</strong> dan akan terhapus saat browser di-refresh.</li>
</ol></div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Panel Kontrol & Filter")
mode_data = st.sidebar.radio("Pilih Sumber Data:", ["Gunakan Data Simulasi (AI-Ready)", "Upload File Excel/CSV Custom"])
if mode_data == "Gunakan Data Simulasi (AI-Ready)":
    df_raw = generate_data_simulasi(); st.sidebar.success("✅ Data simulasi 800 kasus dimuat.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload File Kasus (.xlsx / .csv)", type=["xlsx", "csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
            if missing_cols: st.sidebar.error(f"❌ Kolom wajib kurang: {missing_cols}"); st.stop()
            st.sidebar.success(f"✅ Berhasil memuat {len(df_raw)} baris data.")
        except Exception as e: st.sidebar.error(f"❌ Error membaca file: {e}"); st.stop()
    else: st.sidebar.info("👈 Silakan upload file untuk memulai."); st.stop()

st.sidebar.markdown("---")
st.sidebar.download_button(label="📥 Download Template Excel Standard", data=generate_excel_template(), file_name="Template_Data_Surveilans.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Filter Wilayah & Diagnosis")
list_prov = sorted(df_raw['Provinsi'].dropna().unique().tolist())
sel_prov = st.sidebar.multiselect("Pilih Provinsi:", list_prov, default=list_prov)
df_filtered_prov = df_raw[df_raw['Provinsi'].isin(sel_prov)] if sel_prov else df_raw
list_kab = sorted(df_filtered_prov['Kabupaten'].dropna().unique().tolist())
sel_kab = st.sidebar.multiselect("Pilih Kabupaten:", list_kab, default=list_kab)
sel_tingkat_diag = st.sidebar.multiselect("Pilih Tingkat Diagnosis:", ["Suspek", "Probabel", "Konfirm"], default=["Suspek", "Probabel", "Konfirm"])
daftar_penyakit_tetap = ["Semua Penyakit", "Leptospirosis", "ISPA Berat", "Diare Akut", "Demam Dengue"]
all_diseases_dynamic = []
for col in ["Diagnosis Suspek", "Diagnosis Probabel", "Diagnosis Konfirm"]:
    for val in df_raw[col].dropna().unique():
        val_str = str(val).strip()
        if val_str not in ["Bukan", "nan", ""] and val_str not in daftar_penyakit_tetap: all_diseases_dynamic.append(val_str)
final_disease_list = daftar_penyakit_tetap + sorted(list(set(all_diseases_dynamic)))
sel_disease = st.sidebar.selectbox("Pilih Target Penyakit:", final_disease_list, index=0)

df_active = df_raw.copy()
if sel_prov: df_active = df_active[df_active['Provinsi'].isin(sel_prov)]
if sel_kab: df_active = df_active[df_active['Kabupaten'].isin(sel_kab)]
if sel_disease != "Semua Penyakit":
    diagnosis_cols = {"Suspek": "Diagnosis Suspek", "Probabel": "Diagnosis Probabel", "Konfirm": "Diagnosis Konfirm"}
    selected_cols = [diagnosis_cols[x] for x in sel_tingkat_diag]
    if selected_cols:
        mask = False
        for col in selected_cols: mask = mask | (df_active[col] == sel_disease)
        df_active = df_active[mask]

st.markdown(f"<div class='context-box-sand'>📍 <b>KONTEKS ANALISIS:</b> {sel_disease} | <b>Parameter KLB:</b> ≥{KLB_THRESHOLD} kasus per Desa/Kelurahan</div>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
cfr = (df_active['Is_Meninggal'].sum() / len(df_active) * 100) if len(df_active) > 0 else 0
with m1: st.metric("Total Kasus", len(df_active))
with m2: st.metric("Kasus Konfirm", int(df_active['Is_Konfirm'].sum()))
with m3: st.metric("Rawat Inap", int((df_active['Status Penderita'] == 'Rawat Inap').sum()))
with m4: st.metric("Meninggal", int(df_active['Is_Meninggal'].sum()))
with m5: st.metric("CFR", f"{cfr:.2f}%")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚨 ML Intelligence Center", "📊 Trias Epidemiologi", "📈 Kurva & Forecast", "🗺️ Spatial Intelligence", "🧪 Faktor Risiko", "📁 Dataset & Export"])

with tab1:
    st.markdown("### 🤖 SI-HIS Machine Learning Intelligence Center")
    if len(df_active) < 30:
        st.warning("Data aktif kurang dari 30 baris; hasil ML dapat tidak stabil.")
    else:
        ml1, ml2, ml3, ml4, ml5 = st.tabs(["1 · Severity", "2 · KLB", "3 · Spatial", "4 · Vulnerable", "5 · Forecasting"])
        with ml1:
            st.markdown("#### Case Severity Prediction")
            result = train_case_severity(df_active)
            if result.get('status') == 'ok':
                cols = st.columns(4)
                for c, k in zip(cols, ['roc_auc','pr_auc','accuracy','f1']): c.metric(k.upper(), f"{result.get(k,0):.3f}")
                st.dataframe(result['feature_importance'], use_container_width=True, hide_index=True)
                pred = predict_case_severity(df_active, result['model'])
                st.dataframe(pred.head(100), use_container_width=True, hide_index=True)
            else: st.info(result.get('message','Model belum dapat dilatih.'))
        with ml2:
            st.markdown("#### KLB / Outbreak 7-Day Prediction")
            result = train_klb_prediction(df_active)
            if result.get('status') == 'ok':
                cols = st.columns(4)
                for c, k in zip(cols, ['roc_auc','pr_auc','accuracy','f1']): c.metric(k.upper(), f"{result.get(k,0):.3f}")
                st.dataframe(result['feature_importance'], use_container_width=True, hide_index=True)
                st.dataframe(predict_klb(df_active, result['model']).head(100), use_container_width=True, hide_index=True)
            else: st.info(result.get('message','Model belum dapat dilatih.'))
        with ml3:
            st.markdown("#### Spatial Outbreak Prediction")
            result = train_spatial_outbreak(df_active)
            if result.get('status') == 'ok':
                cols = st.columns(4)
                for c, k in zip(cols, ['roc_auc','pr_auc','accuracy','f1']): c.metric(k.upper(), f"{result.get(k,0):.3f}")
                st.dataframe(result['feature_importance'], use_container_width=True, hide_index=True)
                st.dataframe(predict_spatial_outbreak(df_active, result['model']).head(100), use_container_width=True, hide_index=True)
            else: st.info(result.get('message','Model belum dapat dilatih.'))
        with ml4:
            st.markdown("#### Vulnerable Population Prediction")
            result = train_vulnerable_population(df_active)
            if result.get('status') == 'ok':
                cols = st.columns(4)
                for c, k in zip(cols, ['roc_auc','pr_auc','accuracy','f1']): c.metric(k.upper(), f"{result.get(k,0):.3f}")
                st.dataframe(result['feature_importance'], use_container_width=True, hide_index=True)
                st.dataframe(predict_vulnerable_population(df_active, result['model']).head(100), use_container_width=True, hide_index=True)
            else: st.info(result.get('message','Model belum dapat dilatih.'))
        with ml5:
            st.markdown("#### Robust Forecasting")
            if 'Tanggal Sakit' in df_active.columns:
                daily = df_active.groupby(pd.to_datetime(df_active['Tanggal Sakit']).dt.date).size().reset_index(name='Jumlah Kasus')
                daily.columns = ['Tanggal Sakit','Jumlah Kasus']; daily['Tanggal Sakit'] = pd.to_datetime(daily['Tanggal Sakit'])
                forecast = robust_forecast(daily, forecast_days=14)
                if forecast and forecast.get('status') == 'ok':
                    c1,c2,c3 = st.columns(3); c1.metric("MAE", f"{forecast.get('mae',0):.2f}"); c2.metric("RMSE", f"{forecast.get('rmse',0):.2f}"); c3.metric("Puncak", forecast['peak_date'].strftime('%d %b %Y'))
                    fig, ax = plt.subplots(figsize=(14,5)); ax.plot(daily['Tanggal Sakit'], daily['Jumlah Kasus'], label='Historis'); ax.plot(forecast['dates'], forecast['forecast'], '--', label='Forecast Ensemble'); ax.fill_between(forecast['dates'], forecast['lower'], forecast['upper'], alpha=.15, label='Interval'); ax.legend(); ax.grid(alpha=.3); st.pyplot(fig)
                else: st.info(forecast.get('message','Forecast belum tersedia.') if forecast else 'Forecast belum tersedia.')
            else: st.warning('Kolom Tanggal Sakit tidak tersedia.')
        with st.expander("ℹ️ Catatan Model & Validasi"):
            st.markdown("Model pada data simulasi hanya untuk demonstrasi pipeline. Untuk produksi gunakan data historis, temporal validation, calibration, external validation, model monitoring, dan human-in-the-loop.")

with tab2:
    st.markdown("### 📊 Trias Epidemiologi — Person, Place, Time")
    vulnerable_profiles = identifikasi_vulnerable_profile(df_active)
    if vulnerable_profiles:
        st.dataframe(pd.DataFrame(vulnerable_profiles), use_container_width=True, hide_index=True)
    if len(df_active):
        st.markdown("#### Distribusi Wilayah")
        st.dataframe(df_active.groupby(['Provinsi','Kabupaten','Kecamatan','Desa/Kelurahan']).size().reset_index(name='Jumlah Kasus').sort_values('Jumlah Kasus', ascending=False), use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 📈 Kurva Epidemik & Forecast")
    if 'Tanggal Sakit' in df_active.columns and not df_active['Tanggal Sakit'].isna().all():
        df_epi = df_active.groupby(pd.to_datetime(df_active['Tanggal Sakit']).dt.date).size().reset_index(name='Jumlah Kasus'); df_epi.columns=['Tanggal Sakit','Jumlah Kasus']; df_epi['Tanggal Sakit']=pd.to_datetime(df_epi['Tanggal Sakit'])
        curve = deteksi_bentuk_kurva(df_epi)
        if curve: st.info(f"Bentuk kurva: {curve.get('shape','-')} — {curve.get('desc','')}")
        forecast_result = prediksi_kurva_holt_winters(df_epi, forecast_days=14)
        if isinstance(forecast_result, dict):
            fig, ax = plt.subplots(figsize=(14,5)); ax.bar(df_epi['Tanggal Sakit'], df_epi['Jumlah Kasus'], alpha=.7, label='Historis'); ax.plot(forecast_result['dates'], forecast_result['forecast'], '--', marker='o', label='Holt-Winters 14 Hari'); ax.fill_between(forecast_result['dates'], forecast_result['lower'], forecast_result['upper'], alpha=.15); ax.legend(); ax.grid(alpha=.3); st.pyplot(fig)
            c1,c2,c3=st.columns(3); c1.metric('Tren', forecast_result['trend']); c2.metric('Prediksi Puncak', forecast_result['peak_date'].strftime('%d %b %Y')); c3.metric('Kasus Puncak', f"{forecast_result['peak_value']:.0f}")
        rt_result = hitung_effective_rt(df_epi)
        if rt_result: st.metric('Rₜ rata-rata', f"{rt_result['rt_avg']:.2f}"); st.info(rt_result['interpretation'])
        waves = deteksi_gelombang(df_epi)
        if waves: st.dataframe(pd.DataFrame(waves), use_container_width=True, hide_index=True)
    else: st.warning('Data tanggal tidak tersedia.')

with tab4:
    st.markdown("### 🗺️ Spatial Intelligence — DBSCAN")
    df_map_valid = df_active.dropna(subset=['Latitude','Longitude']).copy()
    if not df_map_valid.empty:
        coords_rad = np.radians(df_map_valid[['Latitude','Longitude']]); db = DBSCAN(eps=3.0/6371.0088, min_samples=4, metric='haversine').fit(coords_rad); df_map_valid['Cluster']=db.labels_
        unique_clusters = sorted(set(db.labels_)-{-1}); n_outliers=int((df_map_valid['Cluster']==-1).sum())
        m_db=folium.Map(location=[df_map_valid['Latitude'].mean(),df_map_valid['Longitude'].mean()],zoom_start=10)
        colors=['darkred','darkblue','darkgreen','purple','cadetblue','darkpurple','orange','pink']
        for _,r in df_map_valid.iterrows():
            cid=r['Cluster']; color='gray' if cid==-1 else colors[int(cid)%len(colors)]; folium.CircleMarker([r['Latitude'],r['Longitude']],radius=4,color=color,fill=True,fill_opacity=.7,tooltip=f"Cluster {cid}").add_to(m_db)
        st_folium(m_db,use_container_width=True,height=550)
        st.info(f"{len(unique_clusters)} cluster terdeteksi; {n_outliers} outlier.")
    else: st.warning('Tidak ada koordinat valid.')

with tab5:
    st.markdown("### 🧪 Analisis Faktor Risiko")
    col1,col2=st.columns(2)
    with col1: var_indep=st.selectbox('Variabel Independen:', INDEPENDENT_VARS,key='bivar_indep2')
    with col2: var_dep=st.selectbox('Variabel Dependen:', DEPENDENT_VARS,key='bivar_dep2')
    df_biv=df_active.copy(); df_biv['Outcome_Binary']=np.where(df_biv[var_dep]==sel_disease,1,0) if sel_disease!='Semua Penyakit' else np.where((df_biv['Diagnosis Suspek']!='Bukan')|(df_biv['Diagnosis Probabel']!='Bukan')|(df_biv['Diagnosis Konfirm']!='Bukan'),1,0)
    analysis_var='Umur_Kategori' if var_indep=='Umur' else var_indep
    if var_indep=='Umur': df_biv['Umur_Kategori']=pd.cut(df_biv['Umur'],bins=[0,18,45,60,100],labels=['<=18','19-45','46-60','>60'])
    res=hitung_bivariat_lengkap(df_biv,analysis_var,'Outcome_Binary'); st.dataframe(res['crosstab'],use_container_width=True,hide_index=True); st.write(f"Chi-Square={res['chi2']:.4f}; p-value={res['p_value']:.4f}; cOR={res['or']:.2f}" if pd.notna(res['or']) else f"Chi-Square={res['chi2']:.4f}; p-value={res['p_value']:.4f}")
    st.markdown('#### Regresi Logistik Multivariat')
    df_multi=df_active[['Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid','Riwayat Perjalanan','Faktor Risiko Lain','Is_Konfirm']].dropna().copy()
    if len(df_multi)>=50:
        try:
            df_enc=pd.get_dummies(df_multi,columns=['Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid','Riwayat Perjalanan','Faktor Risiko Lain'],drop_first=True,dtype=int); X,y=sm.add_constant(df_enc.drop('Is_Konfirm',axis=1)),df_enc['Is_Konfirm']; model=sm.Logit(y,X).fit(disp=False,maxiter=100); res_df=pd.DataFrame({'aOR':np.exp(model.params).round(2),'95% CI Lower':np.exp(model.conf_int()[0]).round(2),'95% CI Upper':np.exp(model.conf_int()[1]).round(2),'p-value':model.pvalues.round(4)}); st.dataframe(res_df,use_container_width=True)
        except Exception as e: st.warning(f'Model multivariat tidak stabil: {e}')
    else: st.warning('Sampel terlalu kecil (<50).')

with tab6:
    st.markdown("### 📁 Dataset Lengkap & Export")
    st.dataframe(df_active[REQUIRED_COLUMNS],use_container_width=True,height=400)
    c1,c2=st.columns(2)
    with c1:
        b=io.BytesIO();
        with pd.ExcelWriter(b,engine='openpyxl') as writer: df_active[REQUIRED_COLUMNS].to_excel(writer,index=False,sheet_name='Data')
        st.download_button('📥 Export Excel',b.getvalue(),'Data_Epidemiologi.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
    with c2: st.download_button('📥 Export CSV',df_active[REQUIRED_COLUMNS].to_csv(index=False).encode('utf-8'),'Data_Epidemiologi.csv','text/csv',use_container_width=True)
