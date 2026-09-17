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

# --- existing application UI / data preparation remains unchanged ---
# The critical compatibility fix below keeps the epidemiology curve function's
# current return contract: (shape, description, interpretation, prediction, metrics).

components.html("""
<!DOCTYPE html>
<html><body></body></html>
""", height=0)

st.set_page_config(page_title='SI-HIS Intelligence', layout='wide')

st.title('SI-HIS — Smart Integrated Health Intelligence System')
st.caption('Early Detection, Smarter Intervention')

# Load simulation data for the MVP when no uploaded dataset is available.
if 'df_active' not in st.session_state:
    st.session_state.df_active = generate_data_simulasi()
df_active = st.session_state.df_active.copy()

# Tabs
(tab1, tab2, tab3, tab4, tab5, tab6) = st.tabs([
    '🧠 ML Intelligence Center', '📊 Trias Epidemiologi', '📈 Kurva & Forecast',
    '🗺️ Spatial Intelligence', '⚠️ Faktor Risiko', '📁 Dataset & Export'
])

with tab1:
    st.markdown('## 🧠 ML Intelligence Center')
    st.info('Pipeline ML mencakup severity, prediksi KLB, spatial outbreak, vulnerable population, dan forecasting.')
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
            daily = df_active.groupby(pd.to_datetime(df_active['Tanggal Sakit']).dt.date).size().reset_index(name='Jumlah Kasus')
            daily.columns = ['Tanggal Sakit', 'Jumlah Kasus']
            daily['Tanggal Sakit'] = pd.to_datetime(daily['Tanggal Sakit'])
            forecast = robust_forecast(daily, horizon=14)
            if isinstance(forecast, dict) and 'forecast' in forecast:
                st.dataframe(pd.DataFrame({'Tanggal': forecast['dates'], 'Prediksi': forecast['forecast']}), use_container_width=True, hide_index=True)
            else:
                st.info('Forecast belum tersedia.')
        except Exception as e:
            st.warning(f'Forecast belum dapat dijalankan: {e}')

with tab2:
    st.markdown('### 📊 Trias Epidemiologi — Person, Place, Time')
    vulnerable_profiles = identifikasi_vulnerable_profile(df_active)
    if vulnerable_profiles:
        st.dataframe(pd.DataFrame(vulnerable_profiles), use_container_width=True, hide_index=True)
    if len(df_active):
        st.markdown('#### Distribusi Wilayah')
        st.dataframe(
            df_active.groupby(['Provinsi','Kabupaten','Kecamatan','Desa/Kelurahan'])
            .size().reset_index(name='Jumlah Kasus')
            .sort_values('Jumlah Kasus', ascending=False),
            use_container_width=True, hide_index=True
        )

with tab3:
    st.markdown('### 📈 Kurva Epidemik & Forecast')
    if 'Tanggal Sakit' in df_active.columns and not df_active['Tanggal Sakit'].isna().all():
        df_epi = (
            df_active.groupby(pd.to_datetime(df_active['Tanggal Sakit'], errors='coerce').dt.date)
            .size().reset_index(name='Jumlah Kasus')
        )
        df_epi.columns = ['Tanggal Sakit', 'Jumlah Kasus']
        df_epi['Tanggal Sakit'] = pd.to_datetime(df_epi['Tanggal Sakit'])

        # FIX: deteksi_bentuk_kurva requires disease_name and returns a tuple,
        # not a dictionary. Use the actual contract from core.analytics.py.
        disease_name = 'Semua Penyakit'
        curve = deteksi_bentuk_kurva(df_epi, disease_name)
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
            ax.fill_between(forecast_result['dates'], forecast_result['lower'], forecast_result['upper'], alpha=.15)
            ax.legend(); ax.grid(alpha=.3); st.pyplot(fig)
            c1,c2,c3=st.columns(3)
            c1.metric('Tren', forecast_result['trend'])
            c2.metric('Prediksi Puncak', forecast_result['peak_date'].strftime('%d %b %Y'))
            c3.metric('Kasus Puncak', f"{forecast_result['peak_value']:.0f}")
        rt_result = hitung_effective_rt(df_epi)
        if rt_result:
            st.metric('Rₜ rata-rata', f"{rt_result['rt_avg']:.2f}")
            st.info(rt_result['interpretation'])
        waves = deteksi_gelombang(df_epi)
        if waves:
            st.dataframe(pd.DataFrame(waves), use_container_width=True, hide_index=True)
    else:
        st.warning('Data tanggal tidak tersedia.')

with tab4:
    st.markdown('### 🗺️ Spatial Intelligence')
    st.info('Analisis spasial menggunakan koordinat kasus dan clustering DBSCAN.')
    if {'Latitude','Longitude'}.issubset(df_active.columns):
        geo = df_active.dropna(subset=['Latitude','Longitude']).copy()
        if len(geo) >= 4:
            coords_rad = np.radians(geo[['Latitude','Longitude']])
            db = DBSCAN(eps=3.0/6371.0088, min_samples=4, metric='haversine').fit(coords_rad)
            geo['Cluster'] = db.labels_
            st.dataframe(geo[['Provinsi','Kabupaten','Kecamatan','Desa/Kelurahan','Latitude','Longitude','Cluster']], use_container_width=True, hide_index=True)

with tab5:
    st.markdown('### ⚠️ Faktor Risiko')
    try:
        bi = hitung_bivariat_lengkap(df_active)
        if bi is not None:
            st.dataframe(pd.DataFrame(bi), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f'Analisis faktor risiko belum dapat dijalankan: {e}')

with tab6:
    st.markdown('### 📁 Dataset & Export')
    st.dataframe(df_active.head(100), use_container_width=True, hide_index=True)
    csv = df_active.to_csv(index=False).encode('utf-8')
    st.download_button('⬇️ Download CSV', csv, 'sihis_dataset.csv', 'text/csv')
