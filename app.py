import warnings
from datetime import datetime, timedelta, timezone

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.analytics import (
    REQUIRED_COLUMNS, get_wib_time, hitung_risk_stratification,
    identifikasi_vulnerable_profile, hitung_early_warning_score,
    deteksi_bentuk_kurva, prediksi_kurva_holt_winters,
    hitung_effective_rt, deteksi_gelombang, lengkapi_koordinat_otomatis,
    generate_data_simulasi, generate_excel_template, hitung_bivariat_lengkap,
)
from core.ml_engine import (
    train_case_severity, predict_case_severity,
    train_klb_prediction, predict_klb,
    train_spatial_outbreak, predict_spatial_outbreak,
    train_vulnerable_population, predict_vulnerable_population,
    robust_forecast,
)

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SI-HIS Intelligence", page_icon="🧠", layout="wide")

st.markdown("""
<div style="background:#f0e68c;padding:18px 22px;border-radius:12px;border:2px solid #d4c886;">
<h1 style="margin:0;color:#1e3a8a;font-size:1.7rem;">SI-HIS — Smart Integrated Health Intelligence System</h1>
<p style="margin:6px 0 0;color:#475569;font-weight:600;">Early Detection, Smarter Intervention</p>
</div>
""", unsafe_allow_html=True)
st.caption(f"🕒 Waktu Sistem: {get_wib_time()['full']}")

st.sidebar.header("⚙️ Panel Kontrol & Filter")
mode_data = st.sidebar.radio("Pilih Sumber Data:", ["Gunakan Data Simulasi (AI-Ready)", "Upload File Excel/CSV Custom"], key="data_source_mode")
if mode_data == "Gunakan Data Simulasi (AI-Ready)":
    df_raw = generate_data_simulasi().copy()
    st.sidebar.success(f"✅ Data simulasi {len(df_raw):,} kasus dimuat.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload File Kasus (.xlsx / .csv)", type=["xlsx", "csv"])
    if uploaded_file is None:
        st.sidebar.info("👈 Silakan upload file untuk memulai.")
        st.stop()
    try:
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
    except Exception as exc:
        st.sidebar.error(f"❌ Error membaca file: {exc}")
        st.stop()
    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        st.sidebar.error(f"❌ Kolom wajib kurang: {missing}")
        st.stop()
    st.sidebar.success(f"✅ Berhasil memuat {len(df_raw):,} baris data.")
try:
    st.sidebar.download_button("📥 Download Template Excel Standard", generate_excel_template(), "Template_Data_Surveilans.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
except Exception:
    pass
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Filter Wilayah & Diagnosis")
list_prov = sorted(df_raw["Provinsi"].dropna().astype(str).unique())
sel_prov = st.sidebar.multiselect("Pilih Provinsi:", list_prov, default=list_prov)
df_prov = df_raw[df_raw["Provinsi"].astype(str).isin(sel_prov)] if sel_prov else df_raw.iloc[0:0]
list_kab = sorted(df_prov["Kabupaten"].dropna().astype(str).unique())
sel_kab = st.sidebar.multiselect("Pilih Kabupaten:", list_kab, default=list_kab)
sel_tingkat_diag = st.sidebar.multiselect("Pilih Tingkat Diagnosis:", ["Suspek", "Probabel", "Konfirm"], default=["Suspek", "Probabel", "Konfirm"])
base_diseases = ["Semua Penyakit", "Leptospirosis", "ISPA Berat", "Diare Akut", "Demam Dengue"]
dynamic = []
for col in ["Diagnosis Suspek", "Diagnosis Probabel", "Diagnosis Konfirm"]:
    if col in df_raw.columns:
        dynamic.extend([str(x).strip() for x in df_raw[col].dropna().unique() if str(x).strip() not in ["Bukan", "nan", ""]])
final_diseases = base_diseases + sorted(set(dynamic) - set(base_diseases))
sel_disease = st.sidebar.selectbox("Pilih Target Penyakit:", final_diseases)
_df = df_raw.copy()
if sel_prov: _df = _df[_df["Provinsi"].astype(str).isin(sel_prov)]
else: _df = _df.iloc[0:0]
if sel_kab: _df = _df[_df["Kabupaten"].astype(str).isin(sel_kab)]
else: _df = _df.iloc[0:0]
mask = pd.Series(False, index=_df.index)
level_cols = {"Suspek": "Diagnosis Suspek", "Probabel": "Diagnosis Probabel", "Konfirm": "Diagnosis Konfirm"}
for level in sel_tingkat_diag:
    col = level_cols[level]
    if col in _df.columns: mask |= _df[col].notna() & (_df[col].astype(str) != "Bukan")
df_active = _df[mask] if sel_tingkat_diag else _df.iloc[0:0]
if sel_disease != "Semua Penyakit":
    dmask = pd.Series(False, index=df_active.index)
    for col in level_cols.values():
        if col in df_active.columns: dmask |= df_active[col].astype(str).eq(sel_disease)
    df_active = df_active[dmask]
if "Tanggal Sakit" in df_active.columns: df_active["Tanggal Sakit"] = pd.to_datetime(df_active["Tanggal Sakit"], errors="coerce")
st.sidebar.markdown("---")
st.sidebar.metric("Kasus Aktif", f"{len(df_active):,}")
st.sidebar.caption(f"Penyakit: **{sel_disease}**")

# Dynamic disease context: keep the selected disease visible as the dashboard title.
disease_context = sel_disease if sel_disease else "Semua Penyakit"
st.markdown(
    f"## 🦠 Dashboard Intelligence: {disease_context}"
)
st.caption(
    f"Seluruh analisis di bawah mengikuti filter penyakit **{disease_context}**, "
    f"wilayah, dan tingkat diagnosis yang dipilih."
)

def build_risk_table(df):
    if df.empty: return pd.DataFrame()
    work = df.copy()
    if "Is_Meninggal" in work.columns: work["_death"] = pd.to_numeric(work["Is_Meninggal"], errors="coerce").fillna(0)
    else: work["_death"] = work.get("Status Penderita", "").astype(str).str.lower().eq("meninggal").astype(int)
    group_cols = ["Provinsi", "Kabupaten", "Kecamatan", "Desa/Kelurahan"]
    g = work.groupby(group_cols, dropna=False).agg(Total_Kasus=(group_cols[-1], "size"), Meninggal=("_death", "sum")).reset_index()
    g["CFR (%)"] = np.where(g["Total_Kasus"] > 0, g["Meninggal"] / g["Total_Kasus"] * 100, 0).round(2)
    return hitung_risk_stratification(g)

def build_epi(df):
    if df.empty or "Tanggal Sakit" not in df.columns: return pd.DataFrame(columns=["Tanggal Sakit", "Jumlah Kasus"])
    d = df.dropna(subset=["Tanggal Sakit"]).copy()
    if d.empty: return pd.DataFrame(columns=["Tanggal Sakit", "Jumlah Kasus"])
    out = d.groupby(d["Tanggal Sakit"].dt.date).size().reset_index(name="Jumlah Kasus")
    out.columns = ["Tanggal Sakit", "Jumlah Kasus"]
    out["Tanggal Sakit"] = pd.to_datetime(out["Tanggal Sakit"])
    return out.sort_values("Tanggal Sakit")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚨 AI Prediction & Recommendation", "📊 Trias Epidemiologi (Detail)", "📈 Kurva Epidemik & Prediksi", "🗺️ Peta Spasial & AI DBSCAN", "🧪 Analisis Faktor Risiko", "🧠 ML Intelligence Engine"])

with tab1:
    st.markdown(f"## 🚨 AI Prediction & Recommendation — {disease_context}")
    st.caption("Situational Intelligence: profiling kerentanan, stratifikasi risiko wilayah, early warning, outlier, dan rekomendasi respons.")
    if df_active.empty: st.warning("Tidak ada data aktif sesuai filter.")
    else:
        vulnerable = identifikasi_vulnerable_profile(df_active); risk_df = build_risk_table(df_active); epi = build_epi(df_active)
        ews_score, trend_pct, ews_level = hitung_early_warning_score(epi) if not epi.empty else (None, None, None)
        c1,c2,c3,c4=st.columns(4); c1.metric("Kasus Aktif",f"{len(df_active):,}"); c2.metric("Kabupaten/Kota",f"{df_active['Kabupaten'].nunique():,}"); c3.metric("Kecamatan",f"{df_active['Kecamatan'].nunique():,}"); c4.metric("Desa/Kelurahan",f"{df_active['Desa/Kelurahan'].nunique():,}")
        st.markdown("### 👥 Profil Populasi Rentan")
        if vulnerable: st.dataframe(pd.DataFrame(vulnerable),use_container_width=True,hide_index=True)
        else: st.info("Belum ditemukan profil populasi rentan pada data aktif.")
        st.markdown("### 🟥 Stratifikasi Risiko Wilayah")
        if not risk_df.empty: st.dataframe(risk_df.sort_values("Risk_Score",ascending=False),use_container_width=True,hide_index=True)
        else: st.info("Stratifikasi risiko belum tersedia.")
        st.markdown("### 🚨 Early Warning System")
        if ews_score is not None:
            a,b,c=st.columns(3); a.metric("EWS Score",f"{ews_score:.1f}/100"); b.metric("Trend 7 hari",f"{trend_pct:+.1f}%"); c.metric("Level",ews_level); st.caption("EWS merupakan skor rule-based untuk sinyal perubahan tren; bukan probabilitas KLB terkalibrasi.")
        else: st.info("Minimal 14 hari observasi diperlukan untuk EWS.")
        st.markdown("### 🗺️ Regional CFR & Outlier")
        if "Is_Meninggal" in df_active.columns:
            reg=df_active.groupby(["Provinsi","Kabupaten"]).agg(Total=("Is_Meninggal","size"),Meninggal=("Is_Meninggal","sum")).reset_index(); reg["CFR (%)"]=(reg["Meninggal"]/reg["Total"]*100).round(2); st.dataframe(reg.sort_values("CFR (%)",ascending=False).head(20),use_container_width=True,hide_index=True)
        st.markdown("### 👤 Profil Demografis")
        d1,d2,d3=st.columns(3); d1.metric("Rata-rata Umur",f"{pd.to_numeric(df_active['Umur'],errors='coerce').mean():.1f} tahun"); d2.metric("Median Umur",f"{pd.to_numeric(df_active['Umur'],errors='coerce').median():.1f} tahun"); mode_gender=df_active["Jenis Kelamin"].mode(); d3.metric("Jenis Kelamin Terbanyak",str(mode_gender.iloc[0]) if not mode_gender.empty else "N/A")
        st.markdown("### 🤖 Strategic AI Recommendations")
        recs=[]
        if ews_score is not None and ews_score>=60: recs.append("Prioritaskan verifikasi lapangan dan penguatan surveilans pada sinyal tren tinggi.")
        if not risk_df.empty and (risk_df["Risk_Level"]=="HIGH").any(): recs.append("Terdapat wilayah dengan skor risiko relatif tinggi; lakukan investigasi epidemiologis terarah.")
        if vulnerable: recs.append("Fokuskan intervensi preventif pada profil populasi rentan yang teridentifikasi.")
        recs.append("Gunakan Trias, kurva epidemik, analisis spasial, faktor risiko, dan ML secara terpadu sebelum keputusan operasional.")
        for rec in recs: st.info("• "+rec)

with tab2:
    st.markdown(f"### 📊 Trias Epidemiologi — {disease_context} — Person, Place, Time")
    if df_active.empty: st.warning("Tidak ada data yang sesuai filter.")
    else:
        st.markdown("#### Place — Distribusi Wilayah"); place=df_active.groupby(["Provinsi","Kabupaten","Kecamatan","Desa/Kelurahan"]).size().reset_index(name="Jumlah Kasus"); st.dataframe(place.sort_values("Jumlah Kasus",ascending=False),use_container_width=True,hide_index=True)
        st.markdown("#### Person — Demografi"); c1,c2=st.columns(2)
        with c1:
            age=pd.to_numeric(df_active["Umur"],errors="coerce").dropna(); fig,ax=plt.subplots(figsize=(8,4)); ax.hist(age,bins=15); ax.set_xlabel("Umur"); ax.set_ylabel("Jumlah Kasus"); ax.grid(alpha=.3); st.pyplot(fig)
        with c2: st.dataframe(df_active["Jenis Kelamin"].value_counts().rename_axis("Jenis Kelamin").reset_index(name="Jumlah"),use_container_width=True,hide_index=True)
        st.markdown("#### Time — Distribusi Kasus"); epi=build_epi(df_active)
        if not epi.empty: st.line_chart(epi.set_index("Tanggal Sakit")["Jumlah Kasus"])
        st.markdown("#### Diagnosis"); rows=[]
        for label,col in [("Suspek","Diagnosis Suspek"),("Probabel","Diagnosis Probabel"),("Konfirm","Diagnosis Konfirm")]:
            if col in df_active.columns: rows.append({"Tingkat Diagnosis":label,"Jumlah Kasus":int((df_active[col].astype(str)!="Bukan").sum())})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

with tab3:
    st.markdown(f"### 📈 Kurva Epidemik & Prediksi — {disease_context}"); epi=build_epi(df_active)
    if epi.empty: st.warning("Data tanggal tidak tersedia.")
    else:
        if len(epi)>=7:
            shape,desc,interp,pred,metrics=deteksi_bentuk_kurva(epi,sel_disease); st.info(f"**Bentuk kurva:** {shape}\n\n{desc}")
            with st.expander("Interpretasi & prediksi"): st.write(interp); st.write(pred); st.json(metrics)
        forecast=prediksi_kurva_holt_winters(epi,14)
        if isinstance(forecast,dict):
            fig,ax=plt.subplots(figsize=(14,5)); ax.bar(epi["Tanggal Sakit"],epi["Jumlah Kasus"],alpha=.7,label="Historis"); ax.plot(forecast["dates"],forecast["forecast"],"--o",label="Holt-Winters 14 Hari"); ax.fill_between(forecast["dates"],forecast["lower"],forecast["upper"],alpha=.15,label="Interval"); ax.legend(); ax.grid(alpha=.3); st.pyplot(fig)
            a,b,c=st.columns(3); a.metric("Tren",forecast["trend"]); b.metric("Prediksi Puncak",forecast["peak_date"].strftime("%d %b %Y")); c.metric("Kasus Puncak",f"{forecast['peak_value']:.0f}")
        elif forecast=="KURANG_DATA": st.info("Data historis kurang untuk forecast 14 hari.")
        rt=hitung_effective_rt(epi)
        if rt: st.metric("Rₜ rata-rata",f"{rt['rt_avg']:.2f}"); st.info(rt["interpretation"])
        waves=deteksi_gelombang(epi)
        if waves: st.markdown("#### 🌊 Deteksi Gelombang Epidemi"); st.dataframe(pd.DataFrame(waves),use_container_width=True,hide_index=True)

with tab4:
    st.markdown(f"### 🗺️ Peta Spasial & AI DBSCAN — {disease_context}")
    if df_active.empty: st.warning("Tidak ada data aktif.")
    else:
        geo=df_active.dropna(subset=["Latitude","Longitude"]).copy()
        if geo.empty:
            st.info("Koordinat belum tersedia. Sistem dapat mencoba geocoding otomatis pada data yang memiliki alamat wilayah.")
            if st.button("📍 Lengkapi koordinat otomatis"):
                df_active=lengkapi_koordinat_otomatis(df_active); geo=df_active.dropna(subset=["Latitude","Longitude"]).copy()
        if not geo.empty:
            from sklearn.cluster import DBSCAN
            coords=np.radians(geo[["Latitude","Longitude"]].astype(float)); labels=DBSCAN(eps=3.0/6371.0088,min_samples=4,metric="haversine").fit(coords).labels_; geo["Cluster"]=labels
            st.dataframe(geo[["Provinsi","Kabupaten","Kecamatan","Desa/Kelurahan","Latitude","Longitude","Cluster"]].head(500),use_container_width=True,hide_index=True); center=[float(geo["Latitude"].mean()),float(geo["Longitude"].mean())]; m=folium.Map(location=center,zoom_start=8)
            for _,r in geo.head(500).iterrows(): folium.CircleMarker([r["Latitude"],r["Longitude"]],radius=4,popup=f"{r['Desa/Kelurahan']} | Cluster {r['Cluster']}").add_to(m)
            st_folium(m,width=None,height=550)

with tab5:
    st.markdown(f"### 🧪 Analisis Faktor Risiko — {disease_context}")
    if df_active.empty: st.warning("Tidak ada data aktif.")
    else:
        try:
            result=hitung_bivariat_lengkap(df_active)
            if isinstance(result,pd.DataFrame): st.dataframe(result,use_container_width=True,hide_index=True)
            elif isinstance(result,dict):
                for key,value in result.items():
                    st.markdown(f"#### {key}")
                    if isinstance(value,pd.DataFrame): st.dataframe(value,use_container_width=True,hide_index=True)
                    else: st.write(value)
            else: st.info("Hasil analisis faktor risiko belum tersedia dalam format tabel.")
        except Exception as exc: st.warning(f"Analisis bivariat tidak dapat dijalankan pada filter ini: {exc}")

with tab6:
    st.markdown(f"### 🧠 ML Intelligence Engine — {disease_context}")
    st.caption("ML menambah lapisan prediktif di atas analitik epidemiologi asli; tidak menggantikan Trias, EWS, DBSCAN, forecast, atau analisis faktor risiko.")
    ml1,ml2,ml3,ml4,ml5=st.tabs(["Severity","KLB","Spatial","Vulnerable","Forecasting"])
    with ml1:
        st.markdown("#### 🩺 Case Severity Prediction")
        st.info("**Fungsi:** memperkirakan risiko severity berdasarkan karakteristik kasus yang tersedia.\n\n**Interpretasi:** gunakan hasil untuk membantu prioritas review klinis, bukan untuk menetapkan diagnosis atau prognosis secara otomatis.")
        if st.button("Train Severity Model",key="train_severity"):
            res=train_case_severity(df_active); st.session_state["severity_model"]=res if res.get("status")=="ok" else None; st.session_state["severity_result"]=res
        res=st.session_state.get("severity_result")
        if res:
            if res.get("status")=="ok":
                st.metric("Model ROC-AUC",f"{res['metrics'].get('roc_auc',np.nan):.2f}"); st.markdown("**🧠 Interpretasi Model**"); st.write(res["metrics"].get("Narrative","")); st.markdown("**🔎 Faktor yang berkontribusi pada prediksi**"); st.dataframe(res["feature_importance"],use_container_width=True,hide_index=True)
                pred=predict_case_severity(df_active,res["model"])
                if not pred.empty: st.dataframe(pred[["Nama","Severity_Risk","Severity_Risk_Level"]].head(20),use_container_width=True,hide_index=True)
            else: st.warning(res.get("message","Training gagal."))
        st.warning("⚠️ **Disclaimer Klinis:** Output merupakan clinical decision-support dan bukan diagnosis, prognosis definitif, atau pengganti penilaian dokter. Hasil harus dikonfirmasi dengan anamnesis, pemeriksaan fisik, pemeriksaan penunjang, rekam medis, dan pertimbangan klinis tenaga kesehatan.")
    with ml2:
        st.markdown("#### 🚨 KLB / Outbreak 7-Day Prediction")
        st.info("**Fungsi:** mendeteksi sinyal risiko peningkatan kasus dalam 7 hari berikutnya pada tingkat wilayah.\n\n**Interpretasi:** hasil dapat membantu menentukan prioritas surveilans dan verifikasi lapangan.")
        if st.button("Train KLB Model",key="train_klb"): st.session_state["klb_result"]=train_klb_prediction(df_active)
        res=st.session_state.get("klb_result")
        if res:
            if res.get("status")=="ok":
                st.metric("Model ROC-AUC",f"{res['metrics'].get('roc_auc',np.nan):.2f}"); st.markdown("**🧠 Interpretasi Model**"); st.write(res["metrics"].get("Narrative","")); st.caption(f"Derived demo threshold: {res.get('derived_threshold')}"); pred=predict_klb(df_active,res["model"])
                if not pred.empty: st.dataframe(pred[["Desa/Kelurahan","KLB_Risk","KLB_Risk_Level"]].head(30),use_container_width=True,hide_index=True)
            else: st.warning(res.get("message","Training gagal."))
        st.warning("⚠️ **Disclaimer Epidemiologi:** Prediksi ini adalah sinyal peringatan dini berbasis model, bukan penetapan bahwa wilayah telah mengalami KLB/wabah. Penetapan harus mengikuti definisi operasional, kriteria surveilans, verifikasi data, investigasi epidemiologi, dan kewenangan otoritas kesehatan.")
    with ml3:
        st.markdown("#### 🗺️ Spatial Outbreak Prediction")
        st.info("**Fungsi:** mengidentifikasi wilayah dengan pola risiko spasial yang perlu diperhatikan.\n\n**Interpretasi:** gunakan bersama DBSCAN, distribusi kasus, dan investigasi epidemiologi.")
        if st.button("Train Spatial Model",key="train_spatial"): st.session_state["spatial_result"]=train_spatial_outbreak(df_active)
        res=st.session_state.get("spatial_result")
        if res:
            if res.get("status")=="ok":
                st.metric("Model ROC-AUC",f"{res['metrics'].get('roc_auc',np.nan):.2f}"); st.markdown("**🧠 Interpretasi Model**"); st.write(res["metrics"].get("Narrative","")); pred=predict_spatial_outbreak(df_active,res["model"])
                if not pred.empty: st.dataframe(pred[["Desa/Kelurahan","Spatial_Risk","Spatial_Risk_Level","Lat","Lon"]].head(30),use_container_width=True,hide_index=True)
            else: st.warning(res.get("message","Training gagal."))
        st.warning("⚠️ **Disclaimer Epidemiologi-Spasial:** Risiko spasial atau cluster tidak secara otomatis membuktikan adanya transmisi, sumber penularan, atau hubungan kausal. Hasil harus diverifikasi dengan data lapangan dan investigasi epidemiologi.")
    with ml4:
        st.markdown("#### 👥 Vulnerable Population Prediction")
        st.info("**Fungsi:** mengidentifikasi profil individu/populasi yang menunjukkan pola risiko relatif lebih tinggi pada data model.\n\n**Interpretasi:** hasil dapat digunakan untuk prioritas pemantauan atau skrining, bukan penetapan status kerentanan secara definitif.")
        if st.button("Train Vulnerable Model",key="train_vulnerable"): st.session_state["vulnerable_result"]=train_vulnerable_population(df_active)
        res=st.session_state.get("vulnerable_result")
        if res:
            if res.get("status")=="ok":
                st.metric("Model ROC-AUC",f"{res['metrics'].get('roc_auc',np.nan):.2f}"); st.markdown("**🧠 Interpretasi Model**"); st.write(res["metrics"].get("Narrative","")); pred=predict_vulnerable_population(df_active,res["model"])
                if not pred.empty: st.dataframe(pred[["Nama","Vulnerable_Risk","Vulnerable_Risk_Level"]].head(30),use_container_width=True,hide_index=True)
            else: st.warning(res.get("message","Training gagal."))
        st.warning("⚠️ **Disclaimer Populasi Rentan:** Prediksi merupakan estimasi berbasis pola data dan bukan penetapan status kerentanan individu secara definitif. Kualitas, kelengkapan, representativitas, dan bias data dapat memengaruhi hasil.")
    with ml5:
        st.markdown("#### 📈 Robust Forecasting")
        st.info("**Fungsi:** memproyeksikan pola jumlah kasus 14 hari ke depan dengan ensemble beberapa pendekatan forecasting.\n\n**Interpretasi:** gunakan untuk perencanaan dan pemantauan, bukan sebagai kepastian kejadian masa depan.")
        epi=build_epi(df_active)
        if st.button("Run Robust Forecast",key="run_forecast"): st.session_state["forecast_result"]=robust_forecast(epi,14)
        res=st.session_state.get("forecast_result")
        if res:
            if res.get("status")=="ok":
                a,b,c=st.columns(3); a.metric("Backtest MAE",f"{res['mae']:.2f}"); b.metric("Backtest RMSE",f"{res['rmse']:.2f}"); c.metric("Prediksi Puncak",f"{res['peak_value']:.1f}"); st.write("Model weights:",res["weights"]); st.markdown("**🧠 Forecast Insight**"); st.write(res.get("narrative","")); fig,ax=plt.subplots(figsize=(14,5)); ax.plot(res["dates"],res["forecast"],"-o",label="Ensemble forecast"); ax.fill_between(res["dates"],res["lower"],res["upper"],alpha=.15); ax.grid(alpha=.3); ax.legend(); st.pyplot(fig)
            else: st.warning(res.get("message","Forecast gagal."))
        st.warning("⚠️ **Disclaimer Forecast:** Forecast adalah estimasi statistik berdasarkan pola historis dan asumsi model, bukan kepastian kejadian di masa depan. Intervensi, perubahan perilaku, mobilitas, musim, sistem pelaporan, dan kejadian eksternal dapat menyebabkan hasil aktual berbeda dari proyeksi.")
    st.markdown("### 🛡️ General AI/ML Governance")
    st.info("Output ML SI-HIS merupakan decision-support. Model tidak boleh digunakan sebagai satu-satunya dasar keputusan klinis, penetapan KLB/wabah, atau tindakan kesehatan masyarakat tanpa verifikasi profesional. Model harus dievaluasi pada populasi dan periode yang relevan, dipantau terhadap perubahan data, dan melalui governance serta validasi yang sesuai sebelum digunakan untuk keputusan operasional.")

st.caption("SI-HIS Intelligence — ML adalah decision-support layer. Output prediktif memerlukan validasi epidemiologi/klinis dan governance sebelum digunakan untuk keputusan operasional.")
