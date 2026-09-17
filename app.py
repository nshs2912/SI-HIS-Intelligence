import warnings
import folium
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.analytics import REQUIRED_COLUMNS, generate_excel_template, get_wib_time, generate_data_simulasi
from core.engine import SIHISIntelligenceEngine
from core.scope import QueryScope

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SI-HIS Intelligence", page_icon="🧠", layout="wide")
st.markdown("""<div style="background:#f0e68c;padding:18px 22px;border-radius:12px;border:2px solid #d4c886;"><h1 style="margin:0;color:#1e3a8a;font-size:1.7rem;">SI-HIS — Smart Integrated Health Intelligence System</h1><p style="margin:6px 0 0;color:#475569;font-weight:600;">Early Detection, Smarter Intervention</p></div>""", unsafe_allow_html=True)
st.caption(f"🕒 Waktu Sistem: {get_wib_time()['full']}")
engine=SIHISIntelligenceEngine()

st.sidebar.header("⚙️ Panel Kontrol & Filter")
source=st.sidebar.radio("Sumber Data",["Gunakan Data Simulasi (AI-Ready)","Upload File Excel/CSV Custom"])
if source.startswith("Gunakan"):
    df_raw=generate_data_simulasi().copy(); st.sidebar.success(f"✅ {len(df_raw):,} data dimuat.")
else:
    f=st.sidebar.file_uploader("Upload File Kasus",type=["xlsx","csv"])
    if f is None: st.info("Upload file kasus untuk memulai."); st.stop()
    try: df_raw=pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
    except Exception as exc: st.error(f"Error membaca file: {exc}"); st.stop()
    missing=[c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing: st.error(f"Kolom wajib kurang: {missing}"); st.stop()
try: st.sidebar.download_button("📥 Download Template Excel Standard",generate_excel_template(),"Template_Data_Surveilans.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
except Exception: pass

# -----------------------------------------------------------------------------
# THREE FILTERS — ALWAYS PRESENT
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Filter Analisis Epidemiologi")
provinces=sorted(df_raw["Provinsi"].dropna().astype(str).unique()) if "Provinsi" in df_raw else []
sel_prov=st.sidebar.selectbox("1. Provinsi",["Semua Provinsi"]+provinces)
dfp=df_raw if sel_prov=="Semua Provinsi" else df_raw[df_raw["Provinsi"].astype(str).eq(sel_prov)]
districts=sorted(dfp["Kabupaten"].dropna().astype(str).unique()) if "Kabupaten" in dfp else []
sel_kab=st.sidebar.selectbox("2. Kabupaten/Kota",["Semua Kabupaten/Kota"]+districts)
dfk=dfp if sel_kab=="Semua Kabupaten/Kota" else dfp[dfp["Kabupaten"].astype(str).eq(sel_kab)]

disease_values=set()
for col in ["Diagnosis Konfirm","Diagnosis Probabel","Diagnosis Suspek"]:
    if col in df_raw.columns:
        disease_values.update(str(x).strip() for x in df_raw[col].dropna().unique() if str(x).strip().lower() not in {"","nan","bukan","none","tidak ada","-"})
diseases=sorted(disease_values)
sel_disease=st.sidebar.selectbox("3. Diagnosis Penyakit",["Semua Penyakit"]+diseases)

# Optional drill-down geography. These never change the three primary filter semantics.
with st.sidebar.expander("Drill-down wilayah (opsional)"):
    kecs=sorted(dfk["Kecamatan"].dropna().astype(str).unique()) if "Kecamatan" in dfk else []
    sel_kec=st.selectbox("Kecamatan",["Semua Kecamatan"]+kecs)
    dbase=dfk if sel_kec=="Semua Kecamatan" else dfk[dfk["Kecamatan"].astype(str).eq(sel_kec)]
    villages=sorted(dbase["Desa/Kelurahan"].dropna().astype(str).unique()) if "Desa/Kelurahan" in dbase else []
    sel_desa=st.selectbox("Desa/Kelurahan",["Semua Desa/Kelurahan"]+villages)
    vbase=dbase if sel_desa=="Semua Desa/Kelurahan" else dbase[dbase["Desa/Kelurahan"].astype(str).eq(sel_desa)]
    pusk=sorted(vbase["Puskesmas"].dropna().astype(str).unique()) if "Puskesmas" in vbase else []
    sel_pusk=st.selectbox("Puskesmas",["Semua Puskesmas"]+pusk)

include_ml=st.sidebar.checkbox("Aktifkan ML layer",False)

# Primary scope dataframe. Disease is intentionally resolved by engine, not QueryScope.
scope=QueryScope(
    province=None if sel_prov=="Semua Provinsi" else sel_prov,
    district=None if sel_kab=="Semua Kabupaten/Kota" else sel_kab,
    kecamatan=None if sel_kec=="Semua Kecamatan" else sel_kec,
    village=None if sel_desa=="Semua Desa/Kelurahan" else sel_desa,
    puskesmas=None if sel_pusk=="Semua Puskesmas" else sel_pusk,
    disease=None if sel_disease=="Semua Penyakit" else sel_disease,
    period_days=3650,
)

# Disease is the switch between descriptive and disease-specific epidemiology.
if sel_disease=="Semua Penyakit":
    result=engine.descriptive(dfk if (sel_prov!="Semua Provinsi" or sel_kab!="Semua Kabupaten/Kota") else df_raw)
    geographic_label="Indonesia" if sel_prov=="Semua Provinsi" and sel_kab=="Semua Kabupaten/Kota" else (sel_kab if sel_kab!="Semua Kabupaten/Kota" else sel_prov)
    st.markdown(f"## 📊 Analisis Deskriptif — {geographic_label}")
    st.caption("Semua Penyakit → hanya analisis deskriptif. Angka kematian gabungan tidak disebut CFR.")
    ov=result["overview"]
    a,b=st.columns(2); a.metric("Total Kunjungan Pasien",f"{ov['total_cases']:,}"); b.metric("Kasus Meninggal",f"{ov['deaths']:,}")
    st.markdown("### 🏆 10 Besar Penyakit")
    st.dataframe(result["top10_diseases"],use_container_width=True,hide_index=True)
    c1,c2=st.columns(2)
    with c1: st.markdown("### Distribusi Penyakit"); st.dataframe(result["disease_distribution"],use_container_width=True,hide_index=True)
    with c2: st.markdown("### Distribusi Jenis Kelamin"); st.dataframe(result["sex_distribution"],use_container_width=True,hide_index=True)
    c3,c4=st.columns(2)
    with c3: st.markdown("### Distribusi Kelompok Umur"); st.dataframe(result["age_distribution"],use_container_width=True,hide_index=True)
    with c4: st.markdown("### Distribusi Kabupaten/Kota"); st.dataframe(result["district_distribution"].head(50),use_container_width=True,hide_index=True)
    st.info("Mode deskriptif tidak menjalankan bivariat, multivariat, faktor risiko, EWS/KLB, DBSCAN, episentrum, kurva epidemik, forecast, atau prediksi penyakit.")
    st.caption("Catatan: CFR pada tabel 10 besar dihitung per penyakit; bukan CFR gabungan seluruh penyakit.")
else:
    # Disease-specific mode: Indonesia / Province / District are all valid analytical scopes.
    result=engine.analyze(df_raw,scope=scope,include_ml=include_ml,mode="epidemiology")
    geographic_label=sel_kab if sel_kab!="Semua Kabupaten/Kota" else (sel_prov if sel_prov!="Semua Provinsi" else "Indonesia")
    st.markdown(f"## 🧬 Analisis Epidemiologi — {sel_disease}")
    st.caption(f"Scope: **{sel_disease} — {geographic_label}** | TIME + PERSON + PLACE")
    if not result.get("eligible",False):
        st.warning("Analisis epidemiologi belum dapat dijalankan.")
        for r in result.get("eligibility",{}).get("reasons",[]): st.write("• "+r)
        st.stop()
    ov=result["overview"]
    # Disease-specific denominator: CFR is valid here.
    mortality=result.get("mortality")
    deaths=0
    if isinstance(mortality,dict): deaths=mortality.get("deaths",mortality.get("meninggal",0)) or 0
    elif isinstance(mortality,pd.DataFrame) and "Meninggal" in mortality.columns: deaths=int(mortality["Meninggal"].sum())
    total=ov["total_cases"]; cfr=(deaths/total*100) if total else 0
    a,b,c=st.columns(3); a.metric(f"Total {sel_disease}",f"{total:,}"); b.metric(f"Meninggal {sel_disease}",f"{deaths:,}"); c.metric("CFR",f"{cfr:.2f}%")
    if result.get("eligibility",{}).get("warnings"):
        for w in result["eligibility"]["warnings"]: st.warning(w)
    tabs=st.tabs(["📊 TIME + PERSON + PLACE","🧪 Faktor Risiko","🚨 EWS/KLB","🗺️ Spatial/DBSCAN","📈 Kurva & Forecast","👥 Vulnerable","🧠 AI/ML"])
    with tabs[0]:
        st.markdown("### PLACE"); st.dataframe(result.get("place",pd.DataFrame()),use_container_width=True,hide_index=True)
        st.markdown("### PERSON")
        person=result.get("person",{})
        if isinstance(person,dict):
            for k,v in person.items():
                st.markdown(f"**{k}**"); st.dataframe(v,use_container_width=True,hide_index=True) if isinstance(v,pd.DataFrame) else st.write(v)
        st.markdown("### TIME")
        t=result.get("time")
        if isinstance(t,pd.DataFrame) and not t.empty: st.line_chart(t.set_index("Tanggal Sakit")["Jumlah Kasus"]); st.dataframe(t,use_container_width=True,hide_index=True)
        st.markdown("### MORTALITY / CFR"); st.write(mortality)
    with tabs[1]:
        r=result.get("risk_factors")
        if isinstance(r,dict):
            for k,v in r.items(): st.markdown(f"#### {k}"); st.dataframe(v,use_container_width=True,hide_index=True) if isinstance(v,pd.DataFrame) else st.write(v)
        else: st.write(r)
        st.caption("Asosiasi/OR bukan bukti kausalitas tanpa desain dan validasi epidemiologis yang sesuai.")
    with tabs[2]:
        st.json(result.get("ews",{})); st.markdown("### Interpretasi temporal disease-aware"); st.info(result.get("temporal_interpretation",{}).get("interpretation","Interpretasi temporal tersedia.")); st.write("Rₜ:",result.get("rt"))
    with tabs[3]:
        spatial=result.get("spatial")
        if isinstance(spatial,pd.DataFrame):
            st.dataframe(spatial,use_container_width=True,hide_index=True)
            geo=spatial.dropna(subset=["Latitude","Longitude"]) if {"Latitude","Longitude"}.issubset(spatial.columns) else pd.DataFrame()
            if not geo.empty:
                m=folium.Map(location=[float(geo.Latitude.mean()),float(geo.Longitude.mean())],zoom_start=9)
                for _,row in geo.head(500).iterrows(): folium.CircleMarker([float(row.Latitude),float(row.Longitude)],radius=4,popup=f"{row.get('Desa/Kelurahan','')} | Cluster {row.get('Cluster','')}").add_to(m)
                st_folium(m,width=None,height=500)
        st.markdown("### Episentrum"); st.dataframe(result.get("epicenters",pd.DataFrame()),use_container_width=True,hide_index=True)
    with tabs[4]:
        st.write(result.get("epidemic_curve_classification")); t=result.get("time")
        if isinstance(t,pd.DataFrame) and not t.empty: st.line_chart(t.set_index("Tanggal Sakit")["Jumlah Kasus"])
        st.write("Forecast:",result.get("forecast")); waves=result.get("waves",[])
        if waves: st.dataframe(pd.DataFrame(waves),use_container_width=True,hide_index=True)
    with tabs[5]:
        v=result.get("vulnerable")
        st.dataframe(pd.DataFrame(v),use_container_width=True,hide_index=True) if isinstance(v,list) else st.write(v)
    with tabs[6]:
        if include_ml: st.json({k:v for k,v in result.get("ml",{}).items() if not isinstance(v,pd.DataFrame)})
        else: st.info("ML layer tidak diaktifkan.")

st.caption("SI-HIS Intelligence — disease-specific epidemiology is activated by Diagnosis Penyakit; geographic scope may be Indonesia, Province, or Kabupaten/Kota.")
