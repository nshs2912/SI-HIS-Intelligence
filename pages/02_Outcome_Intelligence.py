import pandas as pd
import streamlit as st
from core.analytics import generate_data_simulasi
from core.engine import SIHISIntelligenceEngine
from core.scope import QueryScope
from core.intelligence.narrative import outcome_narrative, spatial_narrative

st.set_page_config(page_title="SI-HIS Outcome Intelligence", page_icon="🧪", layout="wide")
st.title("🧪 SI-HIS — Outcome Intelligence")
st.caption("ONE OUTCOME → ONE ANALYTICAL MODEL → ONE EPIDEMIOLOGICAL INTERPRETATION")

@st.cache_data(show_spinner="Memuat synthetic dataset...")
def load_data():
    return generate_data_simulasi().copy()

try:
    df=load_data()
except Exception as exc:
    st.error(f"Dataset tidak dapat dimuat: {exc}")
    st.stop()

engine=SIHISIntelligenceEngine()
provinces=sorted(df["Provinsi"].dropna().astype(str).unique())
sel_prov=st.sidebar.selectbox("Provinsi",["Semua Provinsi"]+provinces)
base=df if sel_prov=="Semua Provinsi" else df[df["Provinsi"].astype(str).eq(sel_prov)]
districts=sorted(base["Kabupaten"].dropna().astype(str).unique())
sel_dist=st.sidebar.selectbox("Kabupaten/Kota",["Semua Kabupaten/Kota"]+districts)
base=base if sel_dist=="Semua Kabupaten/Kota" else base[base["Kabupaten"].astype(str).eq(sel_dist)]

diseases=set()
for col in ["Diagnosis Konfirm","Diagnosis Probabel","Diagnosis Suspek"]:
    if col in df:diseases.update(str(x).strip() for x in df[col].dropna().unique() if str(x).strip().lower() not in {"","nan","bukan","none","tidak ada","-"})
sel_disease=st.sidebar.selectbox("Penyakit",sorted(diseases))

scope=QueryScope(
    province=None if sel_prov=="Semua Provinsi" else sel_prov,
    district=None if sel_dist=="Semua Kabupaten/Kota" else sel_dist,
    disease=sel_disease,
    period_days=3650,
)
result=engine.analyze(df,scope=scope,include_ml=False)

if not result.get("eligible",False):
    st.warning(result.get("message","Analisis belum memenuhi prasyarat."))
    if result.get("eligibility",{}).get("reasons"):st.write(result["eligibility"]["reasons"])
    st.stop()

st.subheader("TIME + PERSON + PLACE")
show=result.get("trias_summary",{}).get("narrative")
if show:st.info(show)
if isinstance(result.get("trias_summary",{}).get("top10_province"),pd.DataFrame):st.dataframe(result["trias_summary"]["top10_province"],use_container_width=True,hide_index=True)

st.subheader("Outcome yang dianalisis")
outcomes=result.get("outcome_analyses",{})
for key in ["Penyakit","Severity","Kasus Meninggal"]:
    outcome=outcomes.get(key,{})
    with st.expander(f"Outcome: {key}",expanded=True):
        st.info(outcome_narrative(outcome))
        if not outcome.get("available",False):continue
        st.caption(f"n={outcome.get('n',0):,} | event={outcome.get('events',0):,} | event rate={outcome.get('event_rate_pct',0):.2f}%")
        analysis=outcome.get("analysis",{})
        for factor,obj in analysis.items():
            if isinstance(obj,pd.DataFrame):
                st.markdown(f"**{factor}**");st.dataframe(obj,use_container_width=True,hide_index=True);continue
            if not isinstance(obj,dict):continue
            st.markdown(f"**{factor}**")
            if "p_value" in obj:st.caption(f"Chi-square={obj.get('chi2')} | p-value={obj.get('p_value')}")
            if isinstance(obj.get("crosstab"),pd.DataFrame):st.dataframe(obj["crosstab"],use_container_width=True)
            if isinstance(obj.get("or_by_group"),pd.DataFrame) and not obj["or_by_group"].empty:st.dataframe(obj["or_by_group"],use_container_width=True,hide_index=True)

st.subheader("Spatial Evidence")
st.info(spatial_narrative(result.get("spatial")))
sp=result.get("spatial")
if isinstance(sp,pd.DataFrame) and not sp.empty:
    cols=[c for c in ["Kabupaten","Provinsi","Latitude","Longitude","Cluster","LISA_cluster","LISA_p","GiZ","Gi_p","Gi_Hotspot"] if c in sp.columns]
    if cols:st.dataframe(sp[cols].drop_duplicates(),use_container_width=True,hide_index=True)

st.caption("Catatan: dataset bersifat synthetic. Temuan statistik/spasial adalah demonstrasi metodologi, bukan gambaran kejadian epidemiologis aktual.")
