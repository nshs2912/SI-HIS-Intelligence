"""Reusable Streamlit client-portal renderer for SI-HIS."""
import pandas as pd
import streamlit as st
from core.client_intelligence import client_spec, PUSKESMAS_FLOW
from core.data_provider import load_source
from core.engine import SIHISIntelligenceEngine
from core.workforce_intelligence import build_workforce_intelligence, workforce_policy_narrative
from core.provider_intelligence import build_provider_intelligence, provider_policy_narrative, PROVIDER_TYPES
from core.disease_intelligence import classify_disease

def render_client_portal(client_name):
    spec = client_spec(client_name)
    st.title(f"🧠 {client_name}")
    st.subheader(spec.get("title", "SI-HIS Client Intelligence"))
    st.caption(f"Level: {spec.get('level','-')} | Scope: {spec.get('scope','-')}")
    st.info("SI-HIS menggunakan satu intelligence engine, lalu menyalurkan informasi sesuai skala, kewenangan, dan kebutuhan keputusan client. Output AI/ML adalah decision support; verifikasi manusia tetap diperlukan.")
    with st.expander("🎯 Pertanyaan yang dijawab dashboard", expanded=True):
        for q in spec.get("questions", []):
            st.markdown(f"- {q}")
    st.markdown("### 📦 Informasi yang disalurkan")
    for x in spec.get("outputs", []):
        st.markdown(f"- {x}")

    if client_name == "Workforce Health Intelligence":
        df, meta = load_source()
        r = build_workforce_intelligence(df, "Corporate Workforce")
        if r.get("status") == "ok":
            st.markdown(workforce_policy_narrative(r))
        else:
            st.warning(r.get("message", "Data workforce belum tersedia."))
        return

    if client_name == "Healthcare Provider Intelligence":
        df, meta = load_source()
        ptype = st.selectbox("Jenis Fasyankes", list(PROVIDER_TYPES.keys()))
        r = build_provider_intelligence(df, ptype)
        if r.get("status") == "ok":
            st.markdown(provider_policy_narrative(r))
            ps = r.get("provider_specific", {})
            if ps.get("status") == "ok":
                st.json(ps)
        else:
            st.warning(r.get("message", "Data fasyankes belum tersedia."))
        return

    df, meta = load_source()
    if df is None or df.empty:
        st.warning("Dataset belum tersedia.")
        return

    st.markdown("### 🗺️ Hierarchical Scope")
    if client_name == "Kemenkes":
        st.write("Indonesia → Provinsi → Kabupaten/Kota → agregat fasilitas")
    elif client_name == "BPJS":
        st.write("Peserta → pelayanan → rujukan → klaim → outcome → biaya")
    elif client_name == "Dinkes Provinsi":
        st.write("Provinsi → Kabupaten/Kota → Puskesmas")
    elif client_name == "Dinkes Kabupaten/Kota":
        st.write("Kabupaten/Kota → Kecamatan → Desa/Kelurahan → Puskesmas")
        st.markdown("#### 🔄 Flow Dinkes Kabupaten/Kota ↔ Puskesmas")
        for step in PUSKESMAS_FLOW:
            st.markdown(f"- {step}")
        if "Puskesmas" in df.columns:
            pks = sorted(df["Puskesmas"].dropna().astype(str).unique())
            selected = st.selectbox("Pilih Puskesmas", ["Semua Puskesmas"] + pks)
            if selected != "Semua Puskesmas":
                df = df[df["Puskesmas"].astype(str).eq(selected)].copy()
                st.success(f"Scope aktif: {selected}")
    elif client_name == "Puskesmas":
        if "Puskesmas" in df.columns:
            pks = sorted(df["Puskesmas"].dropna().astype(str).unique())
            selected = st.selectbox("Puskesmas", pks or ["-"])
            if selected != "-":
                df = df[df["Puskesmas"].astype(str).eq(selected)].copy()
        st.markdown("### 🔄 Continuous Local Intelligence")
        st.code("DATA → ANALYSIS → DETECTION → PREDICTION → VERIFICATION → INTERVENTION → OUTCOME → NEW DATA")

    engine = SIHISIntelligenceEngine()
    disease_values = set()
    for col in ["Diagnosis Konfirm", "Diagnosis Probabel", "Diagnosis Suspek"]:
        if col in df:
            disease_values.update(str(x).strip() for x in df[col].dropna().unique() if str(x).strip())
    disease = st.selectbox("Diagnosis/penyakit", ["Semua Penyakit"] + sorted(disease_values))
    if disease == "Semua Penyakit":
        r = engine.descriptive(df)
        st.metric("Total observasi", r["overview"].get("total_cases", len(df)))
        st.metric("Kasus meninggal", r["overview"].get("deaths", 0))
        st.dataframe(r.get("top10_diseases"), use_container_width=True, hide_index=True)
        return

    from core.scope import QueryScope
    scope = QueryScope(disease=disease, period_days=3650)
    r = engine.analyze(df, scope=scope, include_ml=True, mode="epidemiology")
    if not r.get("eligible"):
        st.warning("Analisis belum eligible untuk scope/data ini.")
        st.json(r.get("eligibility", {}))
        return
    st.markdown("### 📊 Intelligence Snapshot")
    ov = r.get("overview", {})
    c1,c2,c3 = st.columns(3)
    c1.metric("Kasus", int(ov.get("total_cases",0)))
    c2.metric("Meninggal", int(r.get("mortality",{}).get("deaths",0) if isinstance(r.get("mortality"),dict) else 0))
    c3.metric("Family", str(r.get("disease_intelligence",{}).get("family","-")))
    st.markdown("### 🤖 ML / Epidemiological Signals")
    for key in ["ews","rt","forecast","spatial","vulnerable","ml","continuous_intelligence"]:
        value = r.get(key)
        if value is not None:
            with st.expander(key.replace("_"," ").title()):
                if isinstance(value, pd.DataFrame):
                    st.dataframe(value, use_container_width=True, hide_index=True)
                else:
                    st.json(value if isinstance(value,(dict,list)) else str(value))
