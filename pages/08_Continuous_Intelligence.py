import streamlit as st

from core.orchestration import (
    CYCLE_STAGES,
    DATA_DOMAINS,
    INTELLIGENCE_DOMAINS,
    build_data_fusion_contract,
    build_intelligence_routing,
)

st.set_page_config(page_title="SI-HIS Continuous Intelligence", page_icon="🔄", layout="wide")

st.title("🔄 SI-HIS Continuous Intelligence")
st.caption("Continuous Health Intelligence & Orchestration")

st.info(
    "SI-HIS dirancang sebagai continuous intelligence system: data baru terus masuk, "
    "dianalisis, menghasilkan intelligence, diteruskan menjadi tindakan, outcome diukur, "
    "dan outcome kembali menjadi data untuk siklus analisis berikutnya."
)

st.markdown("## Siklus Intelligence Berkelanjutan")
cols = st.columns(len(CYCLE_STAGES))
for col, stage in zip(cols, CYCLE_STAGES):
    col.metric(stage, "✓")

st.markdown("### Arsitektur siklus")
st.code(
    """OBSERVE\n  ↓\nINTEGRATE\n  ↓\nANALYZE\n  ↓\nPREDICT\n  ↓\nDECIDE\n  ↓\nACT\n  ↓\nMEASURE OUTCOME\n  ↓\nLEARN\n  └──────────────→ OBSERVE AGAIN\n""",
    language="text",
)

st.markdown("## Tiga Domain Intelligence")
for key, label in INTELLIGENCE_DOMAINS.items():
    st.subheader(label)
    if key == "personal":
        st.write("Feedback personal untuk pasien melalui NutriMed MyLab dan partner mobile apps.")
    elif key == "clinical":
        st.write("Clinical intelligence untuk dokter dan jejaring pelayanan kesehatan, termasuk laboratory, pharmacy, dietitian, physiotherapy, rehabilitation, radiology, pathology, dan penunjang medis lainnya.")
    else:
        st.write("Population health intelligence untuk Kemenkes, Dinkes, BPJS, corporate health, dan program kesehatan masyarakat.")

st.markdown("## Data Fusion")
contract = build_data_fusion_contract()
for key, label in DATA_DOMAINS.items():
    with st.expander(label, expanded=(key == "individual_health")):
        st.write(contract[key])

st.markdown("## Intelligence Routing")
routing = build_intelligence_routing()
for key, destinations in routing.items():
    st.markdown(f"**{INTELLIGENCE_DOMAINS[key]}**")
    st.write(" → ".join(destinations))

st.markdown("## Closed-Loop Principle")
st.success(
    "Data → Intelligence → Action → Outcome → Feedback → Data. "
    "SI-HIS tidak berhenti pada dashboard atau satu kali prediction; sistem dipersiapkan "
    "untuk siklus analisis yang terus berulang."
)

st.warning(
    "Prototype note: halaman ini mendokumentasikan orchestration contract. "
    "Production operation tetap memerlukan data governance, privacy/security, model registry, "
    "clinical/public-health validation, audit trail, monitoring, dan human oversight."
)
