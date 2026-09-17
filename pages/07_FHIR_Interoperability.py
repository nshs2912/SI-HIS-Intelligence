import json
import uuid

import pandas as pd
import streamlit as st

from core.analytics import REQUIRED_COLUMNS, generate_data_simulasi
from fhir.mapper import patient, encounter, condition, observation, risk_assessment
from fhir.validator import validate_bundle

st.set_page_config(page_title="SI-HIS FHIR Interoperability", page_icon="🔗", layout="wide")

st.title("🔗 SI-HIS FHIR Interoperability")
st.caption("FHIR R4 export layer untuk SI-HIS — development/sandbox ready, bukan klaim kepatuhan SATUSEHAT production.")

st.info(
    "SATUSEHAT menggunakan HL7 FHIR sebagai standar data model dan API. "
    "Halaman ini membuat FHIR-shaped JSON dari canonical SI-HIS data dan memvalidasi struktur dasar. "
    "Profile, terminology, IHS Number, Organization ID, Practitioner ID, dan use-case specific validation "
    "tetap wajib dipenuhi sebelum submission production."
)

mode = st.radio("Sumber data", ["Data simulasi", "Upload CSV/Excel"], horizontal=True)

if mode == "Data simulasi":
    df = generate_data_simulasi()
else:
    upload = st.file_uploader("Upload dataset surveilans", type=["csv", "xlsx"])
    if upload is None:
        st.warning("Upload dataset terlebih dahulu.")
        st.stop()
    try:
        df = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    except Exception as exc:
        st.error(f"Gagal membaca dataset: {exc}")
        st.stop()

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"Kolom wajib belum lengkap: {missing}")
    st.stop()

st.success(f"Dataset siap: {len(df):,} baris")

idx = st.number_input("Pilih nomor baris pasien", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1)
row = df.iloc[int(idx)].to_dict()

left, right = st.columns(2)
with left:
    st.markdown("### Identitas canonical SI-HIS")
    st.write({
        "Nama": row.get("Nama"),
        "Umur": row.get("Umur"),
        "Jenis Kelamin": row.get("Jenis Kelamin"),
        "Tanggal Sakit": str(row.get("Tanggal Sakit")),
        "Diagnosis Konfirm": row.get("Diagnosis Konfirm"),
        "Is_Konfirm": row.get("Is_Konfirm"),
    })
with right:
    st.markdown("### IHS Number")
    ihs = st.text_input(
        "Patient IHS Number (opsional untuk prototype)",
        value=str(row.get("Patient_IHS_Number") or ""),
        help="Untuk production, gunakan IHS Number resmi dari Master Patient Index; jangan gunakan nama pasien sebagai identifier production.",
    )
    encounter_id = st.text_input("Encounter resource ID (opsional)", value="")

patient_id = ihs.strip() or f"demo-{uuid.uuid4().hex[:12]}"
patient_resource = patient(row, ihs_number=patient_id)
patient_resource["id"] = f"patient-{uuid.uuid4().hex[:12]}"
patient_ref = f"Patient/{patient_resource['id']}"

encounter_resource = encounter(row, patient_ref, encounter_id=encounter_id.strip() or None)
encounter_resource["id"] = encounter_resource.get("id") or f"enc-{uuid.uuid4().hex[:12]}"
encounter_ref = f"Encounter/{encounter_resource['id']}"

condition_resource = condition(row, patient_ref, encounter_ref)
condition_resource["id"] = f"condition-{uuid.uuid4().hex[:12]}"

observation_resource = observation(row, patient_ref, encounter_ref)
observation_resource["id"] = f"observation-{uuid.uuid4().hex[:12]}"
observation_resource["code"] = {"text": "SI-HIS epidemiology observation"}
observation_resource["valueString"] = (
    f"Konfirmasi={int(row.get('Is_Konfirm', 0))}; "
    f"Rawat={row.get('Status Penderita', '')}; "
    f"Meninggal={int(row.get('Is_Meninggal', 0))}"
)

# Prototype risk signal. This is a transparent rule-based demonstration until a validated model artifact is supplied.
age = pd.to_numeric(pd.Series([row.get("Umur")]), errors="coerce").fillna(0).iloc[0]
comorbid = str(row.get("Status Komorbid", "")).lower() in {"ya", "yes", "true", "1"}
inpatient = str(row.get("Status Penderita", "")).lower() == "rawat inap"
risk_probability = min(0.95, 0.15 + (0.25 if age >= 60 else 0) + (0.20 if comorbid else 0) + (0.25 if inpatient else 0))

risk_resource = risk_assessment(
    row,
    patient_ref,
    risk_probability=risk_probability,
    outcome="SI-HIS prototype severity risk",
    method="Transparent prototype rule; replace with validated ML model artifact before production",
    encounter_ref=encounter_ref,
)
risk_resource["id"] = f"risk-{uuid.uuid4().hex[:12]}"
risk_resource["condition"] = {"reference": f"Condition/{condition_resource['id']}"}

resources = [patient_resource, encounter_resource, condition_resource, observation_resource, risk_resource]
bundle = {
    "resourceType": "Bundle",
    "type": "collection",
    "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
    "total": len(resources),
    "entry": [
        {
            "fullUrl": f"urn:uuid:{resource['id']}",
            "resource": resource,
        }
        for resource in resources
    ],
}

errors = validate_bundle(bundle)

st.markdown("---")
st.subheader("FHIR R4 Bundle")
if errors:
    st.error("Validasi struktur dasar menemukan masalah:")
    for error in errors:
        st.write(f"- {error}")
else:
    st.success("✅ Struktur dasar Bundle dan resource lulus validator internal SI-HIS.")

st.code(json.dumps(bundle, ensure_ascii=False, indent=2), language="json")

st.download_button(
    "⬇️ Download FHIR Bundle JSON",
    data=json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="sihis-fhir-bundle.json",
    mime="application/fhir+json",
    use_container_width=True,
)

with st.expander("ℹ️ Production checklist"):
    st.markdown(
        """
- Gunakan Patient IHS Number resmi dari Master Patient Index.
- Pastikan Organization, Location, Practitioner/PractitionerRole sesuai onboarding.
- Terapkan profile dan terminology sesuai use case SATUSEHAT.
- Gunakan timestamp UTC+00 pada payload.
- Validasi terhadap Implementation Guide/profile yang berlaku.
- Simpan Client ID/Client Secret hanya di secret manager; jangan di source code atau dataset.
- Submission ke SATUSEHAT dilakukan melalui OAuth2 + Bearer token pada environment Sandbox terlebih dahulu.
- Bundle `collection` pada halaman ini adalah paket interoperabilitas/export; bukan bukti bahwa payload sudah siap dikirim ke production.
        """
    )
