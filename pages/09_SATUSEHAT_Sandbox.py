import json

import streamlit as st

from interoperability.satusehat_auth import SATUSEHATAuth, SATUSEHATAuthError
from interoperability.satusehat_config import load_satusehat_config
from interoperability.satusehat_gateway import SATUSEHATGateway, SATUSEHATGatewayError

st.set_page_config(page_title="SATUSEHAT Sandbox", page_icon="🔗", layout="wide")
st.title("🔗 SATUSEHAT Integration — Sandbox")
st.caption("Transport/authentication layer for SI-HIS. Use Sandbox credentials only during development.")

cfg = load_satusehat_config(st.secrets)

c1, c2, c3 = st.columns(3)
c1.metric("Environment", cfg["environment"].upper())
c2.metric("Organization ID", "Configured" if cfg["organization_id"] else "Not configured")
c3.metric("Credentials", "Configured" if cfg["client_id"] and cfg["client_secret"] else "Not configured")

st.info("Credentials harus disimpan di Streamlit Secrets, bukan di source code/GitHub.")

with st.expander("Konfigurasi yang dibutuhkan", expanded=not (cfg["client_id"] and cfg["client_secret"])):
    st.code('''SATUSEHAT_ENVIRONMENT = "sandbox"
SATUSEHAT_CLIENT_ID = "<client-id>"
SATUSEHAT_CLIENT_SECRET = "<client-secret>"
SATUSEHAT_ORGANIZATION_ID = "<organization-ihs-number>"''', language="toml")

st.divider()
st.subheader("1. Test OAuth2 Access Token")

if st.button("🔐 Ambil Access Token", type="primary"):
    try:
        auth = SATUSEHATAuth(
            client_id=cfg["client_id"],
            client_secret=cfg["client_secret"],
            environment=cfg["environment"],
        )
        token = auth.access_token()
        st.success("OAuth2 berhasil. Access token diterima.")
        st.session_state["satusehat_auth"] = auth
        st.session_state["satusehat_token_ok"] = True
        st.caption(f"Token tersedia untuk session ini; panjang token: {len(token)} karakter.")
    except SATUSEHATAuthError as exc:
        st.error(str(exc))

st.divider()
st.subheader("2. Patient MPI — pencarian berdasarkan NIK")
nik = st.text_input("NIK", type="password", help="Gunakan data sandbox/test yang memang diizinkan untuk pengujian.")

if st.button("🔎 Cari Patient IHS"):
    if not nik.strip():
        st.warning("Masukkan NIK untuk pengujian.")
    elif "satusehat_auth" not in st.session_state:
        st.warning("Ambil access token terlebih dahulu.")
    else:
        try:
            gateway = SATUSEHATGateway(st.session_state["satusehat_auth"])
            result = gateway.get_patient_by_nik(nik.strip())
            st.json(result)
        except SATUSEHATGatewayError as exc:
            st.error(str(exc))

st.divider()
st.subheader("3. Practitioner MPI — pencarian berdasarkan NIK")
practitioner_nik = st.text_input("NIK Tenaga Kesehatan", type="password")

if st.button("🔎 Cari Practitioner IHS"):
    if not practitioner_nik.strip():
        st.warning("Masukkan NIK tenaga kesehatan untuk pengujian.")
    elif "satusehat_auth" not in st.session_state:
        st.warning("Ambil access token terlebih dahulu.")
    else:
        try:
            gateway = SATUSEHATGateway(st.session_state["satusehat_auth"])
            result = gateway.get_practitioner_by_nik(practitioner_nik.strip())
            st.json(result)
        except SATUSEHATGatewayError as exc:
            st.error(str(exc))

st.divider()
st.caption("Tahap berikutnya: validasi profile/use-case, Organization/Location, mapping Patient/Encounter/Condition/Observation, lalu transaction test di Sandbox.")
