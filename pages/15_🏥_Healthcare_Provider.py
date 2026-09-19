import streamlit as st
from core.client_portal import render_client_portal
st.set_page_config(page_title="SI-HIS — Healthcare Provider", page_icon="🏥", layout="wide")
render_client_portal("Healthcare Provider Intelligence")
