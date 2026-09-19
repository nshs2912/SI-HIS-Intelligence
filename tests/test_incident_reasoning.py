import pandas as pd
from core.incident_reasoning import (
    temporal_scan, finest_spatiotemporal_cluster,
    symptom_syndrome_clustering, exposure_attack_rate,
    facility_surge, incident_reasoning_v2,
)

def df():
    t=pd.Timestamp("2026-09-19 08:00")
    return pd.DataFrame({
        "Tanggal Onset":[t+pd.Timedelta(minutes=i*5) for i in range(12)],
        "Tanggal Pemeriksaan":[t+pd.Timedelta(minutes=i*5+20) for i in range(12)],
        "Puskesmas":["Puskesmas A"]*8+["Puskesmas B"]*4,
        "Kecamatan":["Kec A"]*12,
        "Gejala Utama":["mual","muntah","mual","muntah","diare","mual","muntah","mual","demam","mual","muntah","mual"],
        "Status Paparan":["ya"]*8+["tidak"]*4,
        "Is_Case":[1]*12,
    })

def test_temporal_scan():
    x=temporal_scan(df())
    assert x["status"]=="ok"
    assert x["best"]["cases"]>=5

def test_finest_spatiotemporal():
    x=finest_spatiotemporal_cluster(df())
    assert x["level"]=="Puskesmas"
    assert x["best_cluster"]["unit"]=="Puskesmas A"

def test_symptom_cluster():
    x=symptom_syndrome_clustering(df())
    assert x["status"]=="ok"

def test_attack_rate_requires_denominators():
    x=exposure_attack_rate(df())
    assert x["status"]=="ok"
    assert x["exposed_n"]==8 and x["unexposed_n"]==4

def test_facility_surge():
    x=facility_surge(df())
    assert x["status"]=="ok"

def test_incident_reasoning_guardrail():
    x=incident_reasoning_v2(df(),{"signals":["POINT_SOURCE_COMMON_EXPOSURE"]})
    assert x["status"]=="SIGNAL"
    assert "KLB legal" in x["guardrail"]
