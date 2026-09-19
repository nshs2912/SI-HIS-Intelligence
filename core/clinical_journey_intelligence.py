"""Longitudinal clinical journey and care-gap intelligence."""
from __future__ import annotations
import pandas as pd

EVENT_ALIASES={"date":["Date","Tanggal","Tanggal Sakit","Tanggal Pemeriksaan"],"patient":["Patient ID","patient_id","ID Pasien"],"event":["Event Type","Jenis Event","Event","Jenis Pelayanan"]}

def _c(df,k):return next((x for x in EVENT_ALIASES[k] if x in df.columns),None)

def build_patient_timeline(df, patient_id=None):
    p,d,e=_c(df,"patient"),_c(df,"date"),_c(df,"event")
    if not p or not d:return {"status":"INSUFFICIENT_DATA"}
    x=df.copy()
    if patient_id is not None:x=x[x[p].astype(str)==str(patient_id)]
    x["_date"]=pd.to_datetime(x[d],errors="coerce")
    cols=[p,d]+([e] if e else [])
    return {"status":"OK","table":x.sort_values([p,"_date"])[cols].copy()}

def detect_followup_gaps(df, expected_days=90):
    p,d=_c(df,"patient"),_c(df,"date")
    if not p or not d:return {"status":"INSUFFICIENT_DATA"}
    x=df.copy();x["_date"]=pd.to_datetime(x[d],errors="coerce");x=x.sort_values([p,"_date"]);x["gap_days"]=x.groupby(p)["_date"].diff().dt.days
    return {"status":"OK","gaps":x[x["gap_days"]>expected_days][[p,d,"gap_days"]].copy()}

def detect_care_gaps(df, expected_events=("screening","lab","encounter","prescription","follow-up")):
    e=_c(df,"event")
    if not e:return {"status":"INSUFFICIENT_DATA"}
    observed=set(df[e].astype(str).str.lower())
    return {"status":"OK","expected_events":list(expected_events),"missing_event_types":[x for x in expected_events if x.lower() not in observed]}

def treatment_response_loop(df):
    p,e=_c(df,"patient"),_c(df,"event")
    if not p:return {"status":"INSUFFICIENT_DATA"}
    return {"status":"OK","message":"Response loop requires linked intervention, follow-up outcome and relevant measurement IDs; no causal conclusion is inferred."}

def clinical_journey_intelligence(df, patient_id=None):
    return {"module":"Clinical Journey Intelligence","timeline":build_patient_timeline(df,patient_id),"followup_gaps":detect_followup_gaps(df),"care_gaps":detect_care_gaps(df),"treatment_response":treatment_response_loop(df),"guardrail":"Journey intelligence identifies missing/changed care events; it does not diagnose or autonomously prescribe."}
