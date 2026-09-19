"""Clinical Pharmacy Intelligence — medication safety and use analysis."""
from __future__ import annotations
import pandas as pd

ALIASES={"patient_id":["Patient ID","patient_id","ID Pasien"],"drug":["Medication","Obat","Nama Obat","Drug"],"date":["Date","Tanggal","Tanggal Resep"],"dose":["Dose","Dosis"],"frequency":["Frequency","Frekuensi"],"condition":["Diagnosis","Condition","Indication"],"lab":["Lab","Laboratory Result"]}

def _c(df,k):
    return next((c for c in ALIASES[k] if c in df.columns),None)

def medication_timeline(df):
    pid,drug,date=_c(df,"patient_id"),_c(df,"drug"),_c(df,"date")
    if not all([pid,drug,date]):return {"status":"INSUFFICIENT_DATA"}
    x=df.copy();x["_date"]=pd.to_datetime(x[date],errors="coerce");return {"status":"OK","table":x.sort_values([pid,"_date"])[[pid,drug,date]+([_c(x,"dose")] if _c(x,"dose") else [])].copy()}

def duplicate_medication_detection(df):
    pid,drug=_c(df,"patient_id"),_c(df,"drug")
    if not all([pid,drug]):return {"status":"INSUFFICIENT_DATA"}
    x=df.copy();counts=x.groupby([pid,drug]).size().reset_index(name="count");return {"status":"OK","duplicates":counts[counts["count"]>1].copy()}

def refill_adherence_signal(df):
    pid,drug,date=_c(df,"patient_id"),_c(df,"drug"),_c(df,"date")
    if not all([pid,drug,date]):return {"status":"INSUFFICIENT_DATA"}
    x=df.copy();x["_date"]=pd.to_datetime(x[date],errors="coerce");x=x.sort_values([pid,drug,"_date"]);x["gap_days"]=x.groupby([pid,drug])["_date"].diff().dt.days
    return {"status":"OK","table":x[[pid,drug,date,"gap_days"]].copy()}

def interaction_signals(df, drug_pairs=None, drug_lab_rules=None):
    drug_pairs=drug_pairs or set();drug_lab_rules=drug_lab_rules or {}
    pid,drug=_c(df,"patient_id"),_c(df,"drug")
    if not pid or not drug:return {"status":"INSUFFICIENT_DATA"}
    pairs=[]
    for patient,g in df.groupby(pid):
        drugs={str(v).strip() for v in g[drug].dropna()}
        for a,b in drug_pairs:
            if a in drugs and b in drugs:pairs.append({"patient_id":patient,"drug_a":a,"drug_b":b,"signal":"POTENTIAL_DRUG_DRUG_INTERACTION"})
    return {"status":"OK","drug_drug":pairs,"drug_lab_rules":drug_lab_rules,"note":"Interaction knowledge base must be clinically validated and versioned."}

def adverse_event_signal(df):
    candidates=[c for c in ["Adverse Event","Efek Samping","ADR","Side Effect"] if c in df.columns]
    if not candidates:return {"status":"INSUFFICIENT_DATA"}
    c=candidates[0];return {"status":"OK","events":df[df[c].notna() & df[c].astype(str).str.strip().ne("")].copy()}

def pharmacy_demand_signal(df):
    drug=_c(df,"drug");date=_c(df,"date")
    if not drug or not date:return {"status":"INSUFFICIENT_DATA"}
    x=df.copy();x["_date"]=pd.to_datetime(x[date],errors="coerce");d=x.groupby([x["_date"].dt.date,drug]).size().reset_index(name="dispense_count")
    return {"status":"OK","daily_demand":d}

def pharmacy_intelligence(df):
    return {"module":"Pharmacy Intelligence","medication_timeline":medication_timeline(df),"duplicate_medication":duplicate_medication_detection(df),"adherence_refill":refill_adherence_signal(df),"interactions":interaction_signals(df),"adverse_event":adverse_event_signal(df),"demand":pharmacy_demand_signal(df),"guardrail":"Medication signals require pharmacist/clinician review; no autonomous prescribing or medication change."}
