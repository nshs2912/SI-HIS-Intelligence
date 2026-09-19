"""Role-based, evidence-traceable patient clinical summary."""
from __future__ import annotations
import pandas as pd
from .privacy_security import AccessContext, authorize, minimize_columns
from .evidence_safety import provenance_record

ROLE_SUMMARY_FIELDS={
 "doctor":["patient_id","date","diagnosis","condition","allergy","lab","medication","procedure","encounter","outcome"],
 "dietitian":["patient_id","date","diagnosis","condition","lab","medication","weight","height","bmi","nutrition","diet","outcome"],
 "physiotherapist":["patient_id","date","diagnosis","condition","pain","mobility","rom","strength","function","procedure","rehabilitation","outcome"],
 "pharmacist":["patient_id","date","diagnosis","condition","allergy","lab","medication","dose","route","frequency","dispense","adverse_event","outcome"],
 "laboratory":["patient_id","date","diagnosis","condition","lab","specimen","result","unit","reference_range","critical_flag"],
}

def _canon_map(df):
    mapping={}
    for c in df.columns:
        s=c.lower().replace("_"," ").strip()
        if "patient" in s or s in {"id pasien","nik"}:mapping[c]="patient_id"
        elif "diagnos" in s or s=="condition":mapping[c]="diagnosis"
        elif "obat" in s or "medication" in s or "drug" in s:mapping[c]="medication"
        elif "lab" in s or "pemeriksaan" in s or "hasil" in s:mapping[c]="lab"
        elif "berat" in s or s=="weight":mapping[c]="weight"
        elif s=="bmi" or s=="imt":mapping[c]="bmi"
        elif "diet" in s or "nutri" in s:mapping[c]="nutrition"
        elif "procedure" in s or "tindakan" in s:mapping[c]="procedure"
        elif "tanggal" in s or s=="date":mapping[c]="date"
        elif "alerg" in s or "allergy" in s:mapping[c]="allergy"
        elif "nyeri" in s or "pain" in s:mapping[c]="pain"
        elif "mobil" in s:mapping[c]="mobility"
        elif "rom" in s:mapping[c]="rom"
        elif "strength" in s or "kekuatan" in s:mapping[c]="strength"
        elif "fungsi" in s or "function" in s:mapping[c]="function"
    return mapping

def summarize_patient(df, context: AccessContext):
    mapping=_canon_map(df);canonical=df.rename(columns=mapping)
    decision=authorize(context, canonical.columns)
    if not decision.allowed:return {"status":"DENIED","decision":decision.to_dict()}
    fields=[f for f in ROLE_SUMMARY_FIELDS.get(context.role.lower(),[]) if f in canonical.columns]
    minimized=minimize_columns(canonical,fields)
    if context.patient_id is not None and "patient_id" in minimized.columns:minimized=minimized[minimized["patient_id"].astype(str)==str(context.patient_id)].copy()
    timeline=minimized.copy()
    if "date" in timeline.columns:timeline["_date"]=pd.to_datetime(timeline["date"],errors="coerce");timeline=timeline.sort_values("_date").drop(columns=["_date"])
    return {"status":"OK","role":context.role,"purpose":context.purpose,"patient_id":context.patient_id,"summary":{"record_count":len(timeline),"fields":fields,"timeline":timeline},"provenance":provenance_record("Patient Clinical Summary",input_ids=timeline.index.tolist()),"guardrail":"AI-generated summary must be reviewed by the authorized professional and does not replace the source clinical record."}
