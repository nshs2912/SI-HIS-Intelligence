"""Canonical SI-HIS -> HL7 FHIR R4 mapping helpers.

This module creates FHIR-shaped resources without claiming SATUSEHAT submission
compliance. Use the current SATUSEHAT use-case profiles/terminologies before
production submission.
"""
from datetime import timezone
from typing import Any, Dict, Optional
import pandas as pd

FHIR_KEMKES = "https://fhir.kemkes.go.id/r4"

def _utc_iso(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Jakarta")
    return ts.tz_convert(timezone.utc).isoformat()

def patient(row: Dict[str, Any], ihs_number: Optional[str] = None) -> Dict[str, Any]:
    identifier = ihs_number or str(row.get("Patient_IHS_Number") or row.get("patient_ihs_number") or row.get("Nama", ""))
    gender = str(row.get("Jenis Kelamin", "")).lower()
    gender_code = "male" if gender in {"laki-laki", "laki laki", "male"} else "female" if gender in {"perempuan", "female"} else "unknown"
    resource = {
        "resourceType": "Patient",
        "identifier": [{"use": "official", "value": identifier}],
        "gender": gender_code,
    }
    name = str(row.get("Nama", "")).strip()
    if name:
        resource["name"] = [{"use": "official", "text": name}]
    return resource

def encounter(row: Dict[str, Any], patient_ref: str, encounter_id: Optional[str] = None) -> Dict[str, Any]:
    started = _utc_iso(row.get("Tanggal Sakit"))
    resource = {
        "resourceType": "Encounter",
        "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "ambulatory"},
        "subject": {"reference": patient_ref},
    }
    if encounter_id:
        resource["id"] = encounter_id
    if started:
        resource["period"] = {"start": started}
    return resource

def condition(row: Dict[str, Any], patient_ref: str, encounter_ref: Optional[str] = None) -> Dict[str, Any]:
    diagnosis = str(row.get("Diagnosis Konfirm") or row.get("Diagnosis Probabel") or row.get("Diagnosis Suspek") or "").strip()
    resource = {
        "resourceType": "Condition",
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
        "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed" if int(row.get("Is_Konfirm", 0)) == 1 else "provisional"}]},
        "subject": {"reference": patient_ref},
    }
    if diagnosis:
        resource["code"] = {"text": diagnosis}
    if encounter_ref:
        resource["encounter"] = {"reference": encounter_ref}
    return resource

def observation(row: Dict[str, Any], patient_ref: str, encounter_ref: Optional[str] = None) -> Dict[str, Any]:
    resource = {
        "resourceType": "Observation",
        "status": "final",
        "subject": {"reference": patient_ref},
    }
    if encounter_ref:
        resource["encounter"] = {"reference": encounter_ref}
    return resource

def risk_assessment(row: Dict[str, Any], patient_ref: str, risk_probability: float, outcome: str, method: str, encounter_ref: Optional[str] = None) -> Dict[str, Any]:
    p = max(0.0, min(1.0, float(risk_probability)))
    qualitative = "high" if p >= 0.66 else "moderate" if p >= 0.33 else "low"
    resource = {
        "resourceType": "RiskAssessment",
        "status": "final",
        "method": {"text": method},
        "code": {"text": "SI-HIS Machine Learning Risk Assessment"},
        "subject": {"reference": patient_ref},
        "prediction": [{
            "outcome": {"text": outcome},
            "probabilityDecimal": round(p, 6),
            "qualitativeRisk": {"text": qualitative},
            "rationale": "Model prediction from SI-HIS analytics/ML pipeline; requires clinical or public-health review before action."
        }]
    }
    if encounter_ref:
        resource["encounter"] = {"reference": encounter_ref}
    occurred = _utc_iso(row.get("Tanggal Sakit"))
    if occurred:
        resource["occurrenceDateTime"] = occurred
    return resource

def bundle(resources, bundle_type: str = "collection") -> Dict[str, Any]:
    return {
        "resourceType": "Bundle",
        "type": bundle_type,
        "entry": [{"resource": r} for r in resources if r]
    }
