"""Regulatory boundary contracts for SI-HIS.

Engineering/governance boundary only; not a legal classification. Population-health intelligence is separated from individual clinical decision support whose intended purpose may place it in a SaMD/medical-device pathway.
"""
from dataclasses import dataclass, asdict
from typing import Literal

Domain = Literal["HEALTH_INFORMATION_SYSTEM", "CLINICAL_SAMD_CANDIDATE"]

@dataclass(frozen=True)
class RegulatoryBoundary:
    module: str
    domain: Domain
    intended_purpose: str
    primary_user: str
    clinical_decision_support: bool
    individual_patient_output: bool = False
    diagnostic_or_treatment_recommendation: bool = False
    autonomous_diagnosis_or_treatment: bool = False
    human_review_required: bool = True
    traceability_required: bool = True

    def to_dict(self):
        return asdict(self)

POPULATION_BOUNDARY = RegulatoryBoundary(
    module="Population Intelligence",
    domain="HEALTH_INFORMATION_SYSTEM",
    intended_purpose="Population-level surveillance, epidemiology, forecasting, spatial intelligence and public-health decision support.",
    primary_user="Kemenkes/Dinkes/facility/public-health authorized users",
    clinical_decision_support=False,
)

CLINICAL_BOUNDARIES = {
    "laboratory": RegulatoryBoundary(
        "Laboratory Intelligence","CLINICAL_SAMD_CANDIDATE",
        "Analyze individual laboratory data and generate patient-specific clinical review or diagnostic decision-support recommendations.",
        "Laboratory professional/authorized clinician", True,
        individual_patient_output=True, diagnostic_or_treatment_recommendation=True),
    "pharmacy": RegulatoryBoundary(
        "Pharmacy Intelligence","CLINICAL_SAMD_CANDIDATE",
        "Analyze individual medication data for patient-specific medication safety, interaction, adherence or adverse-event decision support.",
        "Pharmacist/authorized clinician", True,
        individual_patient_output=True, diagnostic_or_treatment_recommendation=True),
    "dietitian": RegulatoryBoundary(
        "Dietitian Intelligence","CLINICAL_SAMD_CANDIDATE",
        "Analyze individual clinical, laboratory, medication and nutrition data for patient-specific nutrition decision support.",
        "Dietitian/authorized clinician", True,
        individual_patient_output=True, diagnostic_or_treatment_recommendation=True),
    "clinical_journey": RegulatoryBoundary(
        "Clinical Journey Intelligence","CLINICAL_SAMD_CANDIDATE",
        "Interpret an individual longitudinal clinical journey to support patient-specific diagnosis, follow-up, risk or treatment decisions.",
        "Authorized healthcare professional", True,
        individual_patient_output=True, diagnostic_or_treatment_recommendation=True),
    "patient_summary": RegulatoryBoundary(
        "Patient Clinical Summary","CLINICAL_SAMD_CANDIDATE",
        "Generate an evidence-traceable individual patient summary for clinical decision support.",
        "Authorized healthcare professional", True,
        individual_patient_output=True),
}

def regulatory_inventory():
    return {
        "population": POPULATION_BOUNDARY.to_dict(),
        "clinical": {k:v.to_dict() for k,v in CLINICAL_BOUNDARIES.items()},
        "separation_rule": (
            "Population intelligence remains a Health Information System/public-health surveillance function. Individual-patient analytics that make or support diagnostic, treatment, triage, prognosis or other clinical recommendations are routed to the clinical/SaMD candidate boundary and require independent intended-purpose, risk, validation, safety and change-control assessment."
        ),
    }
