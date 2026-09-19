"""Regulatory boundary contracts for SI-HIS.

This module is intentionally descriptive: it does not determine legal
classification. It keeps population-health information-system functions
separate from clinical/SaMD candidate functions so intended purpose, risk,
validation and governance can be assessed independently.
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
    autonomous_diagnosis_or_treatment: bool = False
    human_review_required: bool = True
    traceability_required: bool = True

    def to_dict(self):
        return asdict(self)

POPULATION_BOUNDARY = RegulatoryBoundary(
    module="Population Intelligence",
    domain="HEALTH_INFORMATION_SYSTEM",
    intended_purpose="Population-level surveillance, descriptive analytics, forecasting and decision support.",
    primary_user="Kemenkes/Dinkes/facility/public-health authorized users",
    clinical_decision_support=False,
)

CLINICAL_BOUNDARIES = {
    "laboratory": RegulatoryBoundary(
        "Laboratory Intelligence","CLINICAL_SAMD_CANDIDATE",
        "Analyze laboratory data and generate laboratory review/recommendation signals.",
        "Laboratory professional/authorized clinician", True),
    "pharmacy": RegulatoryBoundary(
        "Pharmacy Intelligence","CLINICAL_SAMD_CANDIDATE",
        "Analyze medication use, interactions, safety and adverse-event signals.",
        "Pharmacist/authorized clinician", True),
    "dietitian": RegulatoryBoundary(
        "Dietitian Intelligence","CLINICAL_SAMD_CANDIDATE",
        "Analyze clinical, laboratory, medication and nutrition data for nutrition decision support.",
        "Dietitian/authorized clinician", True),
    "clinical_journey": RegulatoryBoundary(
        "Clinical Journey Intelligence","CLINICAL_SAMD_CANDIDATE",
        "Summarize longitudinal care, detect care gaps and support clinical review.",
        "Authorized healthcare professional", True),
    "patient_summary": RegulatoryBoundary(
        "Patient Clinical Summary","CLINICAL_SAMD_CANDIDATE",
        "Generate role-based, evidence-traceable summaries from authorized longitudinal patient data.",
        "Authorized healthcare professional", True),
}

def regulatory_inventory():
    return {
        "population": POPULATION_BOUNDARY.to_dict(),
        "clinical": {k:v.to_dict() for k,v in CLINICAL_BOUNDARIES.items()},
        "separation_rule": "Population intelligence is not a clinical/SaMD decision engine; clinical modules have independent intended purpose, validation, safety and change-control scope.",
    }
