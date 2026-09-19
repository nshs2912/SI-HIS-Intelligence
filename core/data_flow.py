"""SI-HIS individual-to-population data-flow contracts.

The analytical scope is explicit so clinical data cannot accidentally be treated as population intelligence or patient-specific clinical decision support without a declared context.
"""
from dataclasses import dataclass, asdict
from typing import Literal

AnalysisScope = Literal["individual", "population"]
IntelligencePlane = Literal["CLINICAL_SAMD_CANDIDATE", "HEALTH_INFORMATION_SYSTEM"]

@dataclass(frozen=True)
class AnalysisContext:
    scope: AnalysisScope
    plane: IntelligencePlane
    patient_identifiable: bool
    clinical_recommendation: bool = False
    requires_human_review: bool = True
    provenance_required: bool = True

def individual_context(clinical_recommendation: bool = False) -> AnalysisContext:
    return AnalysisContext("individual", "CLINICAL_SAMD_CANDIDATE", True, clinical_recommendation)

def population_context() -> AnalysisContext:
    return AnalysisContext("population", "HEALTH_INFORMATION_SYSTEM", False, False)

def validate_context(context: AnalysisContext) -> dict:
    if context.scope == "individual" and context.plane != "CLINICAL_SAMD_CANDIDATE":
        raise ValueError("Individual clinical analytics must use the clinical/SaMD candidate plane.")
    if context.scope == "population" and context.clinical_recommendation:
        raise ValueError("Population intelligence cannot emit patient-specific clinical recommendations.")
    return asdict(context)

def individual_to_population_rule() -> dict:
    return {"source":"individual","transformations":["purpose_check","authorization","data_quality","provenance","pseudonymization_or_aggregation","minimum_necessary_fields"],"destination":"population","forbidden":["silent patient-level re-identification","patient-specific diagnosis from population score","automatic treatment order"]}

def population_to_individual_rule() -> dict:
    return {"source":"population","allowed":["public-health alert","targeted investigation","screening/program prioritization","contextual risk signal for authorized clinical review"],"forbidden":["automatic individual diagnosis","automatic individual treatment"],"handoff":"clinical review with patient-specific evidence"}

def data_flow_contract() -> dict:
    return {"individual_to_population":individual_to_population_rule(),"population_to_individual":population_to_individual_rule(),"contexts":{"individual":asdict(individual_context()),"population":asdict(population_context())}}
