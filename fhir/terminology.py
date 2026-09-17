"""Terminology and profile configuration boundary.

Keep local mappings configurable because SATUSEHAT terminology/profile rules
are use-case dependent and can change. Do not hard-code clinical codes unless
verified against the current implementation guide.
"""
FHIR_R4 = "R4"
SATUSEHAT_FHIR_BASE = "https://fhir.kemkes.go.id/r4"

RESOURCE_TYPES = [
    "Patient", "Encounter", "Condition", "Observation", "Specimen",
    "DiagnosticReport", "RiskAssessment", "Organization", "Practitioner",
    "Location", "Bundle"
]

def resource_supported(resource_type: str) -> bool:
    return resource_type in RESOURCE_TYPES
