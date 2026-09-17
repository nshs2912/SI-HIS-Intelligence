"""Lightweight structural FHIR validator for SI-HIS development.

This is intentionally not a substitute for validation against the exact
SATUSEHAT Implementation Guide/profile and terminology required by a use case.
"""
REQUIRED_BY_RESOURCE = {
    "Patient": ["resourceType"],
    "Encounter": ["resourceType", "status", "class", "subject"],
    "Condition": ["resourceType", "subject"],
    "Observation": ["resourceType", "status", "subject"],
    "RiskAssessment": ["resourceType", "status", "subject", "prediction"],
    "Bundle": ["resourceType", "type", "entry"],
}

def validate_resource(resource):
    errors = []
    if not isinstance(resource, dict):
        return ["Resource harus berupa object/dict JSON."]
    resource_type = resource.get("resourceType")
    if not resource_type:
        return ["resourceType wajib ada."]
    for field in REQUIRED_BY_RESOURCE.get(resource_type, ["resourceType"]):
        if field not in resource or resource[field] in (None, "", []):
            errors.append(f"{resource_type}.{field} belum terisi")
    return errors

def validate_bundle(bundle):
    errors = validate_resource(bundle)
    if bundle.get("resourceType") == "Bundle":
        for i, entry in enumerate(bundle.get("entry", [])):
            resource = entry.get("resource") if isinstance(entry, dict) else None
            if resource:
                errors.extend([f"entry[{i}]: {e}" for e in validate_resource(resource)])
    return errors
