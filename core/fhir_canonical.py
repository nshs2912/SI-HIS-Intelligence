"""Canonical HL7 FHIR R4 / SATUSEHAT mapping contracts.

This is a mapping contract, not a claim of production SATUSEHAT certification.
Terminology, profiles and mandatory elements must be validated against the
current SATUSEHAT implementation guide for the exact use case.
"""
FHIR_MAP={
"laboratory":{"resources":["Patient","Encounter","ServiceRequest","Specimen","Observation","DiagnosticReport"],"flow":"ServiceRequest -> Specimen -> Observation -> DiagnosticReport","key_paths":["Observation.subject","Observation.code","Observation.value[x]","Observation.referenceRange","Observation.interpretation","DiagnosticReport.subject","DiagnosticReport.result","DiagnosticReport.specimen"]},
"pharmacy":{"resources":["Patient","Encounter","Condition","Medication","MedicationRequest","MedicationDispense","MedicationAdministration","MedicationStatement"],"flow":"Medication + MedicationRequest -> Dispense/Administration -> Statement","key_paths":["MedicationRequest.subject","MedicationRequest.medication[x]","MedicationRequest.dosageInstruction","MedicationRequest.reasonReference","MedicationDispense.subject","MedicationAdministration.request","MedicationStatement.derivedFrom"]},
"dietitian":{"resources":["Patient","Encounter","Observation","Condition","NutritionOrder","CarePlan","Goal"],"flow":"Observation/Condition -> NutritionOrder -> Goal/CarePlan -> follow-up Observation","key_paths":["NutritionOrder.subject","NutritionOrder.encounter","NutritionOrder.dateTime","NutritionOrder.orderer","NutritionOrder.oralDiet","NutritionOrder.supplement"]},
"clinical_journey":{"resources":["Patient","Encounter","Condition","Observation","ServiceRequest","DiagnosticReport","MedicationRequest","MedicationDispense","NutritionOrder","Procedure","CarePlan","Task","RiskAssessment"],"flow":"Longitudinal linked clinical events","key_paths":["Encounter.subject","Condition.subject","Observation.subject","ServiceRequest.subject","MedicationRequest.subject","NutritionOrder.subject","Procedure.subject","CarePlan.subject","Task.for","RiskAssessment.subject"]},
"patient_summary":{"resources":["Patient","Encounter","Condition","Observation","DiagnosticReport","MedicationRequest","MedicationDispense","NutritionOrder","Procedure","CarePlan","RiskAssessment"],"flow":"Read-only aggregation of authorized source resources","key_paths":["Patient.id","Encounter.subject","Condition.subject","Observation.subject","DiagnosticReport.subject","MedicationRequest.subject","NutritionOrder.subject","RiskAssessment.subject"]},
}
COMMON={"Patient":"Master Patient Index/IHS patient identifier","Organization":"Organization IHS identifier","Practitioner":"Master Nakes Index/IHS identifier","Encounter":"Encounter ID from source service","time":"Use UTC+00 for SATUSEHAT payload timestamps"}

def canonical_mapping(module=None):
    out={"standard":"HL7 FHIR R4","target":"SATUSEHAT","common":COMMON,"modules":FHIR_MAP}
    return {"standard":out["standard"],"target":out["target"],"common":COMMON,"modules":FHIR_MAP[module]} if module else out
