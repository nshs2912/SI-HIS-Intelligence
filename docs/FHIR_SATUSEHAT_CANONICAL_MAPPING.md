# FHIR / SATUSEHAT Canonical Mapping

The canonical layer separates SI-HIS internal fields from external FHIR resources. It is designed around FHIR R4 and the SATUSEHAT resource/use-case model.

## Laboratory Intelligence
Patient → Encounter → ServiceRequest → Specimen → Observation → DiagnosticReport.
Observation carries the result; DiagnosticReport links results and specimen. LOINC/National Supporting Examination codes must be resolved through the current SATUSEHAT terminology. 

## Pharmacy Intelligence
Medication + MedicationRequest represent prescribing; MedicationDispense represents dispensing; MedicationAdministration represents administration; MedicationStatement can represent medication history/use.

## Dietitian Intelligence
NutritionOrder is the primary order resource for oral diet, supplements and enteral nutrition. Patient, Encounter and Practitioner/PractitionerRole provide context; Goal/CarePlan support longitudinal nutrition management.

## Clinical Journey
Use linked Patient/Encounter/Condition/Observation/ServiceRequest/DiagnosticReport/MedicationRequest/MedicationDispense/NutritionOrder/Procedure/CarePlan/Task/RiskAssessment resources to build the longitudinal journey. The journey engine is an analytical read model and does not replace source RME.

## Patient Clinical Summary
Read-only, role-scoped aggregation. The summary must retain source resource IDs and provenance so every displayed fact can be traced back to the originating FHIR resource.

## Mandatory implementation rules
1. Validate exact mandatory elements against the current SATUSEHAT guide for the selected use case.
2. Use IHS identifiers for patient, organization and practitioner references where required.
3. Send date-time in UTC+00 as required by SATUSEHAT.
4. Resolve terminology using the current SATUSEHAT terminology catalogue; do not hard-code an unverified code set.
5. Implement idempotency, retry, reconciliation, dead-letter handling and audit logging in the production integration service.
6. Keep the canonical model independent of a single vendor/facility schema.
