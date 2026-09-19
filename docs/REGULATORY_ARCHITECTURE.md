# SI-HIS Regulatory Architecture

## 1. Deliberate separation

SI-HIS is split into two governance domains:

### A. Health Information System / Population Intelligence
Purpose: data integration, descriptive statistics, population surveillance, epidemiology, forecasting, spatial intelligence and public-health decision support.

This repository does **not** treat population-level outputs as a diagnosis, prescription or legal KLB declaration.

### B. Clinical Intelligence / SaMD candidate modules
Purpose is module-specific and may include laboratory analysis, medication safety, nutrition decision support, clinical journey intelligence and role-based clinical summarization.

Each module must have its own:
- intended purpose
- target user
- input/output specification
- clinical risk analysis
- validation plan
- performance monitoring
- version/change control
- human oversight
- auditability and evidence traceability
- cybersecurity/privacy controls
- regulatory assessment before clinical deployment

The repository's labels are an engineering boundary, **not a legal determination that a product is or is not SaMD**.

## 2. Patient summary

Patient Clinical Summary is separated from raw RME. The summary is generated only after authorization, purpose and minimum-necessary field filtering. Each summary carries provenance and must remain traceable to source data.

## 3. PDP/security by design

The implementation includes role-based access, purpose limitation, data minimization, pseudonymization/de-identification helpers, provenance and audit-oriented metadata. These are controls, not a certification of compliance.

## 4. Interoperability

Canonical clinical modules should map to HL7 FHIR/SATUSEHAT through the interoperability layer. Relevant resources may include Patient, Encounter, Condition, Observation, DiagnosticReport, MedicationRequest, MedicationDispense, MedicationAdministration, NutritionOrder, Procedure, RiskAssessment and Task.

## 5. Indonesian regulatory baseline to verify for deployment

The architecture should be reviewed against the current:
- UU 27/2022 Pelindungan Data Pribadi
- UU 17/2023 Kesehatan
- PP 28/2024 as implementation of UU 17/2023
- applicable Ministry of Health regulations and SATUSEHAT requirements
- applicable medical-device/SaMD requirements and conformity assessment

Regulatory status and intended-use classification must be rechecked for the actual release, jurisdiction and clinical workflow.
