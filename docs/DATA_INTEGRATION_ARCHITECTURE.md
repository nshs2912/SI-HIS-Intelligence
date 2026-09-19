# SI-HIS Data Integration Architecture

## Purpose

SI-HIS Intelligence accepts two operating modes:

1. DEMO / Synthetic — deterministic national dummy data for presentations, development and testing.
2. FACTUAL / Operational — factual data supplied by an adapter from an existing health application, database, API or FHIR interface.

The intelligence engine must not depend on the original source application's technology.

## Canonical flow

SIMPUS / FKTP / Clinic / Hospital / RME / NutriMed Mobile / Doctor / Dietitian / Physiotherapist / Radiology / Laboratory / Pharmacy / Other services
        |
        v
SOURCE ADAPTERS
        |
        v
NORMALIZATION / DATA QUALITY
        |
        v
CANONICAL SI-HIS DATA
        |
        v
INTELLIGENCE ENGINE
TIME + PERSON + PLACE
Statistics + Spatial + EWS + KLB Signal + Forecast + ML
        |
        v
API / Dashboards / NutriMed
        |
        v
Decision + Intervention
        |
        v
Outcome / Feedback

## Adapter contract

Every operational connector should eventually implement:

- source_id
- source_system
- source_version
- extracted_at
- organization_id
- facility_id
- record_id
- patient_ihs_number when applicable
- encounter_id when applicable
- canonical clinical/geographic fields
- terminology/code system
- data quality flags
- lineage metadata

The current prototype uses CSV/Excel as the transport for factual data. This is intentional: it allows the presentation to demonstrate the same intelligence engine using factual-shaped data without hard-coding a connection to a particular vendor.

## Suggested production adapters

### SIMPUS / FKTP
Patient/encounter, diagnosis, vital signs, referral, immunization, screening and service utilization.

### Hospital / RME
Outpatient, emergency, inpatient, diagnosis, procedure, discharge, mortality and referral.

### NutriMed MyLab Mobile
Patient-generated health data, symptom diary, screening, food/nutrition, physical activity, medication adherence, telemedicine encounter and longitudinal patient journey.

### Clinical network
Doctor consultation, dietitian consultation, physiotherapy, radiology, pathology and other supporting services.

### Laboratory / LIS
Specimen, test order, result, reference range, abnormal flag and diagnostic report.

### Pharmacy / Apotek
Prescription, medication request, dispensing, medication administration and adherence signal.

## FHIR interoperability boundary

The factual adapter should not bypass the canonical SI-HIS layer. Where FHIR is available, use the FHIR resource as the interoperability transport and map it into the canonical model before intelligence processing.

Candidate resources include:

Patient, Encounter, Condition, Observation, DiagnosticReport, ServiceRequest, Procedure, MedicationRequest, MedicationDispense, MedicationAdministration, NutritionOrder, ImagingStudy, Specimen, Practitioner, Organization, Location, RiskAssessment and Task.

SATUSEHAT uses HL7 FHIR for its data model/API and organizes interoperability by service/use-case. Terminology validation is therefore part of the production adapter boundary.

## Important separation

Do not mix:

- clinical record — what happened to a patient;
- epidemiological signal — what pattern is emerging in a population;
- prediction — what the model estimates may happen;
- recommendation — what the system proposes for human review;
- legal/clinical decision — what an authorized professional or authority decides.

This separation prevents an analytical signal from being presented as an unsupported clinical or regulatory conclusion.

## Recommended next production layer

Add a durable ingestion/audit store with:

raw event -> validated event -> canonical record -> intelligence run -> model/rules version -> output -> intervention -> outcome

This creates traceability for both IT integration and epidemiological verification.
