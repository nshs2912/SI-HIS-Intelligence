# SI-HIS — Smart Integrated Health Intelligence System
## Master Concept, Algorithms, Statistics, ML, Personal Health Data Ingress and Client Information Delivery

**Repository:** `nshs2912/SI-HIS-Intelligence`  
**Product ecosystem:** SI-HIS + NutriMed MyLab  
**Version:** 1.1 — September 2026

## 1. Executive Concept

SI-HIS is a data-, statistics-, AI- and machine-learning-enabled health intelligence architecture that transforms longitudinal health data into decision support appropriate to the user's scale and authority.

Core loop:

**DATA → ANALYSIS → DETECTION → PREDICTION → RECOMMENDATION → INTERVENTION → OUTCOME → NEW DATA → CONTINUOUS LEARNING**

SI-HIS does not replace clinicians, epidemiologists, health authorities, payers, or corporate governance. AI/ML outputs are decision support and must be interpreted with data quality, denominators, uncertainty, model validation and human verification.

## 2. SI-HIS and NutriMed MyLab

SI-HIS is the technology/intelligence layer. NutriMed MyLab is the Personal Health Companion and longitudinal **Personal Health Journey** layer.

NutriMed MyLab is not merely a conventional health application. It is intended to become the personal interface connecting prevention, screening, symptoms, FKTP, laboratory, FKRTL referral, therapy, pharmacy, nutrition, follow-up and outcomes.

The long-term vision is to evolve the mobile application into a **virtual holographic personal health assistant**. This is a future development direction, not a claim about current product capability. The envisioned assistant would support natural dialogue, longitudinal context, authorized health-information explanation, reminders, preventive coaching and service orchestration.

The assistant remains an intelligent companion and orchestration interface, not an autonomous replacement for clinicians.

## 3. Core Architecture

```
DATA SOURCES
   ↓
FHIR / SATUSEHAT / API / ETL
   ↓
DATA QUALITY
   ↓
STATISTICS
   ↓
AI / ML
   ↓
SIGNAL FUSION / INTELLIGENCE ORCHESTRATOR
   ↓
CLIENT INFORMATION DELIVERY
   ↓
HUMAN VERIFICATION / DECISION SUPPORT
   ↓
ACTION
   ↓
OUTCOME
   ↓
NEW DATA → CONTINUOUS LEARNING
```

## 4. Personal Health Data Ingress

The personal-health layer must accept authorized data from multiple sources:

1. **FKTP** — primary-care facilities such as Puskesmas and applicable clinics/practices.
2. **FKRTL** — referral facilities, particularly hospitals and other applicable advanced-care providers.
3. **NutriMed MyLab Mobile App** — self-reported observations, symptoms, nutrition, lifestyle, medication adherence, appointments, consultations, goals and follow-up.
4. **Partner Patient Mobile Apps** — authorized external digital-health applications and their permitted health events.
5. **Laboratory systems**
6. **Pharmacy systems**
7. **Corporate MCU/K3 systems**
8. **Payer/claims systems**
9. **Other authorized environmental, One Health or device sources**

Every important event should retain provenance: clinical/facility-originated, laboratory, pharmacy, payer, patient self-reported, device-generated or partner-app. Self-reported data must not be silently converted into a clinically confirmed diagnosis.

## 5. Personal Health Data Flow

```
FKTP ───────────────┐
FKRTL ──────────────┤
Laboratory ─────────┤
Pharmacy ───────────┤
NutriMed MyLab ─────┤
Partner Mobile Apps ┤
Corporate / MCU ────┤
Other Authorized ───┘
                     │
                     ▼
              SI-HIS INGESTION
                     │
       Authentication / Authorization
                     │
              Identity Resolution
                     │
              Patient / MPI Mapping
                     │
               FHIR / API Mapping
                     │
              Data Quality Control
                     │
              Longitudinal Timeline
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 PERSONAL HEALTH JOURNEY   POPULATION INTELLIGENCE
          │                     │
          ▼                     ▼
 NutriMed MyLab        Kemenkes / BPJS / Dinkes /
 Personal Intelligence Corporate / Providers
```

SATUSEHAT documentation states that SATUSEHAT uses HL7 FHIR for its data model and APIs. For Patient-related health-data exchange, SATUSEHAT uses the patient's IHS number obtained through the Master Patient Index. Production SI-HIS integration must follow the current SATUSEHAT authorization, profiles, terminology and validation rules.

## 6. Longitudinal Personal Health Journey

```
PREVENTION
   ↓
SCREENING
   ↓
SYMPTOM / SELF-OBSERVATION
   ↓
FKTP
   ↓
LAB / DIAGNOSTIC
   ↓
REFERRAL / FKRTL
   ↓
TREATMENT
   ↓
PHARMACY / MEDICATION
   ↓
NUTRITION / LIFESTYLE
   ↓
FOLLOW-UP
   ↓
OUTCOME
   ↓
PREVENTION AGAIN
   ↺
```

NutriMed MyLab is the user-facing layer that makes this journey visible, understandable and actionable. SI-HIS provides the intelligence and orchestration behind it.

## 7. Health Intelligence Hierarchy

```
KEMENKES
   ↓
DINKES PROVINSI
   ↓
DINKES KAB/KOTA
   ├── PUSKESMAS 01
   ├── PUSKESMAS 02
   └── PUSKESMAS N
```

The District/City Health Office has a parent-child intelligence relationship with its Puskesmas network. Upward information should normally be aggregated and role-controlled. Downward information should be limited to the receiving Puskesmas's lawful operational scope.

## 8. Client Information

- **Kemenkes:** national burden, surveillance, trends, forecasts, spatial signals, vulnerable groups and uncertainty.
- **BPJS:** JKN utilization, referral, treatment, medication, outcome, cost and efficiency.
- **Dinkes Provinsi:** regional epidemiology, district/city profiles and coordination intelligence.
- **Dinkes Kabupaten/Kota:** Puskesmas network, EWS, spatial/risk signals and verification workflow.
- **Puskesmas:** local population, village signals, surveillance, prevention and follow-up.
- **Corporate:** aggregate workforce health, risk segmentation, prevention and outcomes.
- **Hospitals/Clinics/Laboratories/Pharmacies:** clinical-operational, diagnostic, medication, capacity and outcome intelligence.

## 9. Disease Intelligence Routing

### PTM / NCD
PERSON, TIME, LAB, BEHAVIOR, RISK FACTOR, OUTCOME.

### Communicable disease
TIME, PERSON, PLACE, transmission/contact and outcome; epidemic curve, EWS, Rt where assumptions permit, growth, spatial analysis and forecasting.

### Vector-borne
Add VECTOR and ENVIRONMENT.

### Zoonotic
Add ANIMAL and ENVIRONMENT for One Health intelligence.

### Unknown event
May activate cautious acute-event/poisoning screening. Known diseases should not automatically trigger toxicology.

## 10. Statistics

- Descriptive: n, mean, median, SD, IQR, proportions, prevalence, CFR and mortality rate where the denominator is appropriate.
- Bivariate: Chi-square/Fisher, t-test/Mann-Whitney, ANOVA/Kruskal-Wallis and correlation.
- Multivariate: logistic, ordinal/multinomial, count, survival and mixed-effects models where appropriate.
- Epidemiological measures: attack rate, incidence, prevalence, CFR, growth, Rt where valid, epidemic curve, change points and spatial clustering.
- Uncertainty, missingness, denominator and assumptions must be reported.

**Association is not causation.**

## 11. Machine Learning

Current architecture includes:
- case-severity modeling;
- infectious KLB/outbreak prediction routing;
- infectious spatial modeling;
- vulnerable-population analysis;
- robust forecasting;
- temporal anomaly detection;
- change-point detection;
- growth risk;
- spatial-neighbor intelligence;
- vulnerability clustering;
- spatiotemporal risk;
- time-person-place risk;
- disease-specific growth;
- continuous signal prioritization.

Case-severity evaluation includes chronological holdout, class balancing, permutation importance, ROC-AUC, PR-AUC, recall, specificity, F1, Brier and probability calibration diagnostics.

Calibration diagnostics include ECE, log-loss and calibration bins.

DBSCAN is a density-clustering method for spatial interpretation. A cluster is not automatically a source of transmission.

## 12. ML Development Roadmap

1. Walk-forward temporal validation
2. Probability calibration
3. Baseline comparison
4. Reporting-delay analysis and nowcasting
5. Calibrated 50/80/95% forecast intervals
6. Coverage and Weighted Interval Score
7. Data/model/concept drift monitoring
8. Epidemiological signal fusion
9. Disease-specific ML
10. Explainable AI
11. Continuous learning with validation and rollback

## 13. Epidemiological Signal Fusion

```
TIME + PERSON + PLACE + OUTCOME + ML
          + ENVIRONMENT / ONE HEALTH
                     ↓
              EVIDENCE FUSION
                     ↓
         CONVERGENT EPIDEMIOLOGICAL SIGNAL
                     ↓
              HUMAN VERIFICATION
                     ↓
                   ACTION
                     ↓
                  OUTCOME
                     ↓
                  NEW DATA
```

Convergence strengthens a signal but does not eliminate the need for validation.

## 14. FHIR / SATUSEHAT

Canonical resources include:
- Patient
- Organization
- Location
- Encounter
- Observation
- Condition
- Procedure
- DiagnosticReport
- Specimen
- Medication
- MedicationRequest
- MedicationDispense
- ServiceRequest
- CarePlan
- RiskAssessment
- DocumentReference

Production implementation must follow the current SATUSEHAT profile, terminology, validation, authorization and use-case requirements.

## 15. Future Holographic Personal Health Assistant

Concept:

**NutriMed MyLab Mobile App → Multimodal AI Companion → Holographic Personal Health Assistant**

The envisioned assistant could:
- maintain contextual dialogue;
- understand the user's longitudinal health journey;
- explain authorized laboratory and health information;
- ask structured follow-up questions;
- remind and coach preventive activities;
- coordinate authorized services;
- receive voice/dialogue-based observations;
- interact with SI-HIS services.

This is a product vision and R&D direction. Any future deployment requires dedicated assessment of human factors, privacy, cybersecurity, clinical safety, accessibility and applicable regulation.

## 16. Dinkes Kabupaten/Kota ↔ Puskesmas

```
Puskesmas → Data → SI-HIS → Cross-Puskesmas Signal
                         ↓
                  Dinkes Kab/Kota
                         ↓
                    Verification
                         ↓
                     Puskesmas
                         ↓
                 Field / Clinical Action
                         ↓
                      Outcome
                         ↓
                    New Data
                         └──→ SI-HIS Re-analysis
```

## 17. Workforce Health Intelligence

Employees can become NutriMed MyLab users while the corporate dashboard receives aggregate workforce intelligence:
- workforce profile;
- risk segmentation;
- temporal signals;
- validated predictive signals;
- preventive programs;
- outcome monitoring.

Corporate health intelligence should apply data minimization, aggregation, cohort suppression, role-based access and audit controls.

## 18. Healthcare Provider Intelligence

Hospital:
capacity, clinical mix, waiting time, length of stay, referral, readmission, mortality/outcome, laboratory, pharmacy and cost.

Clinic:
outpatient demand, waiting, clinical/service mix, referral, follow-up, laboratory and pharmacy.

Laboratory:
test volume, TAT, specimen, pending result, rejection/repeat, referral and reagent/capacity demand.

Pharmacy:
prescription, dispensing, medication utilization, stock, stockout, expiry and refill.

## 19. Governance

AI supports detection, prediction, explanation and verification prioritization. It does not independently diagnose, declare legal KLB, order public-health intervention, deny healthcare or make employment decisions.

Required controls:
- purpose limitation;
- data minimization;
- role-based access;
- security and encryption;
- audit trail;
- retention;
- pseudonymization/de-identification where appropriate;
- consent/lawful basis as applicable;
- model auditability.

## 20. IT Responsibilities

Data Engineering: connectors, ETL/ELT, FHIR, terminology, identity and quality.

Data Science: statistics, ML, forecasting, calibration, validation and drift.

Epidemiology: case definitions, denominator, surveillance and verification.

Backend/API: authentication, authorization, tenant scope, APIs and audit.

Frontend: client dashboards, alerts, drill-down and workflow.

MLOps: registry, versioning, monitoring, rollback and retraining.

## 21. Repository Architecture

```
SI-HIS-Intelligence/
├── app.py
├── core/
│   ├── engine.py
│   ├── ml_engine.py
│   ├── disease_intelligence.py
│   ├── workforce_intelligence.py
│   ├── provider_intelligence.py
│   ├── client_intelligence.py
│   └── client_portal.py
├── pages/
│   ├── Kemenkes
│   ├── BPJS
│   ├── Dinkes Provinsi
│   ├── Dinkes Kab/Kota
│   ├── Puskesmas
│   ├── Workforce Health
│   └── Healthcare Provider
└── docs/
```

## 22. Roadmap 2026–2030

- **2026:** Foundation & Validation
- **2027:** Ecosystem Integration
- **2028:** SI-HIS Intelligence Expansion
- **2029:** Scale & B2B2C/B2G Growth
- **2030:** Sustainable Growth & National Impact

## 23. Final Principle

SI-HIS should answer:

1. **WHAT is happening?**
2. **WHERE and WHO are affected?**
3. **WHAT may happen next?**
4. **WHY is the signal appearing?**
5. **WHAT should humans verify or consider next?**

Final loop:

**DATA → INTELLIGENCE → DECISION SUPPORT → HUMAN VERIFICATION → ACTION → OUTCOME → NEW DATA → CONTINUOUS LEARNING**
