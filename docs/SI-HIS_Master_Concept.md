# SI-HIS — Smart Integrated Health Intelligence System
## Master Concept, Architecture, Algorithms, ML, Statistics, Information Delivery and Operational Flow

**Repository:** `nshs2912/SI-HIS-Intelligence`  
**Product ecosystem:** SI-HIS + NutriMed MyLab  
**Document purpose:** technical reference for IT implementation, product governance, client explanation, and investor/client presentations.  
**Version:** 1.0 — September 2026

---

## 1. Executive Concept

SI-HIS (Smart Integrated Health Intelligence System) is an AI/ML-enabled health intelligence architecture that transforms longitudinal health-service data into decision-support information at the correct organizational scale.

The central principle is:

**DATA → ANALYSIS → DETECTION → PREDICTION → RECOMMENDATION → INTERVENTION → OUTCOME → NEW DATA → CONTINUOUS LEARNING**

SI-HIS is not intended to replace clinicians, epidemiologists, public-health authorities, payers, or corporate health governance. AI outputs are decision support and must be interpreted with data quality, denominator, uncertainty, model validation, legal authority, and field verification.

WHO defines public-health surveillance as continuous, systematic collection, analysis and interpretation of health-related data. Surveillance supports early warning, intervention monitoring, epidemiological understanding, planning, and policy. WHO also describes public-health intelligence as integrating multiple sources to generate actionable insights. This concept is the conceptual foundation of SI-HIS.

---

## 2. Product and Brand Architecture

### 2.1 SI-HIS
**Smart Integrated Health Intelligence System** is the technology and intelligence layer.

Responsibilities:
- data integration
- epidemiological analytics
- statistics
- machine learning
- forecasting
- spatial intelligence
- signal fusion
- early warning
- decision-support orchestration
- client-specific information delivery
- governance and audit

### 2.2 NutriMed MyLab
NutriMed MyLab is the user-facing Personal Health Companion and longitudinal health journey layer.

Typical flow:

**Individual → NutriMed MyLab → SI-HIS → Intelligence → Intervention → Outcome → NutriMed MyLab**

For corporate use, employees can become NutriMed MyLab users while the company receives appropriately aggregated Workforce Health Intelligence.

---

## 3. Core Architecture

```
                    SI-HIS
     SMART INTEGRATED HEALTH INTELLIGENCE
                         |
                 DATA INTEGRATION
                         |
          HL7 FHIR / SATUSEHAT / API / ETL
                         |
                 DATA & KNOWLEDGE
                         |
              ANALYTICS + STATISTICS
                         |
                    AI / ML
                         |
             SIGNAL FUSION / ORCHESTRATOR
                         |
          INFORMATION DELIVERY LAYER
                         |
    +--------+------+--------+--------+---------+
    |        |      |        |        |         |
 KEMENKES  BPJS  DINKES   CORP    PROVIDER   PUSKESMAS
              |      |
              |   PROV/KAB-KOTA
              |      |
              +--- PUSKESMAS NETWORK
```

### 3.1 Information Delivery Layer

The same intelligence engine does not need a separate algorithm for every client. Instead, SI-HIS determines:
- who is requesting information;
- geographic/organizational scope;
- permitted data granularity;
- decision questions;
- relevant indicators;
- aggregation/suppression rules;
- narrative and visualization;
- actions available to that role.

**One Intelligence Engine → Multiple Decision Views.**

---

## 4. Hierarchical Health Intelligence Network

### 4.1 National

**Kemenkes**
→ Indonesia → Province → District/City → aggregate facility signals.

**BPJS**
→ JKN population → utilization → referral → treatment → claim/outcome/cost intelligence.

### 4.2 Provincial

**Dinkes Provinsi**
→ Province → District/City → aggregate Puskesmas intelligence.

### 4.3 District/City

**Dinkes Kabupaten/Kota**
→ District/City → Kecamatan → Desa/Kelurahan → Puskesmas.

Every District/City Health Office has a parent-child intelligence flow to the Puskesmas in its administrative/service area.

### 4.4 Frontline

**Puskesmas**
→ Puskesmas → service area → village → population.

The Puskesmas receives local operational intelligence while the District/City Health Office receives aggregated intelligence across its Puskesmas network.

---

## 5. Dinkes Kabupaten/Kota ↔ Puskesmas Flow

```
Puskesmas 01 ─┐
Puskesmas 02 ─┤
Puskesmas 03 ─┤
Puskesmas 04 ─┼──> DINKES KAB/KOTA
Puskesmas 05 ─┤
Puskesmas N  ─┘
                    |
                    v
          Cross-Puskesmas Intelligence
                    |
        +-----------+-----------+
        |                       |
        v                       v
  Regional Signal        Puskesmas-specific
                              Signal
        |                       |
        +-----------+-----------+
                    |
                    v
            Verification / SOP
                    |
                    v
             Field / Clinical
               Investigation
                    |
                    v
                 Outcome
                    |
                    v
              New Observation
                    |
                    +----> SI-HIS re-analysis
```

The upward flow should normally prioritize aggregated intelligence rather than unrestricted individual clinical data. Downward flow should send only information relevant to the receiving Puskesmas and its lawful operational authority.

---

## 6. Client Information Profiles

### 6.1 Kemenkes — National Health Intelligence

**Scope:** Indonesia.

Key questions:
1. What is happening nationally?
2. Where are important changes occurring?
3. What are the trends, burden, outcomes, vulnerable groups, and forecasts?
4. Which signals require verification or national coordination?

Information:
- national disease burden
- communicable-disease surveillance
- PTM intelligence
- national trend and forecast
- spatial/spatiotemporal signal
- vulnerable populations
- mortality/outcome
- cross-region comparison
- national policy intelligence
- model uncertainty and data-quality indicators

### 6.2 BPJS — JKN Health & Utilization Intelligence

**Scope:** payer and insured-population perspective.

Flow:

**Participant → Service → Referral → Treatment → Medication → Claim → Outcome → Cost**

Information:
- utilization trend
- disease burden in JKN population
- referral pattern
- admission/readmission signal
- chronic disease progression
- medication utilization
- high-utilization/high-cost signals
- service-demand forecast
- outcome and efficiency indicators

AI must not automatically reject claims or make clinical eligibility decisions. Outputs are decision support.

### 6.3 Dinkes Provinsi — Provincial Intelligence

**Scope:** Province → District/City → Puskesmas aggregates.

Information:
- provincial epidemiological profile
- district/city comparative profiles
- regional trends
- early-warning signals
- spatial clusters
- vulnerable populations
- forecasts
- coordination intelligence

Comparison should be descriptive and contextual, not a simplistic ranking of local governments.

### 6.4 Dinkes Kabupaten/Kota — Local Health Intelligence

**Scope:** District/City → Kecamatan → Desa → Puskesmas.

Information:
- local surveillance
- Puskesmas-by-Puskesmas aggregate signals
- TIME/PERSON/PLACE
- spatial cluster
- anomaly/change point/growth signal
- vulnerable population
- forecast
- ML signals
- verification/follow-up workflow

### 6.5 Puskesmas — Frontline Intelligence

**Scope:** Puskesmas and its service population.

Information:
- local population profile
- disease trends
- TIME/PERSON/PLACE
- village-level signals
- anomaly and early warning
- risk/vulnerability
- preventive follow-up
- referral/continuity signal
- outcome feedback

### 6.6 Workforce Health Intelligence

**Scope:** corporate workforce, department/site/aggregate.

Information:
- aggregate workforce health profile
- risk segmentation
- temporal signals
- validated predictive risk signals
- preventive program intelligence
- outcome monitoring

Governance:
- health data minimization
- aggregation
- cohort suppression
- role-based access
- audit trail
- no employment discrimination or individual clinical decision based solely on AI output

### 6.7 Healthcare Provider Intelligence

Provider types:
- Rumah Sakit
- Klinik
- Laboratorium
- Apotek/Farmasi

Hospital:
- capacity
- clinical mix
- waiting time
- length of stay
- referral
- readmission
- mortality/outcome
- lab
- pharmacy
- cost

Clinic:
- outpatient demand
- waiting
- diagnosis/service mix
- referral
- follow-up
- lab/pharmacy

Laboratory:
- test volume
- turnaround time
- specimen
- pending result
- rejection/repeat signal
- referral
- reagent/capacity demand

Pharmacy:
- prescription
- dispensing
- medication utilization
- stock
- stockout
- expiry
- refill

---

## 7. Disease Intelligence Routing

SI-HIS routes algorithms according to disease family.

### 7.1 PTM / NCD

Dimensions:
- PERSON
- TIME
- LAB
- BEHAVIOR
- RISK FACTOR
- OUTCOME

Methods:
- descriptive statistics
- bivariate association
- multivariate modeling
- severity
- mortality
- progression
- vulnerable population
- trend/forecast
- population spatial clustering where relevant

Do not automatically apply:
- KLB
- Rt
- infectious epidemic wave logic
- outbreak logic

### 7.2 Communicable Disease

Dimensions:
- TIME
- PERSON
- PLACE
- transmission/contact
- outcome
- vulnerability

Methods:
- epidemic curve
- EWS
- Rt where assumptions permit
- change points
- growth
- spatial clustering
- spatiotemporal signal
- outbreak/KLB decision-support signal
- forecast
- severity

### 7.3 Vector-borne

Adds:
- VECTOR
- ENVIRONMENT

Examples:
- dengue/DBD
- malaria
- chikungunya

Possible external features:
- rainfall
- temperature
- vector density
- seasonality
- environmental indicators

### 7.4 Zoonotic

Adds:
- ANIMAL
- ENVIRONMENT

Concept:

**Human + Animal + Environment → One Health Intelligence**

### 7.5 Unknown Event

Unknown disease/event classes may activate cautious:
- acute-event screening
- poisoning/toxicology differential
- incident reasoning

Known diseases should not automatically trigger toxicology analysis.

---

## 8. Data Pipeline

```
SOURCE SYSTEMS
  |
  +-- EMR/RME
  +-- Laboratory
  +-- Pharmacy
  +-- Claims
  +-- NutriMed MyLab
  +-- Corporate MCU/K3
  +-- Puskesmas
  +-- Hospital/Clinic
  +-- Environmental/One Health sources
  |
  v
INGESTION
  |
  v
STANDARDIZATION
  |
  +-- Patient identity
  +-- Organization
  +-- Location
  +-- Terminology
  +-- Dates/time
  +-- Units
  +-- Missingness
  |
  v
QUALITY CONTROL
  |
  +-- duplicate
  +-- missing
  +-- invalid range
  +-- temporal consistency
  +-- spatial consistency
  +-- denominator
  |
  v
ANALYTICS / STATISTICS
  |
  v
AI / ML
  |
  v
SIGNAL FUSION
  |
  v
CLIENT-SPECIFIC INFORMATION
```

---

## 9. FHIR / SATUSEHAT Interoperability

SATUSEHAT uses HL7 FHIR for its data model and APIs. Relevant resources include Patient, Organization, Location, Encounter, Observation, Condition, Procedure, DiagnosticReport, Specimen, Medication, MedicationRequest, MedicationDispense, ServiceRequest, CarePlan, RiskAssessment, DocumentReference and others.

Canonical SI-HIS mapping:

| Domain | FHIR resource |
|---|---|
| Patient | Patient |
| Organization | Organization |
| Facility/location | Location |
| Encounter | Encounter |
| Clinical observation | Observation |
| Diagnosis | Condition |
| Procedure | Procedure |
| Laboratory report | DiagnosticReport |
| Specimen | Specimen |
| Medication master | Medication |
| Prescription | MedicationRequest |
| Dispensing | MedicationDispense |
| Service order | ServiceRequest |
| Care pathway | CarePlan |
| Risk | RiskAssessment |
| Clinical document | DocumentReference |

The production implementation must follow the current SATUSEHAT profile, terminology, validation rules, and use case. Do not treat the canonical map as a substitute for official implementation guidance.

---

## 10. Statistical Layer

Statistics provide the evidence layer before and alongside ML.

### 10.1 Descriptive statistics
- n
- mean
- median
- SD
- IQR
- minimum/maximum
- proportions
- prevalence where an appropriate denominator exists
- CFR for case-based fatality
- mortality rate only when population denominator is available

### 10.2 Bivariate analysis

Depending on variable types:
- Chi-square/Fisher exact
- t-test/Mann-Whitney
- ANOVA/Kruskal-Wallis
- correlation

Important interpretation:

**Association ≠ causation.**

### 10.3 Multivariate analysis

Potential models:
- logistic regression
- multinomial/ordinal regression
- count models
- survival models
- mixed effects/hierarchical models where appropriate

Report:
- effect estimate
- confidence interval
- p-value
- sample size
- missingness
- model assumptions

### 10.4 Epidemiological measures

For infectious disease where valid:
- attack rate
- incidence
- prevalence
- CFR
- growth rate
- Rt
- epidemic curve
- temporal change point
- spatial clustering

### 10.5 Uncertainty

Every important prediction should eventually carry:
- point estimate
- prediction interval/confidence interval where appropriate
- model uncertainty
- data-quality warning
- missingness warning
- reporting-delay warning

---

## 11. Machine Learning Layer

Current repository architecture includes primary engines and supporting continuous-intelligence modules.

### 11.1 Case severity

Current approach:
- binary target normalization
- chronological holdout
- Logistic Regression / Random Forest
- class balancing
- permutation feature importance
- ROC-AUC
- PR-AUC
- recall
- specificity
- F1
- Brier score
- probability calibration diagnostics

Calibration diagnostics include:
- ECE
- log-loss
- calibration bins
- predicted vs observed rate

### 11.2 Vulnerable population

Uses combinations of demographic/context variables to identify strata with different observed outcomes.

Important:
- small strata can produce unstable CFR
- a high observed CFR is not automatically causal
- no automatic intervention ranking from raw CFR
- validate sample size and uncertainty

### 11.3 KLB/outbreak prediction

Enabled for communicable disease routing.

Possible features:
- recent cases
- growth
- lagged counts
- temporal anomaly
- spatial signal
- disease context

Output:
- predictive signal, not legal declaration of KLB.

### 11.4 Spatial outbreak model

For communicable diseases:
- coordinates
- density
- spatial clustering
- local concentration
- cluster signal

DBSCAN:
- identifies dense groups without requiring a fixed number of clusters
- labels sparse observations as noise
- supports PLACE interpretation

DBSCAN cluster ≠ transmission source.

### 11.5 Forecasting

Current robust forecast combines:
- ETS
- seasonal naive 7-day baseline
- naive baseline

Model weights are informed by temporal backtesting error.

Current intervals are residual-based and explicitly not calibrated epidemiological prediction intervals.

### 11.6 Continuous intelligence modules

Repository includes:
- temporal anomaly detection
- temporal change points
- growth risk
- spatial neighbor intelligence
- vulnerability clustering
- spatiotemporal risk
- time-person-place risk
- disease-specific growth
- continuous signal prioritization

---

## 12. Recommended ML Evolution Roadmap

### Phase 1 — Reliability
1. Walk-forward temporal validation
2. Probability calibration
3. Baseline comparison
4. Missingness analysis
5. Reporting-delay handling

### Phase 2 — Forecast uncertainty
1. 50/80/95% intervals
2. Coverage
3. Weighted Interval Score
4. Forecast calibration

### Phase 3 — Model monitoring
- data drift
- concept drift
- performance drift
- missingness drift
- geography drift
- outcome-rate drift

### Phase 4 — Nowcasting
Correct apparent declines caused by reporting delay.

### Phase 5 — Epidemiological Signal Fusion

Combine:
- TIME
- PERSON
- PLACE
- OUTCOME
- ML
- environmental/contextual data

Output should be:

**Convergent Epidemiological Signal**

rather than an automatic outbreak declaration.

### Phase 6 — Disease-specific ML

Diabetes:
- progression
- complications
- severity
- mortality
- risk factors

Dengue:
- temporal
- spatial
- vector
- environmental
- forecast

TB:
- contact
- treatment
- transmission
- cluster

Zoonosis:
- human + animal + environment

### Phase 7 — Explainable AI

Every important prediction should answer:

1. What did the model detect?
2. Which features contributed?
3. How strong is the evidence?
4. What uncertainty exists?
5. What should a human verify?

Feature importance is not causal inference.

### Phase 8 — Continuous Learning

```
NEW DATA
   ↓
QUALITY CHECK
   ↓
DRIFT CHECK
   ↓
MODEL PERFORMANCE
   ↓
RETRAIN / RE-CALIBRATE WHEN JUSTIFIED
   ↓
VALIDATION
   ↓
DEPLOY
   ↓
MONITOR
```

---

## 13. Epidemiological Signal Fusion

Recommended final orchestration:

```
                 TIME
                  |
PERSON -------- SIGNAL -------- PLACE
                  |
               OUTCOME
                  |
                  ML
                  |
             ENVIRONMENT
                  |
                  v
          EVIDENCE FUSION
                  |
        +---------+---------+
        |                   |
   CONVERGENT           INSUFFICIENT
     SIGNAL                SIGNAL
        |                   |
        v                   v
   VERIFICATION          MONITOR
        |
        v
  HUMAN DECISION
```

A signal should become stronger when independent dimensions converge, but convergence does not remove the need for validation.

---

## 14. Decision-Support Flow

### National

**Data → National analysis → signal → validation → policy intelligence**

### Provincial

**Provincial data → cross-district analysis → coordination signal → verification**

### District/City

**Puskesmas network → local analysis → signal → selected Puskesmas → field verification → outcome**

### Puskesmas

**Service data → local signal → verification → preventive/clinical action → outcome**

### Corporate

**Employee → NutriMed MyLab → aggregate intelligence → preventive program → outcome**

### Provider

**Patient/service data → operational/clinical intelligence → capacity/pathway action → outcome**

---

## 15. Information Granularity

| Client | Scale | Main intelligence |
|---|---|---|
| Kemenkes | National | Strategic population intelligence |
| BPJS | National/JKN | Payer/utilization intelligence |
| Dinkes Provinsi | Province | Regional coordination intelligence |
| Dinkes Kab/Kota | Local government | Operational surveillance intelligence |
| Puskesmas | Frontline | Local population intelligence |
| Corporate | Workforce | Workforce health intelligence |
| Rumah Sakit | Provider | Clinical-operational intelligence |
| Klinik | Provider | Patient-flow intelligence |
| Laboratorium | Diagnostic | Laboratory intelligence |
| Apotek/Farmasi | Medication | Pharmacy intelligence |

---

## 16. Sidebar / Client Portal Concept

The Streamlit application should expose client-oriented navigation:

- 🇮🇩 Kemenkes
- 🛡️ BPJS
- 🗺️ Dinkes Provinsi
- 🏙️ Dinkes Kabupaten/Kota
- 🏥 Puskesmas
- 🏢 Workforce Health Intelligence
- 🏥 Healthcare Provider Intelligence

AI/ML technical modules remain available as a secondary engineering layer:
- Epidemiology
- Forecast
- Spatial
- Risk
- Early Warning
- Signal Fusion
- Model Monitoring
- Governance

This separates **what the client needs to know** from **how the IT team computes it**.

---

## 17. Governance and Safety

### 17.1 Human-in-the-loop
AI supports:
- detection
- prediction
- explanation
- prioritization of verification

AI does not independently:
- diagnose
- declare legal KLB
- order public-health intervention
- deny healthcare
- make employment decisions

### 17.2 Data governance
Required controls:
- purpose limitation
- data minimization
- role-based access
- consent/lawful basis as applicable
- encryption
- audit trail
- retention policy
- pseudonymization/de-identification where appropriate
- cohort suppression
- model auditability

### 17.3 Corporate privacy
Corporate dashboards should normally expose aggregate workforce intelligence rather than unnecessary individual clinical data.

### 17.4 Epidemiological governance
An alert is a **signal requiring verification**, not automatically a confirmed outbreak.

---

## 18. IT Implementation Responsibilities

### Data Engineering
- connectors
- ETL/ELT
- FHIR mapping
- terminology mapping
- data quality
- identity and organization mapping

### Data Science
- statistical models
- ML
- forecasting
- calibration
- validation
- drift monitoring

### Epidemiology
- case definitions
- denominator
- surveillance rules
- outbreak interpretation
- verification protocols

### Backend/API
- authentication
- authorization
- tenant/client scope
- APIs
- audit

### Frontend
- client dashboard
- information granularity
- alert visualization
- drill-down
- workflow

### MLOps
- model registry
- model version
- metrics
- monitoring
- rollback
- retraining policy

---

## 19. Repository Architecture

Current and intended structure:

```
SI-HIS-Intelligence/
├── app.py
├── core/
│   ├── analytics.py
│   ├── data_provider.py
│   ├── disease_intelligence.py
│   ├── engine.py
│   ├── ml_engine.py
│   ├── narrative.py
│   ├── scope.py
│   ├── workforce_intelligence.py
│   ├── provider_intelligence.py
│   ├── client_intelligence.py
│   └── client_portal.py
├── pages/
│   ├── 01_Kemenkes.py
│   ├── 02_BPJS.py
│   ├── 03_Dinkes_Provinsi.py
│   ├── 04_Dinkes_Kab_Kota.py
│   ├── 05_Puskesmas.py
│   ├── 06_Workforce_Health.py
│   └── 07_Healthcare_Provider.py
├── docs/
│   └── SI-HIS_Master_Concept.md
└── requirements.txt
```

---

## 20. Investor / Client Value Proposition

SI-HIS creates value by separating the **intelligence engine** from the **decision context**.

For the same underlying health data:

- Kemenkes receives national strategic intelligence.
- BPJS receives JKN utilization and outcome intelligence.
- Provincial Health Offices receive regional coordination intelligence.
- District/City Health Offices receive operational surveillance intelligence.
- Puskesmas receive frontline population intelligence.
- Corporates receive workforce health intelligence.
- Providers receive clinical-operational intelligence.

This enables one scalable technology platform with multiple client-specific information products.

### Strategic proposition

**From data fragmentation to integrated health intelligence.**

**From retrospective reporting to continuous surveillance.**

**From descriptive dashboards to predictive decision support.**

**From individual health data to population intelligence — and back to the individual through NutriMed MyLab.**

---

## 21. Implementation Roadmap

### 2026 — Foundation & Validation
- core data model
- disease routing
- epidemiology
- initial ML
- FHIR/SATUSEHAT canonical layer
- client information architecture

### 2027 — Ecosystem Integration
- provider integration
- Dinkes/Puskesmas hierarchy
- BPJS integration pathway
- NutriMed MyLab integration
- role-based access

### 2028 — SI-HIS Intelligence Expansion
- signal fusion
- calibrated forecasting
- nowcasting
- drift monitoring
- disease-specific ML
- One Health

### 2029 — Scale
- B2B2C
- B2G
- national/regional deployments
- workforce intelligence
- provider intelligence

### 2030 — Sustainable Growth & National Impact
- national health intelligence network
- mature MLOps
- continuous learning
- outcome and economic impact measurement

---

## 22. Final Operating Principle

SI-HIS should always answer five questions:

**1. WHAT is happening?**  
Descriptive intelligence.

**2. WHERE and WHO are affected?**  
Epidemiological and population intelligence.

**3. WHAT may happen next?**  
Prediction and forecasting.

**4. WHY is the signal appearing?**  
Statistical/ML explanation and evidence.

**5. WHAT should humans verify or consider next?**  
Decision support.

The final loop is:

**DATA → INTELLIGENCE → DECISION SUPPORT → HUMAN VERIFICATION → ACTION → OUTCOME → NEW DATA → CONTINUOUS LEARNING**

This is the core design principle for the SI-HIS Intelligence repository and its client-facing ecosystem.
