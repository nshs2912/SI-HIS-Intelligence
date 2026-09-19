# SI-HIS Disease-Aware Machine Learning

## Purpose

SI-HIS ML is disease-aware. It does not apply one identical analytical interpretation to all diseases.

### Disease taxonomy

**Non-communicable diseases (PTM)**
- Metabolic: diabetes, obesity, metabolic disorders.
- Cardiovascular/cardiometabolic: hypertension, heart disease, stroke.
- Degenerative/chronic: chronic kidney disease, neurodegenerative and other chronic conditions.
- Neoplasm/cancer.

**Communicable diseases**
1. Direct transmission: respiratory/contact diseases such as TB, measles and influenza-like illness.
2. Indirect/environmental/fecal-oral transmission: selected diarrheal and environmental infections.
3. Vector-borne: dengue/DBD, malaria, chikungunya and related diseases.
4. Zoonotic: leptospirosis, rabies, avian influenza and other One Health conditions.

Unknown diseases remain UNKNOWN until a validated taxonomy profile is added.

## ML strategy

PTM focuses on person-level risk, longitudinal progression, laboratory and clinical outcomes, complications, hospitalization, mortality, behavior and chronic risk factors.

Communicable disease intelligence focuses on TIME + PERSON + PLACE, epidemic curve, anomaly/change-point detection, growth and outbreak signals, spatial/spatio-temporal patterns, forecasting, transmission context, vector/environmental context when available, and One Health context for zoonoses.

The existing five primary ML engines and nine supporting intelligence modules remain intact. Disease classification adds context and routing; it does not replace or silently alter their algorithms.

## Onset-aware epidemic intelligence

For communicable diseases, when Tanggal Onset is available, the epidemiological clock is based on onset. The original service/examination date is preserved as Tanggal Pemeriksaan Asli.

The system additionally calculates onset completeness, onset-to-examination delay, onset-to-reporting delay when available, negative/invalid temporal intervals, suspicious concentration of identical onset dates, epidemic-curve features, and candidate earliest-onset cases.

### Index case guardrail

The earliest recorded onset is NOT automatically declared the index case.

Output wording: Candidate case based on the earliest recorded onset in the available dataset.

Field epidemiological investigation, case definition, exposure/contact investigation, laboratory evidence and data-quality review remain necessary.

## Temporal bias guardrails

If examination date is used where onset should be used, the resulting curve may reflect reporting/service behavior rather than disease occurrence. SI-HIS therefore distinguishes disease time (onset), clinical/service time (examination/encounter), laboratory time (specimen/result), and surveillance time (reporting).

Examples of quality signals: examination before onset, reporting before onset, high missingness of onset, unusually large onset-date batch concentration, and long reporting delay.

These are data-quality signals, not automatic evidence of transmission.

## Continuous intelligence

DATA -> DISEASE CLASSIFICATION -> EPIDEMIOLOGY -> FEATURE ENGINEERING -> ML -> EWS -> DSS -> INTERVENTION -> OUTCOME -> NEW DATA

All predictive outputs remain decision-support. Feature importance is not causality; forecast is not certainty; spatial concentration is not proof of transmission; an internal outbreak score is not a legal KLB declaration.
