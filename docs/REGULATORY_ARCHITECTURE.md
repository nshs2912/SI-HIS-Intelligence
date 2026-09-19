# SI-HIS Regulatory Architecture

## 1. Two intelligence planes
**Population Health / Sistem Informasi Kesehatan** covers surveillance, epidemiology, temporal/spatial analysis, outbreak and poisoning hypothesis generation, forecasting, early warning and public-health decision support.

**Individual Health / Clinical Intelligence** covers patient-specific laboratory interpretation, pharmacy safety, nutrition decision support, longitudinal clinical interpretation and other clinical decision support.

When an individual-patient function interprets clinical data and produces or supports a recommendation about diagnosis, treatment, triage, prognosis or another clinical decision, SI-HIS routes it to the **clinical/SaMD candidate boundary**. This is an engineering/governance classification, not a legal determination; exact intended purpose, claims, risk, validation and applicable Indonesian rules require formal assessment.

## 2. Individual → population → individual

Individual clinical data → canonical data → clinical/SaMD candidate analytics and/or controlled aggregation → privacy/purpose/provenance gate → population epidemiology → surveillance/early warning → public-health action → feedback to the clinical journey.

The population layer must not silently become a patient-level clinical decision engine. Population signals may inform targeted investigation or contextual clinical review, but not automatic individual diagnosis or treatment.

## 3. Explicit analysis scope

SI-HIS should carry an explicit `analysis_scope`: `individual` or `population`. The same laboratory source can therefore have two distinct uses:
- population: aggregate abnormal-result patterns, surveillance indicators, geographic trends;
- individual: interpretation of a patient's results for diagnostic or treatment decision support, which belongs to the clinical/SaMD candidate boundary.

## 4. Toxicology / poisoning

A suspected foodborne, inhalational or chemical poisoning pattern across multiple people remains an SIK/surveillance function when the output is a hypothesis, differential, cluster signal or investigation priority. If the same engine is applied to one identified patient and produces a patient-specific diagnostic recommendation, it crosses into the clinical/SaMD candidate boundary.

## 5. Interoperability and safety

FHIR/SATUSEHAT is the integration layer, not the regulatory classification. Clinical/SaMD candidate functions require intended-purpose definition, risk management, clinical validation, evidence/provenance traceability, human oversight, change control and post-deployment monitoring. Do not claim legal SaMD compliance merely from this engineering boundary.