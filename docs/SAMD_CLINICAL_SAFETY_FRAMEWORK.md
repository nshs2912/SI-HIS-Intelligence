# Clinical Safety & Validation Framework

This framework applies **per module**, not to SI-HIS as one undifferentiated product.

## Stage 1 — Intended purpose
Define target population, user, clinical context, input, output, action and explicit exclusions.

## Stage 2 — Hazard analysis
Assess wrong patient, wrong data, missing data, delayed data, terminology mismatch, false positive, false negative, automation bias, alert fatigue, model drift, cybersecurity compromise and interoperability failure.

## Stage 3 — Verification
Unit tests, data-quality tests, terminology validation, FHIR schema/profile validation, boundary tests and reproducibility.

## Stage 4 — Clinical validation
Use an appropriate reference standard/adjudication set. Predefine metrics, confidence intervals and clinically meaningful acceptance thresholds.

## Stage 5 — Human factors
Evaluate workflow, understandability, alert burden, override behavior and safe escalation.

## Stage 6 — Silent/prospective evaluation
Run without changing care first; compare AI signals with professional adjudication and real workflow outcomes.

## Stage 7 — Release/change control
Version model, rules, terminology, data schema and FHIR profiles. Revalidate material changes.

## Stage 8 — Post-market/production monitoring
Monitor safety incidents, false negatives, false positives, drift, subgroup performance, data quality, latency and user overrides.

## Module evidence package
Each module should have:
- Intended Purpose
- User & use environment
- Risk analysis
- FMEA/hazard log
- Clinical evaluation plan/report
- Verification/validation protocol
- Dataset/data provenance
- Model/rule version
- Terminology version
- FHIR profile/version
- Human factors evidence
- Cybersecurity/privacy assessment
- Change-control record
- Incident/post-deployment monitoring plan

## Regulatory note
A repository label such as “SaMD candidate” is not a legal classification. Formal classification must be performed against the current Indonesian medical-device regulatory framework and the exact intended purpose and claims.
