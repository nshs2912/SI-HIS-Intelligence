# SI-HIS Regulatory Architecture

## A. Sistem Informasi Kesehatan
Population surveillance, epidemiology, forecasting, spatial intelligence and public-health decision support are governed separately from clinical/SaMD modules.

## B. Clinical / SaMD candidate
Laboratory, Pharmacy, Dietitian, Clinical Journey and Patient Clinical Summary each have their own intended purpose, risk assessment, clinical validation, safety controls, human factors, change control and regulatory assessment.

## C. No automatic legal classification
“SaMD candidate” is an engineering/governance designation only. Whether software meets the legal definition/classification of a medical device must be assessed from the exact intended purpose, claims, clinical role, risk and current Indonesian rules.

## D. Interoperability
FHIR/SATUSEHAT mapping is a separate integration layer. Clinical analytics consume canonical data and should retain source identifiers/provenance.

## E. Privacy/security
PDP, health-sector confidentiality, access control, auditability, data minimization and security controls are cross-cutting but assessed according to the actual deployment.
