# Clinical Intelligence Modules

Implemented as a separate clinical boundary from epidemiological intelligence.

## Modules
1. Laboratory Intelligence — reference range, abnormality, delta, critical-value candidate, trajectory and data quality.
2. Pharmacy Intelligence — medication timeline, duplicate signal, refill/adherence proxy, interaction signal, adverse-event signal and demand.
3. Dietitian Intelligence — anthropometric trajectory, nutrition adequacy, disease-specific pattern and adherence.
4. Clinical Journey Intelligence — longitudinal timeline, follow-up gaps, care gaps and treatment-response loop contract.
5. Patient Clinical Summary — role-based summary for doctor, dietitian, physiotherapist, pharmacist and laboratory professional.

## Safety rule

The modules produce signals, summaries or decision-support recommendations. They do not autonomously diagnose, prescribe, change medication or replace the source clinical record.

## Role-based summary

The same patient can have different minimum-necessary summaries:
- doctor: clinical problem/lab/medication/procedure context
- dietitian: nutrition, anthropometry, relevant labs and medication context
- physiotherapist: function, mobility, pain and rehabilitation context
- pharmacist: medication, allergy, lab and adverse-event context
- laboratory professional: specimen/result/reference/critical-value context

Every production deployment must implement identity, role, purpose, authorization, audit, retention and lawful-processing controls outside this prototype.
