# Incident & Outbreak Reasoning Engine v2

SI-HIS v2 adds evidence fusion for acute clusters and outbreak investigation.

## Inputs
- onset/examination/reporting timestamps
- finest available geography
- symptoms/syndromes
- exposure status
- facility
- severity and mortality
- environmental or incident context
- existing acute-event and infectious intelligence

## Reasoning
Temporal windows: 15 minutes, 30 minutes, 1 hour, 3 hours, 6 hours, 12 hours, 24 hours.
Spatial reasoning: evaluates the smallest available operational unit and its temporal concentration.
Exposure reasoning: calculates attack rates and risk ratio when valid exposed/non-exposed denominators exist.
Facility surge: compares the latest facility burden with its historical daily baseline.
Severity/mortality burst: surfaces simultaneous increases in severe disease or deaths.
Symptom clustering: summarizes dominant syndrome labels when available.
Environmental corroboration: surfaces available fields such as flooding, earthquake, fire, chemical exposure, contaminated water/food, weather and mass gathering.

## Output
The engine returns signals, an internal evidence score, investigation priority, differential hypotheses and a guardrail.

Possible priorities:
- ROUTINE_SURVEILLANCE
- TARGETED_INVESTIGATION
- URGENT_INVESTIGATION

The evidence score is an operational triage score, not a probability of diagnosis.

## Safety and governance
The engine does not automatically declare poisoning, outbreak, KLB, disaster, source of exposure, or etiology. Those require case definition, field investigation and appropriate clinical/laboratory/authority confirmation.

## Continuous learning
Confirmed event type, laboratory findings, intervention and outcomes should be persisted as feedback labels for subsequent evaluation and model recalibration.