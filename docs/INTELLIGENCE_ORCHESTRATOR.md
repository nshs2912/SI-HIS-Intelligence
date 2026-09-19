# SI-HIS Intelligence Orchestrator

## Purpose
The Intelligence Orchestrator is a cognitive layer above the existing epidemiology and ML modules. It combines temporal, spatial, disease-aware, acute-event, and model signals into an evidence-aware operational triage.
It does not replace clinical judgment, epidemiological investigation, laboratory confirmation, legal KLB determination, or disaster authority.

## Intelligence layers
1. Multi-window temporal scan: 15m, 30m, 1h, 3h, 6h, 12h, 24h.
2. Finest-unit intelligence: evaluates the smallest available operational geography, prioritizing Puskesmas when present, followed by Desa/Kelurahan, Kecamatan, Kabupaten, and Provinsi.
3. Unknown-event detection: detects dense temporal clusters with heterogeneous diagnosis labels.
4. Evidence fusion: combines acute-event, infectious context, temporal density, finest-unit concentration, and unknown-event signals.
5. Uncertainty intelligence: separates evidence confidence from event/etiology probability.
6. Next-action guidance and continuous-learning feedback requirements.

## Operational interpretation
The system may produce:
- NO_STRONG_SIGNAL
- SIGNAL
- TARGETED_INVESTIGATION
- URGENT_INVESTIGATION

These are internal intelligence states. They are not legal classifications.

## Example reasoning
If many cases arrive within a short period in one Puskesmas and the onset times are tightly concentrated, SI-HIS can surface:
- temporal density,
- finest-unit concentration,
- acute-event/common-exposure signals,
- unknown-event pattern when diagnosis labels are heterogeneous,
- uncertainty/confidence,
- recommended verification actions.

The system then asks for corroborating evidence such as line list, exposure timing/location, symptoms, exposed vs non-exposed population, clinical findings, laboratory/toxicology results, and field investigation.

## Performance
The multi-window scan uses a sorted timestamp/search strategy rather than comparing every timestamp with every other timestamp. This keeps the scan substantially more scalable as the dataset grows.

## Continuous learning
The orchestrator records the need for outcome feedback, including:
- confirmed event type,
- laboratory result,
- intervention,
- outcome.

These outcomes can later become labeled events for model evaluation, recalibration, and retraining.