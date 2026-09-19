# Acute Poisoning and Mass-Casualty Intelligence

SI-HIS now treats sudden clustered acute cases as a separate intelligence problem.

## Why a separate model

A classic infectious epidemic may develop over multiple generations, while a common-source event can produce many patients in a narrow time window. Chemical exposure can have onset within minutes to hours; foodborne/toxin-associated events may have onset over hours to days depending on the agent and exposure. The model therefore detects temporal concentration rather than assuming every burst is an infectious outbreak.

## Signal dimensions

TIME + PLACE + EXPOSURE + PERSON

The detector evaluates:
- number of cases in a short onset window;
- concentration of cases in the latest day versus historical baseline;
- exposure-to-onset interval when exposure time is available;
- geographic concentration;
- candidate common-source pattern.

## Differential hypotheses

A strong acute burst generates a neutral differential that can include:
1. poisoning/common exposure;
2. foodborne or toxin-associated event;
3. non-infectious chemical/environmental disaster;
4. mass gathering/common exposure;
5. acute infectious outbreak with closely spaced onset.

The system does NOT choose the final cause automatically.

## Operational interpretation

A strong signal should trigger rapid verification of:
- line list;
- exact onset time and place;
- common food/water/chemical exposure;
- attack rate among exposed and non-exposed groups;
- symptom pattern and severity;
- environmental/occupational context;
- laboratory and toxicology testing when indicated;
- disaster/incident information;
- ambulance/ED/Puskesmas/RS capacity and surge status.

## Guardrails

This is a triage and investigation-prioritization engine. It does not diagnose poisoning, identify a chemical/toxin, declare a KLB, or declare a disaster. Legal/public-health status remains under the relevant authority and verified evidence.
