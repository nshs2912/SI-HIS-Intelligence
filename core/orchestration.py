"""SI-HIS Continuous Intelligence Orchestrator.

Architecture principle:
    Observe -> Integrate -> Analyze -> Predict -> Decide -> Act -> Measure -> Learn -> Observe

This module provides a lightweight orchestration contract for the prototype.
It does not replace clinical governance, epidemiological definitions, or validated
production ML pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


CYCLE_STAGES = [
    "OBSERVE",
    "INTEGRATE",
    "ANALYZE",
    "PREDICT",
    "DECIDE",
    "ACT",
    "MEASURE",
    "LEARN",
]

INTELLIGENCE_DOMAINS = {
    "personal": "Personal Health Intelligence",
    "clinical": "Clinical Intelligence",
    "population": "Population Health Intelligence",
}

DATA_DOMAINS = {
    "individual_health": "Individual Health Data",
    "healthcare_services": "Healthcare & Medical Support Data",
    "cross_sector": "Cross-Sector Epidemiological Data",
}


@dataclass
class IntelligenceEvent:
    """Traceable unit of SI-HIS intelligence processing."""

    stage: str
    domain: str
    event_type: str
    source: Optional[str] = None
    model_version: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinuousIntelligenceCycle:
    """Minimal state container for one continuous SI-HIS cycle."""

    cycle_id: str
    events: List[IntelligenceEvent] = field(default_factory=list)

    def record(
        self,
        stage: str,
        domain: str,
        event_type: str,
        source: Optional[str] = None,
        model_version: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> IntelligenceEvent:
        stage = stage.upper()
        if stage not in CYCLE_STAGES:
            raise ValueError(f"Unknown SI-HIS cycle stage: {stage}")
        if domain not in INTELLIGENCE_DOMAINS:
            raise ValueError(f"Unknown intelligence domain: {domain}")
        event = IntelligenceEvent(
            stage=stage,
            domain=domain,
            event_type=event_type,
            source=source,
            model_version=model_version,
            payload=payload or {},
        )
        self.events.append(event)
        return event

    def summary(self) -> Dict[str, Any]:
        completed = {event.stage for event in self.events}
        return {
            "cycle_id": self.cycle_id,
            "completed_stages": [s for s in CYCLE_STAGES if s in completed],
            "next_stage": next((s for s in CYCLE_STAGES if s not in completed), None),
            "event_count": len(self.events),
            "continuous": True,
        }


def build_intelligence_routing() -> Dict[str, List[str]]:
    """Describe where SI-HIS intelligence is routed after analysis."""
    return {
        "personal": ["NutriMed MyLab", "Partner Mobile Apps"],
        "clinical": [
            "Doctor",
            "Hospital",
            "Puskesmas",
            "Laboratory",
            "Pharmacy/Apotek",
            "Dietitian",
            "Physiotherapy",
            "Rehabilitation",
            "Other Medical Support",
        ],
        "population": [
            "Kemenkes",
            "Dinkes",
            "BPJS",
            "Corporate Health",
            "Public Health Programs",
        ],
    }


def build_data_fusion_contract() -> Dict[str, List[str]]:
    """Canonical high-level input contract for the intelligence layer."""
    return {
        "individual_health": [
            "patient demographics",
            "clinical records",
            "laboratory results",
            "medications",
            "nutrition and lifestyle",
            "vital signs",
            "medical imaging metadata/findings",
            "patient-generated health data",
        ],
        "healthcare_services": [
            "encounters",
            "diagnosis and treatment",
            "laboratory services",
            "pharmacy services",
            "dietitian services",
            "physiotherapy and rehabilitation",
            "radiology and pathology",
            "other supporting medical services",
        ],
        "cross_sector": [
            "population and demography",
            "environment and climate",
            "water and sanitation",
            "food systems",
            "mobility",
            "education",
            "occupation and workforce",
            "agriculture and livestock",
            "socioeconomic indicators",
            "other epidemiologically meaningful signals",
        ],
    }
