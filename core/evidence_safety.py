"""Evidence, safety and provenance layer for clinical AI outputs."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Iterable
from datetime import datetime, timezone

@dataclass(frozen=True)
class EvidenceItem:
    source_type: str
    source_id: str
    field: str
    value: Any
    timestamp: str | None = None

@dataclass(frozen=True)
class ClinicalSignal:
    module: str
    signal: str
    priority: str
    confidence: float | None
    evidence: tuple[EvidenceItem, ...]
    human_review_required: bool = True

    def to_dict(self):
        d=asdict(self)
        d["evidence"]=[asdict(x) for x in self.evidence]
        return d

def evidence_item(source_type, source_id, field, value, timestamp=None):
    return EvidenceItem(source_type, str(source_id), field, value, timestamp)

def build_signal(module, signal, priority="REVIEW", confidence=None, evidence: Iterable[EvidenceItem]=()):
    if confidence is not None:
        confidence=max(0.0,min(1.0,float(confidence)))
    return ClinicalSignal(module, signal, priority, confidence, tuple(evidence), True)

def provenance_record(module, model_version="rule-based/unspecified", input_ids=None, purpose="clinical decision support"):
    return {
        "module": module,
        "model_version": model_version,
        "input_ids": list(input_ids or []),
        "purpose": purpose,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "human_review_required": True,
        "source_traceability": True,
    }

def safety_gate(result: dict) -> dict:
    out=dict(result)
    out["safety"]={
        "autonomous_diagnosis": False,
        "autonomous_treatment_order": False,
        "human_review_required": True,
        "traceable_evidence_required": True,
    }
    return out
