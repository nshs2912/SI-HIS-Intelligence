"""Privacy/security-by-design primitives for SI-HIS.

These are engineering controls, not a claim of legal compliance. Legal basis,
controller/processor roles, retention and consent/other lawful grounds must be
configured for the actual deployment.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Iterable, Optional
import pandas as pd

@dataclass(frozen=True)
class AccessContext:
    user_id: str
    role: str
    purpose: str
    patient_id: Optional[str] = None
    organization_id: Optional[str] = None

@dataclass(frozen=True)
class PrivacyDecision:
    allowed: bool
    reason: str
    fields: tuple[str, ...] = ()

    def to_dict(self): return asdict(self)

ROLE_FIELDS = {
    "doctor": {"patient_id","date","diagnosis","condition","allergy","lab","medication","procedure","encounter","clinical_note","outcome"},
    "dietitian": {"patient_id","date","diagnosis","condition","lab","medication","weight","height","bmi","nutrition","diet","outcome"},
    "physiotherapist": {"patient_id","date","diagnosis","condition","pain","mobility","rom","strength","function","procedure","rehabilitation","outcome"},
    "pharmacist": {"patient_id","date","diagnosis","condition","allergy","lab","medication","dose","route","frequency","dispense","adverse_event","outcome"},
    "laboratory": {"patient_id","date","diagnosis","condition","lab","specimen","result","unit","reference_range","critical_flag"},
    "admin": set(),
}

def authorize(context: AccessContext, available_fields: Iterable[str]) -> PrivacyDecision:
    role=context.role.lower().strip()
    if role not in ROLE_FIELDS:
        return PrivacyDecision(False, "Unknown role; access denied.")
    if not context.user_id or not context.purpose:
        return PrivacyDecision(False, "User identity and purpose are required.")
    allowed=tuple(sorted(set(available_fields) & ROLE_FIELDS[role]))
    return PrivacyDecision(True, "Access allowed subject to deployment policy and lawful processing basis.", allowed)

def pseudonymize(identifier: object, salt: str) -> str:
    raw=f"{salt}:{identifier}".encode("utf-8")
    return sha256(raw).hexdigest()

def minimize_columns(df: pd.DataFrame, allowed_fields: Iterable[str]) -> pd.DataFrame:
    allowed=set(allowed_fields)
    keep=[c for c in df.columns if c in allowed]
    return df.loc[:, keep].copy(deep=True)

def deidentify_dataframe(df: pd.DataFrame, identifiers=("Nama","Name","NIK","No. KTP","No KTP","Email","No. HP","Phone")) -> pd.DataFrame:
    out=df.copy(deep=True)
    for col in identifiers:
        if col in out.columns:
            out[col]="[REDACTED]"
    return out

def privacy_controls():
    return {
        "principles": ["purpose_limitation","data_minimization","role_based_access","pseudonymization","auditability","provenance","retention_control","secure_transfer"],
        "note": "Implement against the applicable PDP, health-sector and organizational requirements before production.",
    }
