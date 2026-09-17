"""Configuration helpers for SATUSEHAT integration.

Supports Streamlit secrets first, then environment variables. Secrets must
never be committed to GitHub.
"""
from __future__ import annotations

import os
from typing import Any


def _secret(secrets: Any, key: str, default: str = "") -> str:
    try:
        value = secrets.get(key, default)
    except Exception:
        value = default
    return str(value or default)


def load_satusehat_config(secrets: Any = None) -> dict:
    secrets = secrets if secrets is not None else {}
    return {
        "environment": _secret(secrets, "SATUSEHAT_ENVIRONMENT", os.getenv("SATUSEHAT_ENVIRONMENT", "sandbox")),
        "client_id": _secret(secrets, "SATUSEHAT_CLIENT_ID", os.getenv("SATUSEHAT_CLIENT_ID", "")),
        "client_secret": _secret(secrets, "SATUSEHAT_CLIENT_SECRET", os.getenv("SATUSEHAT_CLIENT_SECRET", "")),
        "organization_id": _secret(secrets, "SATUSEHAT_ORGANIZATION_ID", os.getenv("SATUSEHAT_ORGANIZATION_ID", "")),
    }
