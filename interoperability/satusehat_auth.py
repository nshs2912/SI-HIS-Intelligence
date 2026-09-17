"""SATUSEHAT OAuth2 client-credentials authentication.

Credentials must be supplied through Streamlit secrets or environment variables.
No client_id/client_secret is stored in source control.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests

SANDBOX_OAUTH_BASE = "https://api-satusehat-stg.dto.kemkes.go.id/oauth2/v1"
PRODUCTION_OAUTH_BASE = "https://api-satusehat.kemkes.go.id/oauth2/v1"


class SATUSEHATAuthError(RuntimeError):
    pass


class SATUSEHATAuth:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        environment: str = "sandbox",
        timeout: int = 30,
    ) -> None:
        if not client_id or not client_secret:
            raise SATUSEHATAuthError("client_id dan client_secret wajib diisi.")
        env = environment.lower().strip()
        if env not in {"sandbox", "production"}:
            raise SATUSEHATAuthError("environment harus 'sandbox' atau 'production'.")
        self.client_id = client_id
        self.client_secret = client_secret
        self.environment = env
        self.timeout = timeout
        self.base_url = SANDBOX_OAUTH_BASE if env == "sandbox" else PRODUCTION_OAUTH_BASE
        self._token: Optional[str] = None
        self._expires_at = 0.0

    @classmethod
    def from_env(cls, environment: str = "sandbox", timeout: int = 30) -> "SATUSEHATAuth":
        client_id = os.getenv("SATUSEHAT_CLIENT_ID", "")
        client_secret = os.getenv("SATUSEHAT_CLIENT_SECRET", "")
        return cls(client_id, client_secret, environment, timeout)

    def access_token(self, force_refresh: bool = False) -> str:
        now = time.time()
        if not force_refresh and self._token and now < self._expires_at - 60:
            return self._token

        url = f"{self.base_url}/accesstoken"
        response = requests.post(
            url,
            params={"grant_type": "client_credentials"},
            data={"client_id": self.client_id, "client_secret": self.client_secret},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        if not response.ok:
            raise SATUSEHATAuthError(
                f"SATUSEHAT OAuth gagal ({response.status_code}): {response.text[:500]}"
            )

        payload: Dict[str, Any] = response.json()
        token = payload.get("access_token")
        if not token:
            raise SATUSEHATAuthError("Response OAuth tidak mengandung access_token.")

        expires_in = float(payload.get("expires_in", 0) or 0)
        self._token = str(token)
        self._expires_at = now + expires_in
        return self._token
