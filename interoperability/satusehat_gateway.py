"""Authenticated SATUSEHAT FHIR R4 gateway.

This layer deliberately separates transport/authentication from the SI-HIS
canonical model and ML pipeline. Use sandbox until the relevant SATUSEHAT
onboarding, profiles and test cases have been completed.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from .satusehat_auth import SATUSEHATAuth

SANDBOX_FHIR_BASE = "https://api-satusehat-stg.dto.kemkes.go.id/fhir-r4/v1"
PRODUCTION_FHIR_BASE = "https://api-satusehat.kemkes.go.id/fhir-r4/v1"


class SATUSEHATGatewayError(RuntimeError):
    pass


class SATUSEHATGateway:
    def __init__(self, auth: SATUSEHATAuth, timeout: int = 30) -> None:
        self.auth = auth
        self.timeout = timeout
        self.base_url = (
            SANDBOX_FHIR_BASE
            if auth.environment == "sandbox"
            else PRODUCTION_FHIR_BASE
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        token = self.auth.access_token()
        headers = kwargs.pop("headers", {}) or {}
        headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/fhir+json",
        })
        if "json" in kwargs:
            headers.setdefault("Content-Type", "application/fhir+json")
        response = requests.request(
            method,
            f"{self.base_url}/{path.lstrip('/')}",
            headers=headers,
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code == 401:
            token = self.auth.access_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            response = requests.request(
                method,
                f"{self.base_url}/{path.lstrip('/')}",
                headers=headers,
                timeout=self.timeout,
                **kwargs,
            )
        return response

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self._request("GET", path, params=params)
        if not response.ok:
            raise SATUSEHATGatewayError(
                f"GET {path} gagal ({response.status_code}): {response.text[:1000]}"
            )
        return response.json()

    def post(self, resource_type: str, resource: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request("POST", resource_type, json=resource)
        if not response.ok:
            raise SATUSEHATGatewayError(
                f"POST {resource_type} gagal ({response.status_code}): {response.text[:1000]}"
            )
        return response.json()

    def get_patient_by_nik(self, nik: str) -> Dict[str, Any]:
        return self.get(
            "Patient",
            params={"identifier": f"https://fhir.kemkes.go.id/id/nik|{nik}"},
        )

    def get_practitioner_by_nik(self, nik: str) -> Dict[str, Any]:
        return self.get(
            "Practitioner",
            params={"identifier": f"https://fhir.kemkes.go.id/id/nik|{nik}"},
        )
