from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, Query
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install fastapi and uvicorn to run the SI-HIS API server.") from exc

from api.kemenkes_api import build_kemenkes_intelligence
from core.data_provider import get_source_catalog

app = FastAPI(
    title="SI-HIS Intelligence API",
    version="0.2.0",
    description="API layer for SI-HIS-derived health intelligence.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "SI-HIS Intelligence"}



@app.get("/api/data-sources")
def data_sources() -> dict[str, Any]:
    """Return the canonical SI-HIS source registry for integration discovery."""
    return {"sources": get_source_catalog()}

@app.get("/api/intelligence/kemenkes")
def kemenkes_intelligence(
    period: int = Query(14, ge=7, le=30),
    province: str | None = Query(None),
    district: str | None = Query(None),
    kecamatan: str | None = Query(None),
    village: str | None = Query(None),
    puskesmas: str | None = Query(None),
    disease: str | None = Query(None),
    source_id: str = Query("dummy"),
) -> dict[str, Any]:
    """Return Kemenkes intelligence for an isolated read/query scope.

    Query parameters define the requested analytical scope. They never mutate
    the canonical SI-HIS dataset or another user's request.
    """
    scope = {
        "province": province,
        "district": district,
        "kecamatan": kecamatan,
        "village": village,
        "puskesmas": puskesmas,
        "disease": disease,
    }
    return build_kemenkes_intelligence(period_days=period, scope=scope, source_id=source_id)
