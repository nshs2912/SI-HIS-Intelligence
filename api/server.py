from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, Query
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install fastapi and uvicorn to run the SI-HIS API server.") from exc

from api.kemenkes_api import build_kemenkes_intelligence

app = FastAPI(
    title="SI-HIS Intelligence API",
    version="0.2.0",
    description="API layer for SI-HIS-derived health intelligence.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "SI-HIS Intelligence"}


@app.get("/api/intelligence/kemenkes")
def kemenkes_intelligence(
    period: int = Query(14, ge=7, le=30),
    province: str | None = Query(None),
    district: str | None = Query(None),
    kecamatan: str | None = Query(None),
    village: str | None = Query(None),
    puskesmas: str | None = Query(None),
    disease: str | None = Query(None),
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
    return build_kemenkes_intelligence(period_days=period, scope=scope)
