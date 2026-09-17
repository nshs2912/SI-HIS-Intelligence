from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, Query
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install fastapi and uvicorn to run the SI-HIS API server.") from exc

from api.kemenkes_api import build_kemenkes_intelligence

app = FastAPI(
    title="SI-HIS Intelligence API",
    version="0.1.0",
    description="API layer for SI-HIS-derived health intelligence.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "SI-HIS Intelligence"}


@app.get("/api/intelligence/kemenkes")
def kemenkes_intelligence(
    period: int = Query(14, ge=7, le=30),
) -> dict[str, Any]:
    return build_kemenkes_intelligence(period_days=period)
