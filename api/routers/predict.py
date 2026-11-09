"""
AERO.AI MVP1 — routers/predict.py
POST /v1/predict — main inference endpoint.

Flow: auth -> pydantic validation -> envelope warnings -> exact shape features ->
      one deterministic ONNX forward pass -> derived quantities -> response.
Uncertainty fields are returned as null: the exported graph is deterministic.
"""

import time
import structlog
from fastapi import APIRouter, Depends

from auth.keys import require_api_key
from config import settings
from schemas.request import PredictRequest
from schemas.response import PredictResponse
from services.inference import predict as run_predict
from services.derived import build_derived

log = structlog.get_logger()

router = APIRouter(prefix=f"/{settings.api_version}", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    key_meta: dict = Depends(require_api_key),
) -> PredictResponse:
    t_start = time.perf_counter()

    warnings: list[str] = req.envelope_warnings()
    partial = getattr(req, "_partial_cst_count", 0)
    if partial:
        warnings.append(
            f"Partial CST input ({partial}/10 fields) ignored; shape features computed from m, p, t."
        )

    x_raw = req.get_x_raw()
    extra_feats = req.get_extra_feats()
    t_c = req.t / 100.0

    result = run_predict(x_raw=x_raw, extra_feats_raw=extra_feats, re=req.re, t_c=t_c)
    derived = build_derived(result["Cl"], result["Cd"])
    inference_ms = (time.perf_counter() - t_start) * 1000.0

    log.info(
        "predict",
        user=key_meta["name"],
        m=req.m, p=req.p, t=req.t, aoa=req.aoa, re=req.re,
        cl=round(result["Cl"], 4), cd=round(result["Cd"], 5),
        stall=round(result["stall_margin"], 2),
        ms=round(inference_ms, 2), warns=len(warnings),
    )

    return PredictResponse(
        cl=round(result["Cl"], 5),
        cd=round(result["Cd"], 6),
        cm=round(result["Cm"], 5),
        stall_margin=round(result["stall_margin"], 3),
        ld_ratio=derived["ld_ratio"],
        aero_balance=derived["aero_balance"],
        cl_std=None, cd_std=None, confidence=None, confidence_tier="unavailable",
        warnings=warnings,
        inference_ms=round(inference_ms, 3),
        model_version=settings.model_version,
        user_name=key_meta["name"],
    )
