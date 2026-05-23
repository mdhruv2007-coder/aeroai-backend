"""
AERO.AI MVP1 — routers/predict.py
POST /v1/predict — main inference endpoint.

Flow:
  1. Auth       → require_api_key() validates X-API-Key header
  2. Validation → PredictRequest (pydantic) validates + bounds-checks inputs
  3. Warnings   → build warnings list (Re soft floor, CST fallback, partial CST)
  4. Inference  → predict_with_uncertainty() — 30-pass MC Dropout
  5. Derived    → build_derived() — L/D, aero_balance
  6. Response   → PredictResponse assembled and returned
"""

import time
import structlog
from fastapi import APIRouter, Depends

from auth.keys import require_api_key
from config import settings
from schemas.request import PredictRequest
from schemas.response import PredictResponse
from services.inference import predict_with_uncertainty
from services.derived import build_derived

log = structlog.get_logger()

router = APIRouter(prefix=f"/{settings.api_version}", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    key_meta: dict = Depends(require_api_key),
) -> PredictResponse:
    """
    Run aerodynamic inference for a NACA 4-digit airfoil.

    Required inputs : m, p, t, aoa, re
    Optional inputs : cst_0..cst_7, max_thickness, le_radius
                      (server falls back to NACA approximation if omitted)

    Returns         : cl, cd, cm, stall_margin, ld_ratio, aero_balance,
                      cl_std, cd_std, confidence, confidence_tier,
                      warnings, inference_ms, model_version, user_name
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    #  Step 1 — Build warnings list                                       #
    # ------------------------------------------------------------------ #
    warnings: list[str] = []

    # Re soft floor
    if req.re < settings.re_soft_min:
        warnings.append(
            f"Re={req.re:.0f} is below 800k. Predictions are extrapolated — "
            "use with caution."
        )

    # CST fallback — all 10 omitted → using NACA approximation
    if not req.cst_complete:
        partial_count = getattr(req, "_partial_cst_count", 0)
        if partial_count > 0:
            warnings.append(
                f"Partial CST input detected ({partial_count}/10 fields provided). "
                "Falling back to NACA 4-digit approximation for all CST features."
            )
        else:
            warnings.append(
                "CST fields not provided. Using NACA 4-digit approximation "
                "for extra_feats. Supply cst_0..cst_7 + max_thickness + le_radius "
                "for higher accuracy."
            )

    # ------------------------------------------------------------------ #
    #  Step 2 — Prepare inputs                                            #
    # ------------------------------------------------------------------ #
    x_raw          = req.get_x_raw()           # [5]  m, p, t, aoa, Re
    extra_feats    = req.get_extra_feats()     # [10] CST+geom (real or fallback)
    t_c            = req.t / 100.0             # thickness/chord for stall correction

    # ------------------------------------------------------------------ #
    #  Step 3 — MC Dropout inference (30 passes)                          #
    # ------------------------------------------------------------------ #
    result = predict_with_uncertainty(
        x_raw        = x_raw,
        extra_feats_raw = extra_feats,
        re           = req.re,
        t_c          = t_c,
    )

    # ------------------------------------------------------------------ #
    #  Step 4 — Derived quantities                                        #
    # ------------------------------------------------------------------ #
    derived = build_derived(result["Cl"], result["Cd"])

    # ------------------------------------------------------------------ #
    #  Step 5 — Confidence tier                                           #
    # ------------------------------------------------------------------ #
    tier = PredictResponse.confidence_to_tier(result["confidence"])

    # ------------------------------------------------------------------ #
    #  Step 6 — Timing                                                    #
    # ------------------------------------------------------------------ #
    inference_ms = (time.perf_counter() - t_start) * 1000.0

    # ------------------------------------------------------------------ #
    #  Step 7 — Log (structlog — one line per request)                    #
    # ------------------------------------------------------------------ #
    log.info(
        "predict",
        user   = key_meta["name"],
        m=req.m, p=req.p, t=req.t, aoa=req.aoa, re=req.re,
        cl     = round(result["Cl"], 4),
        cd     = round(result["Cd"], 5),
        stall  = round(result["stall_margin"], 2),
        conf   = round(result["confidence"], 3),
        tier   = tier,
        ms     = round(inference_ms, 2),
        warns  = len(warnings),
    )

    # ------------------------------------------------------------------ #
    #  Step 8 — Assemble and return response                              #
    # ------------------------------------------------------------------ #
    return PredictResponse(
        cl            = round(result["Cl"], 5),
        cd            = round(result["Cd"], 6),
        cm            = round(result["Cm"], 5),
        stall_margin  = round(result["stall_margin"], 3),
        ld_ratio      = derived["ld_ratio"],
        aero_balance  = derived["aero_balance"],
        cl_std        = round(result["Cl_std"], 6),
        cd_std        = round(result["Cd_std"], 7),
        confidence    = round(result["confidence"], 4),
        confidence_tier = tier,
        warnings      = warnings,
        inference_ms  = round(inference_ms, 3),
        model_version = settings.model_version,
        user_name     = key_meta["name"],
    )
