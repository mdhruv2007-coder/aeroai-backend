"""
AERO.AI MVP1 — schemas/response.py
PredictResponse — full output schema for POST /v1/predict.

Output fields (from Session Transfer v6, Section 11):
  Aerodynamic:  cl, cd, cm, stall_margin, ld_ratio, aero_balance
  Uncertainty:  cl_std, cd_std, confidence  (MC Dropout — 30 passes)
  Metadata:     warnings, inference_ms, model_version, user_name
  Confidence tiers: > 0.80 → green | 0.50–0.80 → amber | < 0.50 → red
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictResponse(BaseModel):

    # ------------------------------------------------------------------ #
    #  Aerodynamic outputs                                                #
    # ------------------------------------------------------------------ #
    cl: float = Field(..., description="Lift coefficient")
    cd: float = Field(..., description="Drag coefficient")
    cm: float = Field(..., description="Pitching moment coefficient")
    stall_margin: float = Field(
        ...,
        description=(
            "Degrees of AoA remaining before stall. "
            "Corrected for Re and thickness. Always >= 0."
        )
    )
    ld_ratio: float = Field(..., description="Lift-to-drag ratio (Cl / Cd)")
    aero_balance: float = Field(
        ...,
        description=(
            "Aerodynamic balance indicator: Cl / (Cl^2 + Cd^2)^0.5. "
            "Ranges 0–1; higher = more lift-dominant."
        )
    )

    # ------------------------------------------------------------------ #
    #  Uncertainty (MC Dropout — 30 forward passes through adapter)      #
    # ------------------------------------------------------------------ #
    cl_std: float = Field(..., description="Std deviation of Cl across MC passes")
    cd_std: float = Field(..., description="Std deviation of Cd across MC passes")
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description=(
            "Model confidence score (0–1). "
            "Derived from per-sample CV of Cd MC passes. "
            "Green > 0.80 | Amber 0.50–0.80 | Red < 0.50"
        )
    )
    confidence_tier: str = Field(
        ...,
        description="'green', 'amber', or 'red' — computed from confidence score."
    )

    # ------------------------------------------------------------------ #
    #  Metadata                                                          #
    # ------------------------------------------------------------------ #
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal warnings. Examples: Re below 800k extrapolation warning, "
            "CST fallback used, partial CST fields detected."
        )
    )
    inference_ms: float = Field(
        ...,
        description="Total server-side inference time in milliseconds (target p95 < 10ms)."
    )
    model_version: str = Field(..., description="Model identifier string.")
    user_name: str = Field(..., description="Name associated with the API key used.")

    # ------------------------------------------------------------------ #
    #  Convenience class method                                          #
    # ------------------------------------------------------------------ #
    @classmethod
    def confidence_to_tier(cls, confidence: float) -> str:
        """Map raw confidence score to colour tier string."""
        if confidence > 0.80:
            return "green"
        elif confidence >= 0.50:
            return "amber"
        else:
            return "red"
