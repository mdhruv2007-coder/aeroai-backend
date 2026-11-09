"""
AERO.AI - schemas/response.py
Uncertainty fields are Optional: the deployed ONNX graph is deterministic (dropout is removed at
export), so there is NO real uncertainty estimate. They are returned as null instead of fake zeros.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictResponse(BaseModel):
    cl: float = Field(..., description="Lift coefficient")
    cd: float = Field(..., description="Drag coefficient (as labelled in the CFD training set)")
    cm: float = Field(..., description="Pitching moment coefficient")
    stall_margin: float = Field(..., description="Degrees of AoA before stall (corrected, >= 0)")
    ld_ratio: float = Field(..., description="Cl / Cd")
    aero_balance: float = Field(..., description="Cl / sqrt(Cl^2 + Cd^2)")

    cl_std: Optional[float] = Field(None, description="Not available (null): model is deterministic")
    cd_std: Optional[float] = Field(None, description="Not available (null): model is deterministic")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0,
                                        description="Not available (null): no calibrated uncertainty yet")
    confidence_tier: str = Field("unavailable", description="'unavailable' until uncertainty is implemented")

    warnings: List[str] = Field(default_factory=list,
                                description="Non-fatal warnings, e.g. input outside the training envelope")
    inference_ms: float = Field(..., description="Server-side inference time in ms")
    model_version: str = Field(..., description="Model identifier")
    user_name: str = Field(..., description="Name associated with the API key")
