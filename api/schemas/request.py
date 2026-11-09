"""
AERO.AI MVP1 — schemas/request.py
PredictRequest — validated input schema for POST /v1/predict.

Rules (from Session Transfer v6, Section 17 + 20):
  - m, p, t, aoa, re  → required, hard bounds via Field validators
  - cst_0..cst_7, max_thickness, le_radius → all optional
    If any CST field is omitted, inference.py calls naca4_to_extra_feats() as fallback
  - Re hard floor 500k (pydantic rejects below), soft warning 800k (returned in response)
  - Re hard ceiling 2M (pydantic rejects above)
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional
import numpy as np


class PredictRequest(BaseModel):

    # ------------------------------------------------------------------ #
    #  NACA 4-digit geometry — REQUIRED                                   #
    # ------------------------------------------------------------------ #
    m: float = Field(
        ...,
        ge=0.0, le=9.5,
        description="Max camber as % chord (0–9.5). NACA first digit."
    )
    p: float = Field(
        ...,
        ge=0.0, le=9.0,
        description="Camber position in tenths of chord (0–9). NACA second digit."
    )
    t: float = Field(
        ...,
        ge=6.0, le=24.0,
        description="Max thickness as % chord (6–24). NACA last two digits."
    )

    # ------------------------------------------------------------------ #
    #  Flow conditions — REQUIRED                                         #
    # ------------------------------------------------------------------ #
    aoa: float = Field(
        ...,
        ge=-6.0, le=20.0,
        description="Angle of attack in degrees (−6 to +20)."
    )
    re: float = Field(
        ...,
        ge=500_000, le=2_000_000,
        description=(
            "Reynolds number (500k–2M). Hard floor 500k. "
            "Below 800k a soft warning is returned — predictions are extrapolated."
        )
    )

    # ------------------------------------------------------------------ #
    #  CST + geometry features — OPTIONAL                                 #
    #  8 CST coefficients + max_thickness + le_radius = 10 features       #
    #  Server falls back to naca4_to_extra_feats(m, p, t) if omitted.     #
    # ------------------------------------------------------------------ #
    cst_0: Optional[float] = Field(None, description="CST coefficient 0 (upper surface leading edge)")
    cst_1: Optional[float] = Field(None, description="CST coefficient 1")
    cst_2: Optional[float] = Field(None, description="CST coefficient 2")
    cst_3: Optional[float] = Field(None, description="CST coefficient 3")
    cst_4: Optional[float] = Field(None, description="CST coefficient 4")
    cst_5: Optional[float] = Field(None, description="CST coefficient 5")
    cst_6: Optional[float] = Field(None, description="CST coefficient 6")
    cst_7: Optional[float] = Field(None, description="CST coefficient 7 (lower surface)")
    max_thickness: Optional[float] = Field(None, description="Max thickness as fraction of chord (e.g. 0.12)")
    le_radius:     Optional[float] = Field(None, description="Leading edge radius as fraction of chord")

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #
    @property
    def cst_complete(self) -> bool:
        """True only when ALL 10 CST+geometry fields are provided."""
        return all(v is not None for v in [
            self.cst_0, self.cst_1, self.cst_2, self.cst_3,
            self.cst_4, self.cst_5, self.cst_6, self.cst_7,
            self.max_thickness, self.le_radius,
        ])

    def get_extra_feats(self) -> np.ndarray:
        """
        Return the 10-element CST+geometry array ready for the scaler.
        Uses caller-provided values if all 10 are present.
        Otherwise computes them exactly from (m, p, t) via services.cst_features.
        """
        if self.cst_complete:
            return np.array([
                self.cst_0, self.cst_1, self.cst_2, self.cst_3,
                self.cst_4, self.cst_5, self.cst_6, self.cst_7,
                self.max_thickness, self.le_radius,
            ], dtype=np.float32)

        # Exact shape features from the NACA digits (same procedure as the training data)
        from services.cst_features import naca4_to_extra_feats
        return naca4_to_extra_feats(self.m, self.p, self.t)

    def envelope_warnings(self) -> list:
        """Warnings for inputs the data barely covers (cambered airfoils in the data: m 1-9, p 2-7; AoA >= -4)."""
        w = []
        if self.m > 0 and (self.p < 2.0 or self.p > 7.0):
            w.append("Camber position p is outside 2-7, where cambered training airfoils lie. Treat as extrapolation.")
        if 0 < self.m < 1.0:
            w.append("Camber m is between 0 and 1%; training has either symmetric (m=0) or m >= 1. Treat as interpolation.")
        if self.m > 9.0:
            w.append("Camber m above 9% is outside the training data. Treat as extrapolation.")
        if self.aoa < -4.0:
            w.append("AoA below -4 deg is not represented in the locked validation data (min -4). Treat as extrapolation.")
        return w

    def get_x_raw(self) -> np.ndarray:
        """Return the 5-element backbone input [m, p, t, aoa, Re]."""
        return np.array([self.m, self.p, self.t, self.aoa, self.re], dtype=np.float32)

    @model_validator(mode="after")
    def check_cst_all_or_none(self) -> "PredictRequest":
        """
        Warn-only validator: CST fields should either all be provided or all omitted.
        Partial CST is silently completed via NACA fallback (not rejected),
        but this catches accidental partial submissions during development.
        """
        cst_fields = [
            self.cst_0, self.cst_1, self.cst_2, self.cst_3,
            self.cst_4, self.cst_5, self.cst_6, self.cst_7,
            self.max_thickness, self.le_radius,
        ]
        n_provided = sum(v is not None for v in cst_fields)
        if 0 < n_provided < 10:
            # Partial — allowed but router will log a warning in the response
            # Store count so router can detect this case
            object.__setattr__(self, "_partial_cst_count", n_provided)
        return self

    model_config = {"frozen": False}
