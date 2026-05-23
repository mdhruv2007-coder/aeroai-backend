"""
AERO.AI MVP1 — services/derived.py
Compute derived aerodynamic quantities from raw model outputs.
These are pure functions — no model/scaler dependencies.
"""


def ld_ratio(cl: float, cd: float) -> float:
    """
    Lift-to-drag ratio.
    Guards against near-zero Cd to avoid division explosion.
    """
    if abs(cd) < 1e-9:
        return 0.0
    return round(cl / cd, 4)


def aero_balance(cl: float, cd: float) -> float:
    """
    Aerodynamic balance indicator: Cl / sqrt(Cl^2 + Cd^2).
    Ranges 0–1. Higher = more lift-dominant, lower = drag-dominant.
    Returns 0.0 for the zero-vector case.
    """
    mag = (cl ** 2 + cd ** 2) ** 0.5
    if mag < 1e-9:
        return 0.0
    return round(cl / mag, 4)


def build_derived(cl: float, cd: float) -> dict:
    """
    Convenience wrapper — returns both derived quantities in one call.
    Usage in router:
        derived = build_derived(result["Cl"], result["Cd"])
        response.ld_ratio     = derived["ld_ratio"]
        response.aero_balance = derived["aero_balance"]
    """
    return {
        "ld_ratio":     ld_ratio(cl, cd),
        "aero_balance": aero_balance(cl, cd),
    }
