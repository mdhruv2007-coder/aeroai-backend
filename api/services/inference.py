"""
AERO.AI MVP1 — services/inference.py

Responsibilities:
  1. Load ONNX session + all 6 scalers once at startup (via load_model())
  2. naca4_to_extra_feats()   — NACA 4-digit → 10-element CST+geom array
  3. corrected_stall_margin() — apply -0.5° base + Re/thickness corrections
  4. predict()                — single deterministic forward pass
  5. predict_with_uncertainty() — 30-pass MC Dropout, returns mean/std/confidence

CRITICAL scaling rules (Session Transfer v6, Section 03_export_onnx output):
  ONNX inputs are SCALED:
    x_scaled          → cl_model x_scaler   (5 features: m, p, t, aoa, Re)
    extra_feats_scaled → extra_feats_scaler  (10 features: cst_0..7, max_thickness, le_radius)
  ONNX outputs are SCALED — unscale with:
    output[0] Cl           → cl_y_scaler.inverse_transform()
    output[1] Cd           → cd_y_scaler.inverse_transform()
    output[2] Cm           → cm_y_scaler.inverse_transform()
    output[3] stall_raw    → stall_y_scaler.inverse_transform() → corrected_stall_margin()
"""

import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort

from config import settings


# ------------------------------------------------------------------ #
#  Module-level singletons — loaded once at startup                   #
# ------------------------------------------------------------------ #
_ort_session:       ort.InferenceSession | None = None
_cl_x_scaler  = None
_cl_y_scaler  = None
_cm_y_scaler  = None
_cd_y_scaler  = None
_stall_y_scaler = None
_extra_feats_scaler = None


# ------------------------------------------------------------------ #
#  Startup loader                                                      #
# ------------------------------------------------------------------ #
def load_model() -> None:
    """
    Load ONNX session and all scalers into module-level singletons.
    Call once from FastAPI lifespan on startup. Never call per-request.
    """
    global _ort_session
    global _cl_x_scaler, _cl_y_scaler, _cm_y_scaler
    global _cd_y_scaler, _stall_y_scaler, _extra_feats_scaler

    model_dir = settings.model_dir

    # --- ONNX session ---
    onnx_path = model_dir / settings.onnx_model
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    _ort_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    # --- Scalers ---
    def _load(fname: str):
        p = model_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Scaler not found: {p}")
        with open(p, "rb") as f:
            return pickle.load(f)

    _cl_x_scaler      = _load(settings.cl_x_scaler)
    _cl_y_scaler      = _load(settings.cl_y_scaler)
    _cm_y_scaler      = _load(settings.cm_y_scaler)
    _cd_y_scaler      = _load(settings.cd_y_scaler)
    _stall_y_scaler   = _load(settings.stall_y_scaler)
    _extra_feats_scaler = _load(settings.extra_feats_scaler)


# ------------------------------------------------------------------ #
#  naca4_to_extra_feats                                               #
# ------------------------------------------------------------------ #
def naca4_to_extra_feats(m: float, p: float, t: float) -> np.ndarray:
    """
    Approximate CST coefficients + geometry features from NACA 4-digit params.
    Returns 10-element float32 array: [cst_0..cst_7, max_thickness, le_radius].

    Used as API fallback when caller omits CST fields.
    Same implementation as in 02_train_surrogate_multifidelity.py — do not drift.
    """
    m_frac = m / 100.0
    p_frac = p / 10.0
    t_frac = t / 100.0

    cst = np.array([
        0.2969 * t_frac,
        -0.1260 * t_frac,
        -0.3516 * t_frac + 0.5 * m_frac,
         0.2843 * t_frac + m_frac * p_frac,
        -0.1015 * t_frac,
         m_frac * (1 - p_frac),
         m_frac * 0.5,
         t_frac * 0.1,
    ], dtype=np.float32)

    max_thickness = np.float32(t_frac)
    le_radius     = np.float32(1.1019 * (t_frac ** 2))

    return np.concatenate([cst, [max_thickness, le_radius]])


# ------------------------------------------------------------------ #
#  corrected_stall_margin                                             #
# ------------------------------------------------------------------ #
def corrected_stall_margin(raw_sm: float, re: float, t_c: float) -> float:
    """
    Apply conservative corrections to raw stall margin output.
    Base: -0.5° (NOT -1.5° — BigFoil model is already conservative).
    Re factor:    -0.5° if Re < 1M | +0.3° if Re > 1.5M | 0 otherwise
    Thick factor: -0.5° if t/c < 0.10
    Floor at 0.0 — never return negative stall margin.

    Same implementation as in 02_train_surrogate_multifidelity.py — do not drift.
    """
    base         = -0.5
    re_factor    = -0.5 if re < 1e6 else (0.3 if re > 1.5e6 else 0.0)
    thick_factor = -0.5 if t_c < 0.10 else 0.0
    return max(0.0, raw_sm + base + re_factor + thick_factor)


# ------------------------------------------------------------------ #
#  Internal: single ONNX forward pass (scaled in → scaled out)       #
# ------------------------------------------------------------------ #
def _forward(x_scaled: np.ndarray,
             extra_feats_scaled: np.ndarray) -> np.ndarray:
    """
    Run one forward pass through the ONNX session.
    Inputs must already be scaled. Returns raw scaled output [N, 4].
    """
    if _ort_session is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    feeds = {
        "x_scaled":           x_scaled.astype(np.float32),
        "extra_feats_scaled": extra_feats_scaled.astype(np.float32),
    }
    return _ort_session.run(["outputs"], feeds)[0]   # shape [N, 4]


# ------------------------------------------------------------------ #
#  Internal: scale inputs                                             #
# ------------------------------------------------------------------ #
def _scale_inputs(x_raw: np.ndarray,
                  extra_feats_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scale raw inputs using loaded scalers. Both inputs must be 2-D [N, features]."""
    x_scaled           = _cl_x_scaler.transform(x_raw)
    extra_feats_scaled = _extra_feats_scaler.transform(extra_feats_raw)
    return x_scaled.astype(np.float32), extra_feats_scaled.astype(np.float32)


# ------------------------------------------------------------------ #
#  Internal: unscale outputs                                          #
# ------------------------------------------------------------------ #
def _unscale_outputs(scaled_out: np.ndarray) -> dict:
    """
    Unscale ONNX output [N, 4] → physical values.
    Output column order: [Cl, Cd, Cm, stall_raw]
    Each output uses its own y_scaler — do not mix them.
    """
    def _inv(scaler, col: int) -> np.ndarray:
        return scaler.inverse_transform(
            scaled_out[:, col].reshape(-1, 1)
        ).flatten()

    return {
        "Cl":        _inv(_cl_y_scaler,    0),
        "Cd":        _inv(_cd_y_scaler,    1),
        "Cm":        _inv(_cm_y_scaler,    2),
        "stall_raw": _inv(_stall_y_scaler, 3),
    }


# ------------------------------------------------------------------ #
#  Public: deterministic predict                                      #
# ------------------------------------------------------------------ #
def predict(x_raw: np.ndarray,
            extra_feats_raw: np.ndarray,
            re: float,
            t_c: float) -> dict:
    """
    Single deterministic forward pass.

    Args:
        x_raw           : [1, 5] float32 — [m, p, t, aoa, Re] unscaled
        extra_feats_raw : [1, 10] float32 — CST+geom unscaled
        re              : Reynolds number (for stall correction)
        t_c             : thickness/chord ratio (for stall correction)

    Returns dict with keys: Cl, Cd, Cm, stall_margin
    """
    x_s, ef_s = _scale_inputs(
        x_raw.reshape(1, 5),
        extra_feats_raw.reshape(1, 10),
    )
    scaled_out = _forward(x_s, ef_s)
    raw        = _unscale_outputs(scaled_out)

    return {
        "Cl":           float(raw["Cl"][0]),
        "Cd":           float(raw["Cd"][0]),
        "Cm":           float(raw["Cm"][0]),
        "stall_margin": corrected_stall_margin(float(raw["stall_raw"][0]), re, t_c),
    }


# ------------------------------------------------------------------ #
#  Public: MC Dropout uncertainty                                     #
# ------------------------------------------------------------------ #
def predict_with_uncertainty(x_raw: np.ndarray,
                              extra_feats_raw: np.ndarray,
                              re: float,
                              t_c: float) -> dict:
    """
    30-pass MC Dropout inference.
    The ONNX model has dropout baked in — each call produces a slightly
    different output because ONNX dropout uses random seeds per call.

    Confidence fix (Session Transfer v6, Section 17):
      Compute per-sample CV BEFORE averaging to avoid CV saturation at 0.0.
      cv_per_sample[i] = std_i / (|mean_i| + eps)
      confidence = clip(1 - mean(cv_per_sample), 0, 1)

    Returns dict with keys:
      Cl, Cd, Cm, stall_margin  — mean values
      Cl_std, Cd_std            — std across passes
      confidence                — scalar in [0, 1]
    """
    x_s, ef_s = _scale_inputs(
        x_raw.reshape(1, 5),
        extra_feats_raw.reshape(1, 10),
    )

    n_passes = settings.n_mc_passes
    cl_runs  = np.zeros(n_passes, dtype=np.float32)
    cd_runs  = np.zeros(n_passes, dtype=np.float32)
    cm_runs  = np.zeros(n_passes, dtype=np.float32)
    sm_runs  = np.zeros(n_passes, dtype=np.float32)

    for i in range(n_passes):
        scaled_out = _forward(x_s, ef_s)
        raw        = _unscale_outputs(scaled_out)
        cl_runs[i] = raw["Cl"][0]
        cd_runs[i] = raw["Cd"][0]
        cm_runs[i] = raw["Cm"][0]
        sm_runs[i] = raw["stall_raw"][0]

    # --- Per-sample CV before averaging (confidence fix) ---
    eps    = 1e-8
    cd_cv  = cd_runs.std() / (np.abs(cd_runs.mean()) + eps)   # scalar for N=1 batch
    cl_cv  = cl_runs.std() / (np.abs(cl_runs.mean()) + eps)
    mean_cv = 0.5 * (cd_cv + cl_cv)   # blend Cl and Cd uncertainty
    confidence = float(np.clip(1.0 - mean_cv, 0.0, 1.0))

    # --- Stall margin on mean stall_raw ---
    mean_stall_margin = corrected_stall_margin(
        float(sm_runs.mean()), re, t_c
    )

    return {
        "Cl":           float(cl_runs.mean()),
        "Cd":           float(cd_runs.mean()),
        "Cm":           float(cm_runs.mean()),
        "stall_margin": mean_stall_margin,
        "Cl_std":       float(cl_runs.std()),
        "Cd_std":       float(cd_runs.std()),
        "confidence":   confidence,
    }
