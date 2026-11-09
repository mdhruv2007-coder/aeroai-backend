"""
AERO.AI - services/cst_features.py
Serving-time shape features, computed with the SAME procedure the training data used
(scripts/01_merge_and_clean.py: naca4_to_cst + max_thickness + le_radius).
Verified against all 709 unique airfoils in cd_val_locked.csv: max abs error 1.3e-15.
numpy + stdlib only. ~0.07 ms per call. Accepts continuous m, p, t.
"""
from math import comb
import numpy as np

_ORDER = 4
_X = np.linspace(0.0, 1.0, 300)
_MASK = (_X > 0.01) & (_X < 0.99)


def _basis(x: np.ndarray) -> np.ndarray:
    s = (x ** 0.5) * (1 - x)
    n = _ORDER - 1
    return np.column_stack([s * comb(n, i) * x ** i * (1 - x) ** (n - i) for i in range(_ORDER)])


_B = _basis(_X[_MASK])


def naca4_to_cst(m: float, p: float, t: float) -> np.ndarray:
    """m in percent, p in tenths, t in percent -> [upper_0..3, lower_0..3]."""
    mf, pf, tf = m / 100.0, p / 10.0, t / 100.0
    x = _X
    yt = 5 * tf * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                   + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    if mf == 0 or pf == 0:
        yc = np.zeros_like(x)
    else:
        yc = np.where(x < pf,
                      mf / pf ** 2 * (2 * pf * x - x ** 2),
                      mf / (1 - pf) ** 2 * ((1 - 2 * pf) + 2 * pf * x - x ** 2))
    cu = np.linalg.lstsq(_B, (yc + yt)[_MASK], rcond=None)[0]
    cl = np.linalg.lstsq(_B, (yc - yt)[_MASK], rcond=None)[0]
    return np.concatenate([cu, cl])


def naca4_to_extra_feats(m: float, p: float, t: float) -> np.ndarray:
    """[cst_0..cst_7, max_thickness, le_radius] as float32 (same order the adapter was trained on)."""
    tf = t / 100.0
    return np.concatenate([naca4_to_cst(m, p, t), [tf, 1.1019 * tf ** 2]]).astype(np.float32)
