"""
00_generate_lhs.py
AERO.AI MVP — CFD Campaign
Generate 50 Latin Hypercube Sampling (LHS) parameter combinations
targeting the Cd-sparse region: high AoA (8-14 deg), high Re (800k-2M),
high-camber geometries.

Save to:
  /mnt/d/AEROAI/MVP1/cfd_campaign/scripts/
Run with:
  python3 00_generate_lhs.py
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import os

# ── Output path ────────────────────────────────────────────────────────────────
BASE = "/mnt/d/AEROAI/MVP1/cfd_campaign"
OUT_CSV = os.path.join(BASE, "lhs_cases.csv")

# ── Parameter bounds (targeting Cd-sparse region) ─────────────────────────────
# Each row: [lower_bound, upper_bound]
# Parameters: AoA, Re, max_camber(m), camber_pos(p), thickness(t_c)
# These map to NACA 4-digit: NACA mptt → e.g. NACA 2412 → m=0.02, p=0.4, t=0.12
PARAMS = {
    "AoA":        (8.0,    14.0),   # degrees — Cd-sparse high-AoA region
    "Re":         (800e3,  2e6),    # Reynolds number
    "m":          (0.02,   0.09),   # max camber (fraction of chord) — high camber
    "p":          (0.3,    0.6),    # camber position (fraction of chord)
    "t_c":        (0.08,   0.15),   # thickness-to-chord ratio
}

N_CASES = 50
SEED    = 42

# ── Generate LHS ───────────────────────────────────────────────────────────────
sampler = qmc.LatinHypercube(d=len(PARAMS), seed=SEED)
sample  = sampler.random(n=N_CASES)

lower = np.array([v[0] for v in PARAMS.values()])
upper = np.array([v[1] for v in PARAMS.values()])
scaled = qmc.scale(sample, lower, upper)

df = pd.DataFrame(scaled, columns=PARAMS.keys())
df.index.name = "case_id"
df.index = df.index + 1  # 1-indexed

# ── Derive NACA 4-digit string for each case ───────────────────────────────────
def to_naca4(m, p, tc):
    """Convert fractional params to NACA 4-digit string e.g. '2412'"""
    m_d  = int(round(m  * 100))      # first digit  (max camber %)
    p_d  = int(round(p  * 10))       # second digit (camber position tenths)
    tc_d = int(round(tc * 100))      # last two digits (thickness %)
    return f"{m_d}{p_d}{tc_d:02d}"

df["naca4"] = df.apply(lambda r: to_naca4(r["m"], r["p"], r["t_c"]), axis=1)

# Re as integer for cleaner display
df["Re"] = df["Re"].astype(int)

# Round floats
df["AoA"] = df["AoA"].round(3)
df["m"]   = df["m"].round(4)
df["p"]   = df["p"].round(4)
df["t_c"] = df["t_c"].round(4)

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(BASE, exist_ok=True)
df.to_csv(OUT_CSV)

print(f"✅ Generated {N_CASES} LHS cases → {OUT_CSV}")
print(f"\nParameter ranges:")
for k, (lo, hi) in PARAMS.items():
    print(f"  {k:6s}: [{lo:.4g} → {hi:.4g}]")
print(f"\nFirst 5 cases:")
print(df.head())
print(f"\nNACA airfoils sampled: {df['naca4'].unique()[:10].tolist()} ...")
