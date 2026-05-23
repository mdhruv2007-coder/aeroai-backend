"""
AERO.AI Prototype2 — Script 01: Merge & Clean
==============================================
Loads all raw data sources, computes CST for NACA shapes,
merges into a single master dataset, then splits into
Cl and Cd training sets.

Source tags:
  1 = XFOIL panel       → Cl + Cm only  (Cd NEVER used)
  2 = OpenFOAM/RANS CFD → Cl + Cd       (highest fidelity)

Cl model features : m, p, thickness, aoa, Re         (NACA rows only)
Cd model features : cst_0..7, aoa, Re, max_camber, max_thickness, le_radius  (CFD rows only)

Usage:
  cd "/mnt/d/Projects/Personal project 1/Prototype2"
  python scripts/01_merge_and_clean.py
"""

import pandas as pd
import numpy as np
from numpy.linalg import lstsq
from scipy.special import comb
from pathlib import Path
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path("/mnt/d/Projects/Personal project 1-AERO AI/Prototype2")
RAW    = BASE / "data" / "raw"
PROC   = BASE / "data" / "processed"

KANA_CSV  = RAW / "kanakaero_polars.csv"
XFOIL_CSV = RAW / "xfoil_polars.csv"
P1_DIR    = RAW / "p1_cfd"
SKIP_P1   = {"cfd_results_v3.csv"}          # broken file — skip permanently

OUT_MASTER = PROC / "master_dataset.csv"
OUT_CL     = PROC / "cl_training.csv"
OUT_CD     = PROC / "cd_training.csv"

CST_COLS   = [f"cst_{i}" for i in range(8)]
MASTER_COLS = (
    ["naca", "m", "p", "thickness", "aoa", "Re",
     "Cl", "Cd", "Cm", "data_source"] + CST_COLS
)

# Cd model: restrict to Re=100k (kanakaero range) — expand later if needed



# ── NACA helpers ───────────────────────────────────────────────────────────────
def is_naca4(s) -> bool:
    try:
        return len(str(s).strip().zfill(4)) == 4 and str(s).strip().zfill(4).isdigit()
    except Exception:
        return False


def parse_naca4(df: pd.DataFrame) -> pd.DataFrame:
    """Extract m, p, thickness digits from NACA 4-digit code."""
    df = df.copy()
    df["naca"] = df["naca"].astype(str).str.strip()
    mask = df["naca"].apply(is_naca4)
    df["m"] = np.nan
    df["p"] = np.nan
    df["thickness"] = np.nan
    if mask.any():
        n4 = df.loc[mask, "naca"].str.zfill(4)
        df.loc[mask, "m"]         = n4.str[0].astype(int)
        df.loc[mask, "p"]         = n4.str[1].astype(int)
        df.loc[mask, "thickness"] = n4.str[2:].astype(int)
    return df


# ── CST (Kulfan) ───────────────────────────────────────────────────────────────
_cst_cache: dict = {}

def _cst_basis(x: np.ndarray, order: int = 4) -> np.ndarray:
    S = (x ** 0.5) * (1 - x)
    N = order - 1
    cols = []
    for i in range(order):
        bi = comb(N, i, exact=True) * (x ** i) * ((1 - x) ** (N - i))
        cols.append(S * bi)
    return np.column_stack(cols)


def naca4_to_cst(m: int, p: int, t: int, order: int = 4) -> np.ndarray:
    """8 CST coefficients [upper_0..3, lower_0..3] for a NACA 4-digit airfoil."""
    m_f, p_f, t_f = m / 100.0, p / 10.0, t / 100.0
    x = np.linspace(0, 1, 300)
    yt = 5 * t_f * (
        0.2969 * np.sqrt(x) - 0.1260 * x
        - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )
    if m_f == 0 or p_f == 0:
        yc = np.zeros_like(x)
    else:
        yc = np.where(
            x < p_f,
            m_f / p_f**2 * (2 * p_f * x - x**2),
            m_f / (1 - p_f)**2 * ((1 - 2 * p_f) + 2 * p_f * x - x**2),
        )
    mask = (x > 0.01) & (x < 0.99)
    B = _cst_basis(x[mask], order=order)
    cu, *_ = lstsq(B, (yc + yt)[mask], rcond=None)
    cl, *_ = lstsq(B, (yc - yt)[mask], rcond=None)
    return np.concatenate([cu, cl])


def add_cst_naca(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach cst_0..7 for all rows that have valid m/p/thickness."""
    df = df.copy()
    for c in CST_COLS:
        df[c] = np.nan

    valid = df["m"].notna() & df["p"].notna() & df["thickness"].notna()
    shapes = df.loc[valid, ["m", "p", "thickness"]].drop_duplicates()
    print(f"    Computing CST for {len(shapes)} unique NACA shapes...", end=" ", flush=True)

    for _, row in shapes.iterrows():
        key = (int(row["m"]), int(row["p"]), int(row["thickness"]))
        if key not in _cst_cache:
            _cst_cache[key] = naca4_to_cst(*key)
        cst = _cst_cache[key]
        mask = (
            valid
            & (df["m"] == row["m"])
            & (df["p"] == row["p"])
            & (df["thickness"] == row["thickness"])
        )
        for i, c in enumerate(CST_COLS):
            df.loc[mask, c] = cst[i]

    print("done")
    return df


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_kanakaero() -> pd.DataFrame:
    """
    Kanakaero: 33,705 OpenFOAM CFD rows. CST already saved as cst_0..7.
    Re=100k. Source tag = 2.
    """
    if not KANA_CSV.exists():
        print("  SKIP  kanakaero : file not found")
        return pd.DataFrame()

    df = pd.read_csv(KANA_CSV)
    df["data_source"] = 2
    df["naca"] = "kana_ext"

    # CST already named correctly by 00_download_external.py
    missing = [c for c in CST_COLS if c not in df.columns]
    if missing:
        print(f"  ERROR kanakaero : CST columns missing {missing} — check source file")
        sys.exit(1)

    if "Cm" not in df.columns:
        df["Cm"] = np.nan

    print(f"  LOAD  kanakaero : {len(df):>7} rows  (Re=100k, CFD Cl+Cd, CST ready)")
    return df


def load_xfoil() -> pd.DataFrame:
    """
    XFOIL: panel-method polars. Cl + Cm only. Cd is set to NaN (never trusted).
    Source tag = 1.
    """
    if not XFOIL_CSV.exists():
        print("  SKIP  xfoil     : file not found")
        return pd.DataFrame()

    df = pd.read_csv(XFOIL_CSV)
    df["Cd"] = np.nan          # XFOIL Cd permanently excluded
    df["data_source"] = 1
    if "Cm" not in df.columns:
        df["Cm"] = np.nan

    df = parse_naca4(df)
    df = add_cst_naca(df)
    print(f"  LOAD  xfoil     : {len(df):>7} rows  (Cd=NaN, source=1)")
    return df


def load_p1_cfd() -> pd.DataFrame:
    """
    Prototype1 CFD datasets (OpenFOAM/RANS). Cl + Cd both trusted.
    Skips broken files listed in SKIP_P1.
    Source tag = 2.
    """
    if not P1_DIR.exists():
        print(f"  SKIP  p1_cfd    : directory not found at {P1_DIR}")
        return pd.DataFrame()

    frames = []
    for f in sorted(P1_DIR.glob("*.csv")):
        if f.name in SKIP_P1:
            print(f"  SKIP  p1/{f.name}: in skip list")
            continue

        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]

            # Normalise column names
            rename = {}
            for col in df.columns:
                cl = col.lower().strip()
                if   cl in ("naca", "airfoil", "foil"):                    rename[col] = "naca"
                elif cl in ("aoa", "alpha", "angle_of_attack", "angle"):   rename[col] = "aoa"
                elif cl in ("cl", "lift_coeff"):                           rename[col] = "Cl"
                elif cl in ("cd", "drag_coeff"):                           rename[col] = "Cd"
                elif cl in ("cm", "moment_coeff"):                         rename[col] = "Cm"
                elif cl in ("re", "reynolds", "reynolds_number"):          rename[col] = "Re"
                elif cl in ("velocity", "vel", "v", "speed"):             rename[col] = "velocity"
            df.rename(columns=rename, inplace=True)

            # Derive Re from velocity if missing
            if "Re" not in df.columns and "velocity" in df.columns:
                df["Re"] = (pd.to_numeric(df["velocity"], errors="coerce")
                            * 0.1 / 1.5e-5).round(0)

            required = ["naca", "aoa", "Cl", "Cd", "Re"]
            missing  = [c for c in required if c not in df.columns]
            if missing:
                print(f"  WARN  p1/{f.name}: missing {missing} — skipping")
                continue

            if "Cm" not in df.columns:
                df["Cm"] = np.nan

            df["data_source"] = 2
            frames.append(df)
            print(f"  LOAD  p1/{f.name}: {len(df)} rows")

        except Exception as e:
            print(f"  SKIP  p1/{f.name}: {e}")

    if not frames:
        print("  SKIP  p1_cfd    : no usable files")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = parse_naca4(combined)
    combined = add_cst_naca(combined)
    print(f"  LOAD  p1_cfd    : {len(combined):>7} rows total from {len(frames)} file(s)")
    return combined


# ── Clean ──────────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    for col in ["aoa", "Re", "Cl", "Cd", "Cm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["aoa"].between(-8, 20)]
    df = df[df["Re"].between(1e4, 1e7)]
    df = df[df["Cl"].between(-1.5, 2.5)]

    # Drop physically impossible Cd values (where Cd is present)
    has_cd  = df["Cd"].notna()
    bad_cd  = has_cd & ~df["Cd"].between(0.001, 0.5)
    df = df[~bad_cd]

    print(f"  Clean   : {n0} → {len(df)} rows  ({n0 - len(df)} dropped)")
    return df.reset_index(drop=True)


# ── Dedup ──────────────────────────────────────────────────────────────────────
def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    For NACA rows: keep highest-fidelity source per (naca, aoa, Re).
    Non-NACA rows (kanakaero): all kept — each is a unique airfoil shape.
    """
    n0 = len(df)
    is_naca  = df["naca"].apply(is_naca4)
    naca_df  = df[is_naca].copy()
    other_df = df[~is_naca].copy()

    # Sort ascending so highest source number (CFD=2) wins on keep="last"
    naca_df = naca_df.sort_values("data_source", ascending=True)
    naca_df = naca_df.drop_duplicates(subset=["naca", "aoa", "Re"], keep="last")

    out = pd.concat([naca_df, other_df], ignore_index=True)
    print(f"  Dedup   : {n0} → {len(out)} rows  ({n0 - len(out)} removed)")
    print(f"            NACA: {len(naca_df)} | Non-NACA (all kept): {len(other_df)}")
    return out.reset_index(drop=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    PROC.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AERO.AI Prototype2 — 01_merge_and_clean.py")
    print("=" * 60)

    # Load all sources
    frames = []
    for loader in [load_xfoil, load_p1_cfd]:
        df = loader()
        if not df.empty:
            frames.append(df)

    if not frames:
        print("\n  ERROR: No data loaded. Aborting.")
        sys.exit(1)

    # Align columns before concat
    for i, df in enumerate(frames):
        for col in MASTER_COLS:
            if col not in df.columns:
                df[col] = np.nan
        frames[i] = df[MASTER_COLS]

    master = pd.concat(frames, ignore_index=True)
    master = clean(master)
    master = dedup(master)
    master.to_csv(OUT_MASTER, index=False)

    # ── Cl training set ────────────────────────────────────────────────────────
    # NACA rows only (have m/p/thickness). All sources. Cl must be present.
    cl_mask = (
        master["Cl"].notna() &
        master["m"].notna() &
        master["p"].notna() &
        master["thickness"].notna()
    )
    cl_df = master[cl_mask][
        ["naca", "m", "p", "thickness", "aoa", "Re", "Cl", "Cm", "data_source"]
    ].copy()
    cl_df.to_csv(OUT_CL, index=False)

    # ── Cd training set ────────────────────────────────────────────────────────
    # CFD rows only (source=2). Cd present. CST complete. Re restricted to 100k.
    cd_mask = (
        master["Cd"].notna() &
        (master["data_source"] == 2) &
        master[CST_COLS].notna().all(axis=1)
    )
    cd_df = master[cd_mask][
        ["naca", "aoa", "Re", "Cd", "data_source"] + CST_COLS
    ].copy()

    # ── Physics features derived from NACA digits ──────────────────────────────
    # These make shape geometry explicit rather than forcing the model to infer
    # it from CST coefficients alone.
    #   max_camber    : m/100  — drives pressure drag via flow curvature
    #   max_thickness : t/100  — strongest single predictor of form drag
    #   le_radius     : 1.1019 * t²  — leading edge radius, drives separation onset
    n4 = cd_df["naca"].astype(str).str.zfill(4)
    cd_df["max_camber"]    = n4.str[0].astype(int) / 100.0
    cd_df["max_thickness"] = n4.str[2:].astype(int) / 100.0
    cd_df["le_radius"]     = 1.1019 * cd_df["max_thickness"] ** 2

    dropped_cst = (
        master["Cd"].notna() &
        (master["data_source"] == 2) &
        ~master[CST_COLS].notna().all(axis=1)
    ).sum()
    if dropped_cst > 0:
        print(f"  WARN  : {dropped_cst} Cd rows dropped — CST incomplete")

    cd_df.to_csv(OUT_CD, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    labels = {0: "UIUC experimental", 1: "XFOIL panel", 2: "CFD (OpenFOAM/RANS)"}
    print()
    print("=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    for src, grp in master.groupby("data_source"):
        cd_n   = grp["Cd"].notna().sum()
        cst_ok = grp[CST_COLS].notna().all(axis=1).sum()
        print(f"  Source {int(src)} ({labels.get(int(src), '?')}): "
              f"{len(grp):>6} rows | Cd: {cd_n} | CST complete: {cst_ok}")

    print(f"\n  Master   : {len(master):>7} rows → {OUT_MASTER}")
    print(f"  Cl train : {len(cl_df):>7} rows → {OUT_CL}")
    print(f"             features : m, p, thickness, aoa, Re")
    print(f"  Cd train : {len(cd_df):>7} rows → {OUT_CD}")
    print(f"             features : cst_0..7, aoa, Re, max_camber, max_thickness, le_radius  (13 total)")
    print(f"\n  Unique airfoils : {master['naca'].nunique()}")
    print(f"  AoA range       : {master['aoa'].min():.1f}° to {master['aoa'].max():.1f}°")
    print(f"  Re range        : {master['Re'].min():.0f} to {master['Re'].max():.0f}")
    print(f"\n  Next step: python scripts/02_train_surrogate.py")


if __name__ == "__main__":
    main()
