"""
=================================================================
  AERO.AI MVP1 — 00_lock_validation_set.py
  Issue Fix #1: Lock Validation Set (Doc 2, Section 2)
=================================================================
  RUN THIS SCRIPT EXACTLY ONCE.
  Never re-run with a different seed.
  Never re-run after adding new data.
  The locked CSVs produced here are ground truth forever.
=================================================================
"""

import pandas as pd
import numpy as np
import os
import hashlib
import json
from datetime import datetime

# ── Fixed seed — NEVER change this ──────────────────────────────
RANDOM_SEED = 42
VAL_FRACTION = 0.15   # 15% held out for validation

# ── Base path ────────────────────────────────────────────────────
BASE_DIR = r"/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── Input files ──────────────────────────────────────────────────
CD_TRAINING_CSV = os.path.join(DATA_DIR, "cd_training.csv")
CL_TRAINING_CSV = os.path.join(DATA_DIR, "cl_training.csv")

# ── Output files — written ONCE, never overwritten ───────────────
CD_TRAIN_LOCKED  = os.path.join(DATA_DIR, "cd_train_locked.csv")
CD_VAL_LOCKED    = os.path.join(DATA_DIR, "cd_val_locked.csv")
CL_TRAIN_LOCKED  = os.path.join(DATA_DIR, "cl_train_locked.csv")
CL_VAL_LOCKED    = os.path.join(DATA_DIR, "cl_val_locked.csv")
LOCK_MANIFEST    = os.path.join(DATA_DIR, "validation_lock_manifest.json")

# ─────────────────────────────────────────────────────────────────
def sha256_of_file(filepath):
    """Compute SHA-256 hash of a CSV file for integrity checking."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_split(df, val_fraction, seed, label):
    """
    Stratified split across AoA bands and Re bands.
    Ensures validation set covers the full operating envelope,
    not just the dense centre of the distribution.
    Handles both lowercase (aoa, re) and capitalised (Re) column names.

    Returns (train_df, val_df, original_train_idx, original_val_idx).
    The original index sets are captured BEFORE reset_index so that
    check_no_contamination can do a reliable identity check.
    """
    df = df.copy().reset_index(drop=True)

    # ── Normalise column names to lowercase for internal use ──────
    col_map = {c: c.lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # Create stratification bands
    df["_aoa_band"] = pd.cut(
        df["aoa"],
        bins=[-5, 0, 5, 10, 15, 21],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    )
    df["_re_band"] = pd.cut(
        df["re"],
        bins=[0, 1_000_000, 1_500_000, 2_100_000],
        labels=[0, 1, 2],
        include_lowest=True
    )

    # Fill NaN bands (edge values) with nearest valid band
    df["_aoa_band"] = df["_aoa_band"].cat.add_categories([-1]).fillna(-1)
    df["_re_band"]  = df["_re_band"].cat.add_categories([-1]).fillna(-1)

    # Stratified sample
    np.random.seed(seed)
    val_idx = (
        df.groupby(["_aoa_band", "_re_band"], group_keys=False, observed=True)
          .apply(lambda x: x.sample(frac=val_fraction, random_state=seed))
          .index
    )

    train_df = df[~df.index.isin(val_idx)].drop(columns=["_aoa_band", "_re_band"])
    val_df   = df[ df.index.isin(val_idx)].drop(columns=["_aoa_band", "_re_band"])

    # ── Capture original indices BEFORE reset so contamination check
    #    compares genuine row identities, not freshly-assigned 0…N ints ──
    original_train_idx = set(train_df.index)
    original_val_idx   = set(val_df.index)

    # Print AoA and Re coverage of the val set
    print(f"\n  [{label}] Split summary:")
    print(f"    Train : {len(train_df):>6} rows")
    print(f"    Val   : {len(val_df):>6} rows  ({100*len(val_df)/(len(train_df)+len(val_df)):.1f}%)")
    print(f"    Val AoA range  : {val_df['aoa'].min():.1f}° → {val_df['aoa'].max():.1f}°")
    print(f"    Val Re  range  : {val_df['re'].min():.0f} → {val_df['re'].max():.0f}")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        original_train_idx,
        original_val_idx,
    )


def check_no_contamination(original_train_idx, original_val_idx, label):
    """
    Verify that no row appears in both train and val.
    Compares the original pre-reset index sets captured inside
    stratified_split — the only reliable row-identity check.
    """
    overlap = len(original_train_idx & original_val_idx)

    if overlap == 0:
        print(f"  [{label}] Contamination check: PASS — 0 rows shared between train and val")
    else:
        print(f"  [{label}] ⚠️  WARNING: {overlap} rows appear in BOTH train and val!")
        print(f"           This should not happen. Investigate before proceeding.")

    return overlap


def abort_if_already_locked():
    """
    Safety check: if locked files already exist, refuse to overwrite them.
    This enforces the ONE-RUN rule.
    """
    locked_files = [CD_TRAIN_LOCKED, CD_VAL_LOCKED, CL_TRAIN_LOCKED, CL_VAL_LOCKED]
    existing = [f for f in locked_files if os.path.exists(f)]

    if existing:
        print("\n" + "="*65)
        print("  ⛔  ABORT — LOCKED VALIDATION SET ALREADY EXISTS")
        print("="*65)
        for f in existing:
            print(f"    Found: {os.path.basename(f)}")
        print()
        print("  This script must only be run ONCE.")
        print("  The locked validation set must never be regenerated.")
        print()
        print("  If you genuinely need to re-lock (e.g. major data overhaul),")
        print("  manually delete the locked CSV files and the manifest,")
        print("  then re-run. Document the reason in your changelog.")
        print("="*65 + "\n")
        return True
    return False


# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  AERO.AI MVP1 — 00_lock_validation_set.py")
    print("=" * 65)

    # ── Safety check — do not overwrite existing lock ────────────
    if abort_if_already_locked():
        return

    # ── Load data ─────────────────────────────────────────────────
    print(f"\n  Loading cd_training.csv ...")
    cd_df = pd.read_csv(CD_TRAINING_CSV)
    print(f"    Loaded {len(cd_df)} rows | cols: {list(cd_df.columns)}")

    print(f"\n  Loading cl_training.csv ...")
    cl_df = pd.read_csv(CL_TRAINING_CSV)
    print(f"    Loaded {len(cl_df)} rows | cols: {list(cl_df.columns)}")

    # ── Column checks (case-insensitive) ─────────────────────────
    for name, df in [("cd_training", cd_df), ("cl_training", cl_df)]:
        lower_cols = [c.lower() for c in df.columns]
        for col in ["aoa", "re"]:
            if col not in lower_cols:
                raise ValueError(f"Column '{col}' (any case) missing from {name}.csv — cannot stratify.")

    # ── Compute source file hashes (for manifest) ─────────────────
    cd_source_hash = sha256_of_file(CD_TRAINING_CSV)
    cl_source_hash = sha256_of_file(CL_TRAINING_CSV)

    # ── Perform stratified splits ─────────────────────────────────
    print("\n  Performing stratified split on cd_training ...")
    cd_train, cd_val, cd_train_idx, cd_val_idx = stratified_split(
        cd_df, VAL_FRACTION, RANDOM_SEED, "Cd"
    )

    print("\n  Performing stratified split on cl_training ...")
    cl_train, cl_val, cl_train_idx, cl_val_idx = stratified_split(
        cl_df, VAL_FRACTION, RANDOM_SEED, "Cl"
    )

    # ── Contamination checks (use original pre-reset index sets) ──
    print()
    cd_overlap = check_no_contamination(cd_train_idx, cd_val_idx, "Cd")
    cl_overlap = check_no_contamination(cl_train_idx, cl_val_idx, "Cl")

    if cd_overlap > 0 or cl_overlap > 0:
        print("\n  ⚠️  Contamination detected. Saving anyway, but investigate before training.")

    # ── Save locked files ─────────────────────────────────────────
    print("\n  Saving locked CSVs ...")
    cd_train.to_csv(CD_TRAIN_LOCKED,  index=False)
    cd_val.to_csv(  CD_VAL_LOCKED,    index=False)
    cl_train.to_csv(CL_TRAIN_LOCKED,  index=False)
    cl_val.to_csv(  CL_VAL_LOCKED,    index=False)

    # ── Compute output file hashes ────────────────────────────────
    manifest = {
        "locked_on"        : datetime.now().isoformat(),
        "random_seed"       : RANDOM_SEED,
        "val_fraction"      : VAL_FRACTION,
        "source_files": {
            "cd_training.csv": {
                "path"    : CD_TRAINING_CSV,
                "rows"    : len(cd_df),
                "sha256"  : cd_source_hash
            },
            "cl_training.csv": {
                "path"    : CL_TRAINING_CSV,
                "rows"    : len(cl_df),
                "sha256"  : cl_source_hash
            }
        },
        "output_files": {
            "cd_train_locked.csv": {
                "rows"   : len(cd_train),
                "sha256" : sha256_of_file(CD_TRAIN_LOCKED)
            },
            "cd_val_locked.csv": {
                "rows"   : len(cd_val),
                "sha256" : sha256_of_file(CD_VAL_LOCKED)
            },
            "cl_train_locked.csv": {
                "rows"   : len(cl_train),
                "sha256" : sha256_of_file(CL_TRAIN_LOCKED)
            },
            "cl_val_locked.csv": {
                "rows"   : len(cl_val),
                "sha256" : sha256_of_file(CL_VAL_LOCKED)
            }
        },
        "contamination_check": {
            "cd_overlap_rows": cd_overlap,
            "cl_overlap_rows": cl_overlap
        },
        "golden_rule": (
            "This validation set is locked forever. "
            "All new autoloop cases go into TRAIN only. "
            "Never add rows to the val CSVs. "
            "Report R² on val set only — never on training data."
        )
    }

    with open(LOCK_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  LOCK COMPLETE")
    print("=" * 65)
    print(f"  cd_train_locked.csv  →  {len(cd_train):>6} rows  (training)")
    print(f"  cd_val_locked.csv    →  {len(cd_val):>6} rows  (validation — NEVER train on this)")
    print(f"  cl_train_locked.csv  →  {len(cl_train):>6} rows  (training)")
    print(f"  cl_val_locked.csv    →  {len(cl_val):>6} rows  (validation — NEVER train on this)")
    print(f"\n  Manifest saved → validation_lock_manifest.json")
    print()
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  GOLDEN RULE (enforced by manifest)                  │")
    print("  │  Every new autoloop case → train only.               │")
    print("  │  Val set is frozen from this moment forward.         │")
    print("  │  R² you report = R² on val set. Nothing else.        │")
    print("  └──────────────────────────────────────────────────────┘")
    print()
    print("  NEXT STEP:")
    print("    Run Issue Fix #2 — Mesh Quality Audit")
    print("    (01_mesh_quality_audit.py)")
    print("=" * 65)


if __name__ == "__main__":
    main()
