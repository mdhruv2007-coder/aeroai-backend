"""
AERO.AI MVP1 — Fix #2: Mesh Quality Audit
Script: 01_mesh_quality_audit.py

Flags suspect rows in cd_train_locked.csv based on physics sanity bounds,
outputs a per-case report and a summary JSON with pass/fail gate.

Gate criterion: >=90% of cases must pass (i.e. <=10% flagged/quarantined).
If >5% quarantined → caller should remove and retrain.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# PATHS  (edit BASE_DIR if your layout differs)
# ─────────────────────────────────────────────
BASE_DIR = Path("/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1")
DATA_DIR = BASE_DIR / "data" / "processed"
SCRIPTS_DIR = BASE_DIR / "scripts"

INPUT_CSV = DATA_DIR / "cd_train_locked.csv"
REPORT_CSV = DATA_DIR / "mesh_quality_report.csv"
SUMMARY_JSON = DATA_DIR / "mesh_quality_summary.json"

RANDOM_SEED = 42  # never change

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
CD_MIN = 0.004
CD_MAX = 0.18
LD_MAX = 130.0

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading {path}")
    if not path.exists():
        sys.exit(f"[ERROR] Input file not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


def detect_cd_column(df: pd.DataFrame) -> str:
    """Case-insensitive search for the Cd column."""
    candidates = [c for c in df.columns if c.strip().lower() in ("cd", "drag_coeff", "drag_coefficient")]
    if not candidates:
        sys.exit("[ERROR] Cannot find a 'cd' column. Please check column names.")
    if len(candidates) > 1:
        print(f"[WARN] Multiple Cd candidates found: {candidates}. Using first: {candidates[0]}")
    return candidates[0]


def detect_cl_column(df: pd.DataFrame):
    """Return Cl column name or None if absent."""
    candidates = [c for c in df.columns if c.strip().lower() in ("cl", "lift_coeff", "lift_coefficient")]
    return candidates[0] if candidates else None


def detect_ld_column(df: pd.DataFrame):
    """Return L/D column name or None if absent."""
    candidates = [c for c in df.columns if c.strip().lower() in ("l/d", "ld", "lift_to_drag", "l_d")]
    return candidates[0] if candidates else None


def detect_converged_column(df: pd.DataFrame):
    """Return converged column name or None if absent."""
    candidates = [c for c in df.columns if c.strip().lower() in ("converged", "convergence", "is_converged")]
    return candidates[0] if candidates else None


# ─────────────────────────────────────────────
# MAIN AUDIT
# ─────────────────────────────────────────────

def run_audit(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    cd_col = detect_cd_column(df)
    cl_col = detect_cl_column(df)
    ld_col = detect_ld_column(df)
    conv_col = detect_converged_column(df)

    print(f"\n[INFO] Using columns  →  Cd: '{cd_col}' | Cl: {cl_col!r} | L/D: {ld_col!r} | converged: {conv_col!r}")

    # ── per-flag boolean columns ──────────────────────────────────────────
    df = df.copy()

    # Flag 1: Cd out of physical range
    cd = pd.to_numeric(df[cd_col], errors="coerce")
    df["flag_cd_low"]  = cd < CD_MIN
    df["flag_cd_high"] = cd > CD_MAX
    df["flag_cd_nan"]  = cd.isna()

    # Flag 2: L/D unrealistically high
    if ld_col:
        ld = pd.to_numeric(df[ld_col], errors="coerce")
        df["flag_ld_high"] = ld > LD_MAX
    elif cl_col:
        # compute L/D on the fly; guard against zero Cd
        cl = pd.to_numeric(df[cl_col], errors="coerce")
        ld = cl / cd.replace(0, np.nan)
        df["flag_ld_high"] = ld > LD_MAX
        print("[INFO] L/D column absent — computed from Cl/Cd on the fly.")
    else:
        df["flag_ld_high"] = False
        print("[WARN] Neither L/D nor Cl column found — L/D check skipped.")

    # Flag 3: Not converged
    if conv_col:
        conv = df[conv_col]
        # Handle boolean-like strings ('True'/'False', '1'/'0')
        if conv.dtype == object:
            conv = conv.astype(str).str.strip().str.lower().map(
                {"true": True, "1": True, "false": False, "0": False}
            )
        df["flag_not_converged"] = conv == False  # noqa: E712
    else:
        df["flag_not_converged"] = False
        print("[INFO] 'converged' column not found — convergence check skipped.")

    # ── composite quarantine flag ─────────────────────────────────────────
    flag_cols = ["flag_cd_low", "flag_cd_high", "flag_cd_nan",
                 "flag_ld_high", "flag_not_converged"]
    df["quarantine"] = df[flag_cols].any(axis=1)

    # ── reason string (human-readable) ────────────────────────────────────
    def reason(row):
        reasons = []
        if row["flag_cd_low"]:   reasons.append(f"cd<{CD_MIN}")
        if row["flag_cd_high"]:  reasons.append(f"cd>{CD_MAX}")
        if row["flag_cd_nan"]:   reasons.append("cd=NaN")
        if row["flag_ld_high"]:  reasons.append(f"L/D>{LD_MAX}")
        if row["flag_not_converged"]: reasons.append("not_converged")
        return "; ".join(reasons) if reasons else "OK"

    df["audit_reason"] = df.apply(reason, axis=1)

    return df, flag_cols


def build_summary(df: pd.DataFrame, flag_cols: list) -> dict:
    n = len(df)
    n_quarantine = int(df["quarantine"].sum())
    pct_quarantine = 100.0 * n_quarantine / n if n > 0 else 0.0
    pct_pass = 100.0 - pct_quarantine

    per_flag = {col: int(df[col].sum()) for col in flag_cols}

    gate_pass = pct_pass >= 90.0          # >=90% cases pass
    retrain_required = pct_quarantine > 5.0  # >5% quarantined → remove & retrain

    summary = {
        "input_file": str(INPUT_CSV),
        "total_cases": n,
        "quarantined": n_quarantine,
        "passed": n - n_quarantine,
        "pct_quarantined": round(pct_quarantine, 3),
        "pct_passed": round(pct_pass, 3),
        "gate_criterion": ">=90% cases pass",
        "gate_PASS": gate_pass,
        "retrain_required": retrain_required,
        "thresholds": {
            "cd_min": CD_MIN,
            "cd_max": CD_MAX,
            "ld_max": LD_MAX,
        },
        "per_flag_counts": per_flag,
    }
    return summary


def print_summary(summary: dict):
    sep = "─" * 55
    print(f"\n{sep}")
    print("  MESH QUALITY AUDIT — SUMMARY")
    print(sep)
    print(f"  Total cases    : {summary['total_cases']:,}")
    print(f"  Passed         : {summary['passed']:,}  ({summary['pct_passed']:.1f}%)")
    print(f"  Quarantined    : {summary['quarantined']:,}  ({summary['pct_quarantine']:.1f}%)")
    print(f"\n  Per-flag breakdown:")
    for flag, cnt in summary["per_flag_counts"].items():
        print(f"    {flag:<25} : {cnt:,}")
    print(f"\n  Gate (≥90% pass) : {'✅ PASS' if summary['gate_PASS'] else '❌ FAIL'}")
    print(f"  Retrain required  : {'⚠️  YES — quarantine >5%' if summary['retrain_required'] else '✅ NO'}")
    print(sep)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    np.random.seed(RANDOM_SEED)

    df_raw = load_data(INPUT_CSV)
    df_audited, flag_cols = run_audit(df_raw)
    summary = build_summary(df_audited, flag_cols)

    # ── FIX: Ensure pct_quarantine exists ─────────────────────────────────
    # (If build_summary forgot to calculate it, we calculate it here)
    if 'pct_quarantine' not in summary:
        total = summary.get('total', 0)
        quarantined = summary.get('quarantined', 0)
        summary['pct_quarantine'] = (quarantined / total * 100) if total > 0 else 0.0

    # ── write outputs ────────────────────────────────────────────────────
    # Report: include all original columns + audit columns
    audit_cols = flag_cols + ["quarantine", "audit_reason"]
    report_df = df_audited[list(df_raw.columns) + audit_cols]
    report_df.to_csv(REPORT_CSV, index=False)
    print(f"\n[INFO] Per-case report  → {REPORT_CSV}")

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary JSON     → {SUMMARY_JSON}")

    print_summary(summary)

    # ── optional: show a sample of quarantined rows ───────────────────────
    quarantined_df = df_audited[df_audited["quarantine"]]
    if not quarantined_df.empty:
        print("\n[INFO] Sample quarantined rows (up to 10):")
        show_cols = [c for c in df_raw.columns[:6]] + ["audit_reason"]
        print(quarantined_df[show_cols].head(10).to_string(index=True))

    # ── exit code for CI/pipeline gating ────────────────────────────────
    if not summary["gate_PASS"]:
        print("\n[FATAL] Gate FAILED — fewer than 90% of training cases pass quality check.")
        sys.exit(1)

    if summary["retrain_required"]:
        print("\n[ACTION REQUIRED] >5% cases quarantined. Remove flagged rows and retrain.")
        print(f"  → Quarantine index list saved in report: {REPORT_CSV}")
        # Non-fatal exit so the pipeline can continue to the removal step
        sys.exit(2)

    print("\n[OK] Mesh quality audit complete. Proceed to Fix #3.")
    sys.exit(0)


if __name__ == "__main__":
    main()
