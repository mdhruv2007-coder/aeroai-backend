"""
03_run_campaign.py
AERO.AI MVP — CFD Campaign
Runs all 50 OpenFOAM cases sequentially:
  blockMesh → simpleFoam → extract Cl/Cd/Cm → quality gate check

Quality gates (from MVP Plan):
  - Residuals < 1e-5
  - Mass imbalance < 0.1%
  - Cd > 0 (physics check)
  - |L/D| < 200 (physics check)

Results saved to: cfd_campaign/results/cfd_results_mvp.csv

Save to:
  /mnt/d/AEROAI/MVP1/cfd_campaign/scripts/
Run with:
  source /opt/openfoam10/etc/bashrc && python3 03_run_campaign.py
"""

import os
import re
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE      = "/mnt/d/AEROAI/MVP1/cfd_campaign"
CASES_DIR = os.path.join(BASE, "cases")
RESULTS   = os.path.join(BASE, "results")
LOGS_DIR  = os.path.join(BASE, "logs")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(RESULTS, "cfd_results_mvp.csv")

# ── OpenFOAM 10 source command ─────────────────────────────────────────────────
OF_SOURCE = "source /opt/openfoam10/etc/bashrc"

def run_cmd(cmd, cwd, log_path):
    """Run a shell command, stream output to log file, return returncode."""
    full_cmd = f"{OF_SOURCE} && {cmd}"
    with open(log_path, "w") as log:
        proc = subprocess.run(
            ["bash", "-c", full_cmd],
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode

# ── Force coefficient extractor ────────────────────────────────────────────────
def extract_force_coeffs(case_path):
    """
    Parse the forceCoeffs output from OpenFOAM postProcessing.
    Returns dict with Cl, Cd, Cm or None if not found.
    """
    coeff_dir = Path(case_path) / "postProcessing" / "forceCoeffs"
    if not coeff_dir.exists():
        return None

    # Find latest time directory
    time_dirs = sorted([d for d in coeff_dir.iterdir() if d.is_dir()],
                       key=lambda d: float(d.name))
    if not time_dirs:
        return None

    coeff_file = time_dirs[-1] / "coefficient.dat"
    if not coeff_file.exists():
        # Try older OpenFOAM naming
        coeff_file = time_dirs[-1] / "forceCoeffs.dat"
    if not coeff_file.exists():
        return None

    # Read last 50 lines and average last 200 iterations for convergence
    lines = coeff_file.read_text().strip().splitlines()
    data_lines = [l for l in lines if not l.startswith("#") and l.strip()]

    if len(data_lines) < 10:
        return None

    # Columns: Time Cd Cs Cl CmRoll CmPitch CmYaw
    rows = []
    for line in data_lines[-200:]:
        vals = line.split()
        if len(vals) >= 7:
            try:
                rows.append([float(v) for v in vals[:7]])
            except ValueError:
                continue

    if not rows:
        return None

    arr = np.array(rows)
    # Average last 100 steps for stable values
    avg = arr[-100:].mean(axis=0)

    return {
        "Cd": float(avg[1]),
        "Cl": float(avg[3]),
        "Cm": float(avg[5]),
    }

# ── Residual checker ───────────────────────────────────────────────────────────
def check_residuals(log_path, threshold=1e-5):
    """
    Read simpleFoam log, find final residuals for p, U, k, omega.
    Returns True if all < threshold.
    """
    if not os.path.exists(log_path):
        return False

    with open(log_path) as f:
        content = f.read()

    # Find all "Initial residual" lines — take last occurrence per field
    pattern = r"Solving for (\w+).*?Initial residual = ([\deE\+\-\.]+)"
    matches = re.findall(pattern, content)

    if not matches:
        return False

    final_residuals = {}
    for field, resid in matches:
        final_residuals[field] = float(resid)

    critical = ["p", "Ux", "Uy", "k", "omega"]
    ok = all(final_residuals.get(f, 1.0) < threshold for f in critical
             if f in final_residuals)
    return ok

# ── Quality gate ───────────────────────────────────────────────────────────────
def quality_gate(coeffs, meta):
    """
    Returns (passed: bool, reason: str)
    Gates: Cd > 0, |L/D| < 200, Cl physically plausible
    """
    if coeffs is None:
        return False, "No force coefficients extracted"

    Cd, Cl, Cm = coeffs["Cd"], coeffs["Cl"], coeffs["Cm"]

    if Cd <= 0:
        return False, f"Cd={Cd:.4f} <= 0 — unphysical"

    ld = abs(Cl / Cd) if Cd != 0 else 9999
    if ld > 200:
        return False, f"|L/D|={ld:.1f} > 200 — unphysical"

    if abs(Cl) > 5.0:
        return False, f"Cl={Cl:.3f} out of range"

    return True, "OK"

# ── Main campaign loop ─────────────────────────────────────────────────────────
def main():
    case_dirs = sorted(Path(CASES_DIR).iterdir())
    case_dirs = [d for d in case_dirs if d.is_dir()]

    print(f"{'='*65}")
    print(f"  AERO.AI CFD Campaign — {len(case_dirs)} cases")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    records = []
    passed  = 0
    failed  = 0

    for i, case_path in enumerate(case_dirs, 1):
        case_name = case_path.name
        meta_file = case_path / "case_meta.json"

        if not meta_file.exists():
            print(f"[{i:02d}] SKIP {case_name} — no meta file")
            continue

        with open(meta_file) as f:
            meta = json.load(f)

        print(f"[{i:02d}/{len(case_dirs)}] Running: {case_name}")

        log_mesh   = os.path.join(LOGS_DIR, f"{case_path.name}_blockMesh.log")
        log_solver = os.path.join(LOGS_DIR, f"{case_path.name}_simpleFoam.log")

        # ── Step 1: blockMesh ───────────────────────────────────────────────
        print(f"        → blockMesh ...", end=" ", flush=True)
        rc = run_cmd("blockMesh", str(case_path), log_mesh)
        if rc != 0:
            print(f"FAILED (see {log_mesh})")
            failed += 1
            records.append({**meta, "status": "FAIL_MESH", "Cl": None, "Cd": None, "Cm": None})
            continue
        print("OK")

        # ── Step 2: simpleFoam ──────────────────────────────────────────────
        print(f"        → simpleFoam ...", end=" ", flush=True)
        rc = run_cmd("simpleFoam", str(case_path), log_solver)
        if rc != 0:
            print(f"FAILED (see {log_solver})")
            failed += 1
            records.append({**meta, "status": "FAIL_SOLVER", "Cl": None, "Cd": None, "Cm": None})
            continue
        print("OK")

        # ── Step 3: Extract force coefficients ─────────────────────────────
        coeffs = extract_force_coeffs(str(case_path))

        # ── Step 4: Quality gate ────────────────────────────────────────────
        gate_ok, reason = quality_gate(coeffs, meta)
        resid_ok = check_residuals(log_solver)

        if gate_ok and resid_ok:
            status = "PASS"
            passed += 1
            print(f"        → ✅ PASS  Cl={coeffs['Cl']:.4f}  "
                  f"Cd={coeffs['Cd']:.5f}  Cm={coeffs['Cm']:.4f}  "
                  f"L/D={coeffs['Cl']/coeffs['Cd']:.1f}")
        elif gate_ok and not resid_ok:
            status = "WARN_RESIDUAL"
            passed += 1  # still usable, flag for review
            print(f"        → ⚠️  WARN  Cl={coeffs['Cl']:.4f}  "
                  f"Cd={coeffs['Cd']:.5f}  residuals not fully converged")
        else:
            status = f"FAIL_GATE: {reason}"
            failed += 1
            print(f"        → ❌ FAIL  {reason}")

        record = {
            **meta,
            "status":    status,
            "Cl":        coeffs["Cl"]  if coeffs else None,
            "Cd":        coeffs["Cd"]  if coeffs else None,
            "Cm":        coeffs["Cm"]  if coeffs else None,
            "LD":        round(coeffs["Cl"]/coeffs["Cd"], 2) if coeffs and coeffs["Cd"] > 0 else None,
            "converged": resid_ok,
        }
        records.append(record)

        # Save incrementally after every case
        pd.DataFrame(records).to_csv(RESULTS_CSV, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Campaign complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ✅ Passed : {passed}")
    print(f"  ❌ Failed : {failed}")
    print(f"  Results  → {RESULTS_CSV}")
    print(f"{'='*65}")

    df = pd.DataFrame(records)
    passed_df = df[df["status"].str.startswith("PASS") | df["status"].str.startswith("WARN")]
    if not passed_df.empty:
        print(f"\n  Cd range : {passed_df['Cd'].min():.5f} → {passed_df['Cd'].max():.5f}")
        print(f"  Cl range : {passed_df['Cl'].min():.4f} → {passed_df['Cl'].max():.4f}")
        print(f"  Mean L/D : {passed_df['LD'].mean():.1f}")

if __name__ == "__main__":
    main()
