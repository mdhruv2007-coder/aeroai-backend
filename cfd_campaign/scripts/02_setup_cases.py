"""
02_setup_cases.py
AERO.AI MVP — CFD Campaign
Reads lhs_cases.csv, clones base_case for each of the 50 runs,
injects AoA / Re / NACA geometry into each case directory,
and generates the 2D airfoil mesh using blockMesh.

Save to:
  /mnt/d/AEROAI/MVP1/cfd_campaign/scripts/
Run with:
  source /opt/openfoam10/etc/bashrc && python3 02_setup_cases.py
"""

import os
import shutil
import numpy as np
import pandas as pd
import subprocess
import math

BASE      = "/mnt/d/AEROAI/MVP1/cfd_campaign"
BASE_CASE = os.path.join(BASE, "base_case")
CASES_DIR = os.path.join(BASE, "cases")
LHS_CSV   = os.path.join(BASE, "lhs_cases.csv")

NU_AIR    = 1.5e-5   # kinematic viscosity of air at 15°C [m^2/s]
CHORD     = 1.0      # unit chord [m]
RHO       = 1.225    # air density [kg/m^3]

# ── NACA 4-digit profile generator ────────────────────────────────────────────
def naca4_profile(m, p, t, n_points=100):
    """
    Returns (x, y_upper, y_lower) for a NACA 4-digit airfoil.
    m = max camber (fraction), p = camber position, t = thickness fraction
    """
    x = np.linspace(0, 1, n_points)

    # Thickness distribution
    yt = (t / 0.2) * (0.2969 * np.sqrt(x)
                      - 0.1260 * x
                      - 0.3516 * x**2
                      + 0.2843 * x**3
                      - 0.1015 * x**4)

    # Camber line
    yc = np.where(x < p,
                  (m / p**2) * (2*p*x - x**2),
                  (m / (1-p)**2) * ((1 - 2*p) + 2*p*x - x**2))

    dyc_dx = np.where(x < p,
                      (2*m / p**2) * (p - x),
                      (2*m / (1-p)**2) * (p - x))

    theta = np.arctan(dyc_dx)

    xu = x  - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl

def write_airfoil_stl(path, xu, yu, xl, yl):
    """Write a simple ASCII STL for the airfoil (2D extruded in z ±0.05)"""
    z0, z1 = -0.05, 0.05
    n = len(xu)

    def tri(v1, v2, v3):
        nx, ny, nz = 0, 0, 1  # dummy normal
        return (f"  facet normal {nx} {ny} {nz}\n"
                f"    outer loop\n"
                f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n"
                f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n"
                f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n"
                f"    endloop\n"
                f"  endfacet\n")

    # Combine upper + lower surface into closed loop
    # upper: TE→LE (xu,yu), lower: LE→TE (xl,yl)
    pts = ([(xu[i], yu[i]) for i in range(n)] +
           [(xl[i], yl[i]) for i in range(n-1, -1, -1)])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("solid airfoil\n")
        for i in range(len(pts) - 1):
            x0a, y0a = pts[i]
            x1a, y1a = pts[i+1]
            # two triangles per quad panel (front + back face)
            f.write(tri((x0a,y0a,z0),(x1a,y1a,z0),(x1a,y1a,z1)))
            f.write(tri((x0a,y0a,z0),(x1a,y1a,z1),(x0a,y0a,z1)))
        f.write("endsolid airfoil\n")

def rotate_velocity(U_mag, aoa_deg):
    """Return (Ux, Uy) for given freestream speed and angle of attack."""
    aoa = math.radians(aoa_deg)
    return U_mag * math.cos(aoa), U_mag * math.sin(aoa)

def compute_turbulence(U_mag, Re, chord=1.0, TI=0.001, nu=NU_AIR):
    """
    Compute k and omega for inlet BCs.
    TI  = turbulence intensity (0.1% = 0.001 for clean wind tunnel)
    L   = turbulent length scale ~ 0.07 * chord
    """
    k     = 1.5 * (U_mag * TI)**2
    L     = 0.07 * chord
    Cmu   = 0.09
    omega = math.sqrt(k) / (Cmu**0.25 * L)
    return k, omega

def patch_file(path, replacements):
    """In-place string replacement in an OpenFOAM dict file."""
    with open(path, "r") as f:
        content = f.read()
    for old, new in replacements.items():
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)

# ── Main setup loop ────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(LHS_CSV, index_col="case_id")
    os.makedirs(CASES_DIR, exist_ok=True)

    print(f"Setting up {len(df)} cases in: {CASES_DIR}\n")

    for case_id, row in df.iterrows():
        aoa   = row["AoA"]
        Re    = int(row["Re"])
        m     = row["m"]
        p     = row["p"]
        t_c   = row["t_c"]
        naca  = row["naca4"]

        case_name = f"case_{case_id:03d}_NACA{naca}_AoA{aoa:.1f}_Re{Re}"
        case_path = os.path.join(CASES_DIR, case_name)

        # Skip if already set up
        if os.path.exists(case_path):
            print(f"  [SKIP] {case_name} already exists")
            continue

        # Clone base case
        shutil.copytree(BASE_CASE, case_path)

        # ── Derived quantities ──────────────────────────────────────────────
        nu    = NU_AIR
        U_mag = Re * nu / CHORD        # U = Re * nu / c  (unit chord)
        Ux, Uy = rotate_velocity(U_mag, aoa)
        k, omega = compute_turbulence(U_mag, Re)

        # ── Patch 0/U ───────────────────────────────────────────────────────
        patch_file(f"{case_path}/0/U", {
            "uniform (1 0 0);   // placeholder":
                f"uniform ({Ux:.6f} {Uy:.6f} 0);",
            "freestreamValue uniform (1 0 0); }":
                f"freestreamValue uniform ({Ux:.6f} {Uy:.6f} 0); }}",
        })

        # ── Patch 0/k ───────────────────────────────────────────────────────
        patch_file(f"{case_path}/0/k", {
            "uniform 1.5e-5;    // placeholder: 0.1% TI at U=1":
                f"uniform {k:.6e};",
            "freestreamValue uniform 1.5e-5;":
                f"freestreamValue uniform {k:.6e};",
            "value uniform 1.5e-5;":
                f"value uniform {k:.6e};",
        })

        # ── Patch 0/omega ───────────────────────────────────────────────────
        patch_file(f"{case_path}/0/omega", {
            "uniform 1.0;       // placeholder":
                f"uniform {omega:.6e};",
            "freestreamValue uniform 1.0;":
                f"freestreamValue uniform {omega:.6e};",
            "value uniform 1.0;":
                f"value uniform {omega:.6e};",
        })

        # ── Patch constant/transportProperties (nu = nu_air) ───────────────
        patch_file(f"{case_path}/constant/transportProperties", {
            "nu              1.5e-05;   // placeholder — overridden per-case by run script":
                f"nu              {nu:.6e};   // Re={Re}, U={U_mag:.4f} m/s",
        })

        # ── Patch system/controlDict (magUInf for force coefficients) ───────
        patch_file(f"{case_path}/system/controlDict", {
            "magUInf         1.0;       // placeholder — overridden per case":
                f"magUInf         {U_mag:.6f};",
        })

        # ── Write case metadata ─────────────────────────────────────────────
        meta = {
            "case_id": case_id, "naca4": naca, "AoA": aoa, "Re": Re,
            "m": m, "p": p, "t_c": t_c,
            "U_mag": round(U_mag, 4), "Ux": round(Ux, 6), "Uy": round(Uy, 6),
            "k": round(k, 8), "omega": round(omega, 4), "nu": nu,
        }
        import json
        with open(f"{case_path}/case_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # ── Generate NACA airfoil profile and write STL ─────────────────────
        xu, yu, xl, yl = naca4_profile(m, p, t_c)
        stl_path = f"{case_path}/constant/triSurface/airfoil.stl"
        write_airfoil_stl(stl_path, xu, yu, xl, yl)

        print(f"  [OK]   case_{case_id:03d}  NACA{naca}  AoA={aoa:.1f}°  "
              f"Re={Re:,}  U={U_mag:.2f} m/s  k={k:.2e}  ω={omega:.2f}")

    print(f"\n✅ All cases set up in: {CASES_DIR}")
    print("Next: run 03_run_campaign.py to execute all 50 CFD simulations")

if __name__ == "__main__":
    main()
