"""Script B - compare your running API with XFoil for NACA 2412 at Re = 1,000,000.

Usage (the key is read from the environment and never printed):
    read -rs AERO_KEY; export AERO_KEY
    python B_check_api_vs_xfoil.py https://xxxx.trycloudflare.com
XFoil reference: airfoiltools.com, polar xf-naca2412-il-1000000 (Ncrit 9).
Needs:   only the Python standard library.
"""
import json
import os
import sys
import urllib.error
import urllib.request

# alpha (deg), Cl, Cd, Cm from XFoil
REF = [(0, .2442, .00568, -.0525), (2, .4549, .00581, -.0482),
       (4, .7153, .00711, -.0566), (6, .9016, .00945, -.0497),
       (8, 1.0885, .01270, -.0440), (10, 1.2696, .01591, -.0378),
       (12, 1.4114, .02015, -.0262)]


def predict(base, key, aoa):
    body = json.dumps({"m": 2, "p": 4, "t": 12, "aoa": aoa, "re": 1000000})
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/predict", data=body.encode(), method="POST",
        headers={"X-API-Key": key, "Content-Type": "application/json",
                 "User-Agent": "aeroai-self-check/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main(base):
    key = os.environ.get("AERO_KEY", "")
    if not key:
        sys.exit("Set AERO_KEY first:  read -rs AERO_KEY; export AERO_KEY")
    print("NACA 2412, Re = 1,000,000   (api vs XFoil)")
    print(f"{'AoA':>4} | {'Cl api':>7} {'Cl xf':>7} | {'Cd api':>8} {'Cd xf':>8}"
          f" {'ratio':>6} | {'Cm api':>7} {'Cm xf':>7} | {'L/D api':>7}")
    ratios, cl_errs = [], []
    for aoa, cl, cd, cm in REF:
        try:
            r = predict(base, key, aoa)
        except urllib.error.HTTPError as e:
            sys.exit(f"HTTP {e.code}: 401/403 = key problem, 422 = input rejected.")
        except urllib.error.URLError as e:
            sys.exit(f"Could not reach the API: {e.reason}")
        ratio = r["cd"] / cd
        ratios.append(ratio)
        cl_errs.append(abs(r["cl"] - cl))
        ld = r.get("ld_ratio", r["cl"] / r["cd"])
        print(f"{aoa:>4} | {r['cl']:>7.3f} {cl:>7.3f} | {r['cd']:>8.5f} {cd:>8.5f}"
              f" {ratio:>6.1f} | {r['cm']:>7.4f} {cm:>7.4f} | {ld:>7.1f}")
    mean_ratio = sum(ratios) / len(ratios)
    print(f"\nmean |Cl error|  : {sum(cl_errs) / len(cl_errs):.3f}")
    print(f"mean Cd ratio    : {mean_ratio:.1f}x XFoil")
    if mean_ratio < 1.5:
        print("-> Drag is close to XFoil. The README example is just stale.")
    elif mean_ratio < 3:
        print("-> Drag is above XFoil. RANS is often higher, but note the size.")
    else:
        print("-> Drag is several times XFoil. Treat absolute Cd with caution and"
              " be ready to say so (Part 7, problem 1).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
