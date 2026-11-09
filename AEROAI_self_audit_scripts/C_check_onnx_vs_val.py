"""Script C - score the DEPLOYED ONNX model on your Cd validation rows.

Usage:   python C_check_onnx_vs_val.py MODELS_DIR path/to/cd_val_locked.csv
Scores Cd twice: with the real shape features stored in the CSV, and with the
approximation the API uses (naca4_to_extra_feats). The gap shows the effect of
training-serving skew. If the first R2 is far below 0.9247, the saved model
differs from the one you scored during training.
Needs:   numpy, pandas, onnxruntime, scikit-learn (to unpickle the scalers).
Only unpickle files that you created yourself.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ONNX = "aero_ai_v2.0_multifidelity.onnx"
X_SCALER = "cl_model_x_scaler.pkl"
EXTRA_SCALER = "rans_adapter_extra_feats_scaler.pkl"
CD_SCALER = "rans_adapter_cd_y_scaler.pkl"
EXTRA_COLS = [f"cst_{i}" for i in range(8)] + ["max_thickness", "le_radius"]


def naca4_to_extra_feats(m, p, t):
    """Copied from api/services/inference.py (m, t in percent; p in tenths)."""
    m, p, t = m / 100.0, p / 10.0, t / 100.0
    cst = [0.2969 * t, -0.1260 * t, -0.3516 * t + 0.5 * m,
           0.2843 * t + m * p, -0.1015 * t, m * (1 - p), m * 0.5, t * 0.1]
    return np.array(cst + [t, 1.1019 * t ** 2], dtype=np.float32)


def parse_p(name):
    s = str(name).upper().replace("NACA", "").strip()
    try:
        return float(s[1])
    except (IndexError, ValueError):
        return 4.0


def build_inputs(df):
    """Columns m, p, thickness, aoa, re - same rules as the training script."""
    m = df["m"] if "m" in df else (df["max_camber"] * 100).clip(0, 9)
    p = df["p"] if "p" in df else df["naca"].map(parse_p)
    t = (df["thickness"] if "thickness" in df
         else (df["max_thickness"] * 100).clip(6, 24))
    return np.column_stack([m, p, t, df["aoa"], df["re"]]).astype(float)


def predict_cd(run, sx, se, scd, x_raw, extra_raw):
    x = sx.transform(x_raw).astype(np.float32)
    e = se.transform(extra_raw).astype(np.float32)
    out = run(x, e)
    return scd.inverse_transform(out[:, 1].reshape(-1, 1)).ravel()


def report(label, y, yhat):
    err = yhat - y
    r2 = 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"\n{label}")
    print(f"  R2 {r2:.4f} | RMSE {np.sqrt(np.mean(err ** 2)):.5f}"
          f" | MAE {np.mean(np.abs(err)):.5f}"
          f" | mean true {y.mean():.4f} | mean predicted {yhat.mean():.4f}")
    for lo, hi in [(0, 0.02), (0.02, 0.06), (0.06, 10)]:
        k = (y >= lo) & (y < hi)
        if k.any():
            print(f"  true Cd {lo:.2f}-{min(hi, 9.99):.2f}: n={k.sum():>5}"
                  f"  MAE {np.mean(np.abs(err[k])):.5f}")
    return r2


def main(models_dir, csv):
    import onnxruntime as ort
    d = Path(models_dir)
    sess = ort.InferenceSession(str(d / ONNX), providers=["CPUExecutionProvider"])
    load = lambda name: pickle.load(open(d / name, "rb"))
    sx, se, scd = load(X_SCALER), load(EXTRA_SCALER), load(CD_SCALER)

    def run(x, e):
        return sess.run(["outputs"], {"x_scaled": x, "extra_feats_scaled": e})[0]

    df = pd.read_csv(csv)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[df["cd"] > 0].reset_index(drop=True)
    x_raw = build_inputs(df)
    y = df["cd"].to_numpy(float)
    e_real = df[EXTRA_COLS].to_numpy(float)
    e_api = np.array([naca4_to_extra_feats(*row[:3]) for row in x_raw])

    print(f"validation rows with Cd > 0: {len(df):,}")
    r_real = report("A) REAL shape features from the CSV",
                    y, predict_cd(run, sx, se, scd, x_raw, e_real))
    r_api = report("B) API APPROXIMATION of the shape features",
                   y, predict_cd(run, sx, se, scd, x_raw, e_api))
    print(f"\nR2 lost to the approximation: {r_real - r_api:.4f}")
    print("Your training log said Cd R2 = 0.9247. Compare it with line A.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
