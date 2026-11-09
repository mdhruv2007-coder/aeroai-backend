"""
READ-ONLY verification of the two suspected root causes. Writes nothing.
Run from the project root:
  python verify_root_cause.py
It loads the clean adapter, the adapter inside the unified .pt, and a hybrid
(unified weights + clean BatchNorm buffers), and scores each on cd_val_locked.csv
with (a) the training script's input parsing and (b) correctly parsed NACA digits.
"""
import importlib.util, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import r2_score

BASE = Path("/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1")
M = BASE / "models"
spec = importlib.util.spec_from_file_location("exp", BASE / "scripts" / "03_export_onnx.py")
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)  # reuse exact architecture

df = pd.read_csv(BASE / "data/processed/cd_val_locked.csv")
df = df[df.cd > 0].reset_index(drop=True)

def parse_p_train(v):                      # copy of the training script's logic
    d = str(v).upper().replace("NACA", "").strip()
    return float(d[1]) if len(d) >= 4 else 4.0

def build_x(df, mode):
    if mode == "train":                    # what 02_train... actually fed the backbone
        m = (df.max_camber * 100).clip(0, 9); p = df.naca.apply(parse_p_train)
        t = (df.max_thickness * 100).clip(6, 24)
    else:                                  # NACA digits with the leading zeros restored
        s = df.naca.astype(int).astype(str).str.zfill(4)
        m, p, t = s.str[0].astype(float), s.str[1].astype(float), s.str[2:].astype(float)
    return np.column_stack([m, p, t, df.aoa, df.re]).astype(np.float32)

cols = [f"cst_{i}" for i in range(8)] + ["max_thickness", "le_radius"]
E_raw = df[cols].values.astype(np.float32)
ef_sc = pickle.load(open(M / "rans_adapter_extra_feats_scaler.pkl", "rb"))
cd_sc = pickle.load(open(M / "rans_adapter_cd_y_scaler.pkl", "rb"))
E_s = torch.tensor(ef_sc.transform(E_raw).astype(np.float32))
bb = exp.load_backbone("cl_model")

clean = torch.load(M / "rans_adapter_weights.pt", map_location="cpu", weights_only=True)
uni = torch.load(M / "aero_ai_v2.0_multifidelity.pt", map_location="cpu", weights_only=True)
uni_ad = {k[len("adapter."):]: v for k, v in uni.items() if k.startswith("adapter.")}
isbn = lambda k: k.endswith(("running_mean", "running_var", "num_batches_tracked"))
hybrid = {k: (clean[k] if isbn(k) else v) for k, v in uni_ad.items()}

def score(state, x_raw):
    ad = exp.RANSAdapter(); ad.load_state_dict(state); ad.eval()
    xs = bb.scale_inputs(x_raw)
    with torch.no_grad():
        h, _ = bb(xs)
        out = ad(h, xs, E_s).numpy()
    pred = cd_sc.inverse_transform(out.reshape(-1, 1)).ravel()
    ok = df.naca.values >= 1000
    return r2_score(df.cd, pred), r2_score(df.cd[ok], pred[ok]), pred.mean()

print(f"rows {len(df)} | mean true Cd {df.cd.mean():.4f} | rows with naca<1000: {(df.naca<1000).sum()}")
print(f"{'adapter weights':<34}{'inputs':<8}{'R2 all':>8}{'R2 naca>=1000':>15}{'mean pred':>11}")
for name, st in [("clean rans_adapter_weights.pt", clean),
                 ("unified .pt (as deployed)", uni_ad),
                 ("unified weights + clean BN", hybrid)]:
    for mode in ("train", "true"):
        a, b, mp = score(st, build_x(df, mode))
        print(f"{name:<34}{mode:<8}{a:>8.4f}{b:>15.4f}{mp:>11.4f}")
print("\nRead it as: clean ~0.92 and unified ~0.23 -> BatchNorm corruption in the saved unified file.")
print("hybrid ~ clean -> BN buffers alone explain the gap. 'true' vs 'train' rows show the NACA-parsing damage.")
