"""Script F - is the locked validation set really independent of the training set? (read-only)
Run from the project root:   python AEROAI_self_audit_scripts/F_check_leakage.py
Reports exact duplicate rows, train/val overlap, and the deployed v2.1 model's R2 on the
subsets that are NOT leaked. Needs the API running? No - it loads the ONNX file directly."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, onnxruntime as ort
from sklearn.metrics import r2_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "api" / "services"))
from cst_features import naca4_to_extra_feats

MD, P = ROOT / "models_fixed", ROOT / "data" / "processed"
tr = pd.read_csv(P / "cd_train_locked.csv")
va = pd.read_csv(P / "cd_val_locked.csv")
key = ["naca", "aoa", "re", "cd"]
rows = lambda d: [tuple(r) for r in d[key].round(6).values.tolist()]

print(f"train rows {len(tr):,} | val rows {len(va):,}")
print(f"exact duplicate rows inside train: {tr.duplicated(key).sum():,} ({tr.duplicated(key).mean()*100:.1f}%)")
print(f"exact duplicate rows inside val  : {va.duplicated(key).sum():,} ({va.duplicated(key).mean()*100:.1f}%)")
tr_keys = set(rows(tr))
va["in_train"] = [k in tr_keys for k in rows(va)]
va["airfoil_in_train"] = va.naca.isin(set(tr.naca))
print(f"val rows that also appear in train (same airfoil, AoA, Re AND Cd): {va.in_train.sum():,} ({va.in_train.mean()*100:.1f}%)")
print(f"val airfoils never seen in train: {va.loc[~va.airfoil_in_train,'naca'].nunique()} of {va.naca.nunique()} "
      f"({(~va.airfoil_in_train).sum():,} rows)")

va = va[va.cd > 0].reset_index(drop=True)
s = va.naca.astype(int).astype(str).str.zfill(4)
m, p, t = s.str[0].astype(float), s.str[1].astype(float), s.str[2:].astype(float)
X = np.column_stack([m, p, t, va.aoa, va.re]).astype(np.float32)
E = np.array([naca4_to_extra_feats(a, b, c) for a, b, c in zip(m, p, t)])
ld = lambda n: pickle.load(open(MD / n, "rb"))
sess = ort.InferenceSession(str(MD / "aero_ai_v2.1_multifidelity.onnx"), providers=["CPUExecutionProvider"])
out = sess.run(["outputs"], {"x_scaled": ld("cl_model_x_scaler.pkl").transform(X).astype(np.float32),
                             "extra_feats_scaled": ld("rans_adapter_extra_feats_scaler.pkl").transform(E).astype(np.float32)})[0]
va["pred"] = ld("rans_adapter_cd_y_scaler.pkl").inverse_transform(out[:, 1:2]).ravel()

def line(name, d):
    if len(d) < 30: print(f"  {name:<46} n={len(d):>5}  (too few)"); return
    print(f"  {name:<46} n={len(d):>5}  R2 {r2_score(d.cd, d.pred):.4f}  MAE {np.abs(d.cd - d.pred).mean():.4f}")

print("\ndeployed v2.1 model on the locked val set:")
line("all val rows", va)
line("val rows NOT present in train", va[~va.in_train])
line("... and de-duplicated", va[~va.in_train].drop_duplicates(key))
line("val rows of airfoils never seen in train", va[~va.airfoil_in_train])
