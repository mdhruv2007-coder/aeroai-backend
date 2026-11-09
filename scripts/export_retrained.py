"""
Build the v2.2 release from the RETRAINED adapter (models_retrain/) - read-only on models/ and models_retrain/.
Run from the project root (after 02_train_surrogate_multifidelity_v2_1.py finished):
    python scripts/export_retrained.py
Writes models_release/: ONNX (+ .onnx.data), .pt, all scalers/configs. Then scores it the way the API will
use it (exact CST features) on the held-out-airfoil validation set.
"""
import importlib.util, json, pickle, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, onnxruntime as ort
from sklearn.metrics import r2_score

BASE = Path.cwd()
M, R, OUT = BASE / "models", BASE / "models_retrain", BASE / "models_release"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE / "api" / "services"))
from cst_features import naca4_to_extra_feats

spec = importlib.util.spec_from_file_location("exp", BASE / "scripts" / "03_export_onnx.py")
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)   # architecture + load_backbone (reads models/)

# 1. rebuild + load the retrained unified weights (strict) ---------------------------------
model = exp.MultiFidelityAeroAI(exp.load_backbone("cl_model"), exp.load_backbone("cm_model"),
                                exp.load_backbone("stall_margin_model"), exp.RANSAdapter())
model.load_state_dict(torch.load(R / "aero_ai_v2.1_multifidelity.pt", map_location="cpu", weights_only=True))
model.eval()
torch.save(model.state_dict(), OUT / "aero_ai_v2.2_multifidelity.pt")

# 2. ONNX export (same call as 03_export_onnx.py) ----------------------------------------------
onnx_path = OUT / "aero_ai_v2.2_multifidelity.onnx"
torch.onnx.export(model, (torch.zeros(1, 5), torch.zeros(1, 10)), str(onnx_path), export_params=True,
                  opset_version=17, do_constant_folding=True, input_names=["x_scaled", "extra_feats_scaled"],
                  output_names=["outputs"],
                  dynamic_axes={"x_scaled": {0: "batch"}, "extra_feats_scaled": {0: "batch"}, "outputs": {0: "batch"}})

# 3. scalers/configs: backbones from models/, adapter files from the retrain -----------------------
for f in list(M.glob("*.pkl")) + list(M.glob("*.json")):
    shutil.copy(f, OUT / f.name)
for n in ("rans_adapter_cd_y_scaler.pkl", "rans_adapter_extra_feats_scaler.pkl", "rans_adapter_config.json"):
    shutil.copy(R / n, OUT / n)
print("built", OUT)

# 4. gate: score exactly like the API on held-out airfoils ---------------------------------------
va = pd.read_csv(BASE / "data/processed/clean_v2/cd_val.csv")
va = va[va.cd > 0].reset_index(drop=True)
s = va.naca.astype(int).astype(str).str.zfill(4)
m, p, t = s.str[0].astype(float), s.str[1].astype(float), s.str[2:].astype(float)
X = np.column_stack([m, p, t, va.aoa, va.re]).astype(np.float32)
E = np.array([naca4_to_extra_feats(a, b, c) for a, b, c in zip(m, p, t)])
ld = lambda n: pickle.load(open(OUT / n, "rb"))
xs = ld("cl_model_x_scaler.pkl").transform(X).astype(np.float32)
es = ld("rans_adapter_extra_feats_scaler.pkl").transform(E).astype(np.float32)
cdsc = ld("rans_adapter_cd_y_scaler.pkl")

sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
o_onnx = sess.run(["outputs"], {"x_scaled": xs, "extra_feats_scaled": es})[0]
with torch.no_grad():
    o_pt = model(torch.tensor(xs), torch.tensor(es)).numpy()
cd = cdsc.inverse_transform(o_onnx[:, 1:2]).ravel()
print(f"\nONNX vs PyTorch on REAL validation inputs: max |diff| = {np.abs(o_onnx - o_pt).max():.2e} (want < 1e-4)")
print(f"Cd R2 on {len(va):,} held-out-airfoil rows, API-style features: {r2_score(va.cd, cd):.4f}")
print(f"mean true Cd {va.cd.mean():.4f} | mean predicted {cd.mean():.4f} | MAE {np.abs(va.cd - cd).mean():.4f}")

print("\nNACA 2412, Re 1M (what Script B will show):")
for a in (0, 2, 4, 6, 8, 10, 12):
    x = np.array([[2, 4, 12, a, 1e6]], dtype=np.float32)
    e = naca4_to_extra_feats(2, 4, 12)[None, :]
    o = sess.run(["outputs"], {"x_scaled": ld("cl_model_x_scaler.pkl").transform(x).astype(np.float32),
                               "extra_feats_scaled": ld("rans_adapter_extra_feats_scaler.pkl").transform(e).astype(np.float32)})[0]
    print(f"  AoA {a:>2}  Cd {cdsc.inverse_transform(o[:, 1:2])[0, 0]:.5f}")
json.dump({"version": "v2.2", "cd_r2_heldout_airfoils": round(float(r2_score(va.cd, cd)), 4),
           "split": "data/processed/clean_v2 (de-duplicated, held-out airfoils)"},
          open(OUT / "RELEASE_NOTES.json", "w"), indent=2)
