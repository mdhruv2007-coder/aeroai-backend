"""Accuracy gates on the LOCKED validation CSVs. These would have caught the BatchNorm bug.
Skipped automatically when the (gitignored) data/model files are absent."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from config import settings
from services import inference
from services.cst_features import naca4_to_extra_feats

DATA = settings.model_dir.parent / "data" / "processed"
ONNX = settings.model_dir / settings.onnx_model


@pytest.fixture(scope="module", autouse=True)
def _load():
    if not ONNX.exists():
        pytest.skip(f"model not found: {ONNX}")
    inference.load_model()


def _run(x_raw, extra_raw):
    xs, es = inference._scale_inputs(x_raw, extra_raw)
    return inference._unscale_outputs(inference._forward(xs, es))


def test_cd_r2_on_heldout_airfoils():
    """clean_v2/cd_val.csv: de-duplicated, whole airfoils held out (see G_make_clean_split.py)."""
    f = DATA / "clean_v2" / "cd_val.csv"
    if not f.exists():
        pytest.skip("clean_v2/cd_val.csv not present")
    df = pd.read_csv(f)
    df = df[df.cd > 0].reset_index(drop=True)
    s = df.naca.astype(int).astype(str).str.zfill(4)          # restore the leading zeros lost in the CSV
    m, p, t = (s.str[0].astype(float), s.str[1].astype(float), s.str[2:].astype(float))
    x = np.column_stack([m, p, t, df.aoa, df.re]).astype(np.float32)
    e = np.array([naca4_to_extra_feats(a, b, c) for a, b, c in zip(m, p, t)])
    pred = _run(x, e)["Cd"]
    r2 = r2_score(df.cd, pred)
    assert r2 > 0.85, f"Cd R2 {r2:.4f} on held-out airfoils (v2.2 measured 0.887)"
    assert abs(pred.mean() - df.cd.mean()) < 0.01, "systematic Cd bias"


@pytest.mark.parametrize("col,out,thr", [("cl", "Cl", 0.95), ("cm", "Cm", 0.90)])
def test_cl_cm_r2_on_locked_val(col, out, thr):
    f = DATA / "cl_val_locked.csv"
    if not f.exists():
        pytest.skip("cl_val_locked.csv not present")
    df = pd.read_csv(f)
    df = df[df.data_source == 1].dropna(subset=["m", "p", "thickness", "aoa", "re", col])
    x = df[["m", "p", "thickness", "aoa", "re"]].values.astype(np.float32)
    pred = _run(x, np.zeros((len(x), 10), dtype=np.float32))[out]
    r2 = r2_score(df[col], pred)
    assert r2 > thr, f"{out} R2 {r2:.4f} below {thr} (scaler/backbone mismatch?)"
