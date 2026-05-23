"""
AERO.AI MVP1 — Script 04: ONNX Validation
==========================================
Validates aero_ai_v2.0_multifidelity.onnx against the PyTorch model.

Tests:
  1. Input/output name check
  2. Output shape check  [N, 4]
  3. 50-point agreement  — max abs diff < 1e-4  (numeric noise only)
  4. 500-point agreement — same threshold, random physical inputs
  5. Batch size sweep    — N=1, 8, 32, 64 (dynamic axes check)

Pass condition: ALL 5 tests green.
Do NOT build the API until this script reports ALL PASS.

Usage:
  conda activate aerocfd
  cd "/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1/scripts"
  python 04_validate_onnx.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path("/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1")
MODEL_DIR = BASE / "models"

MULTIFIDELITY_PT   = MODEL_DIR / "aero_ai_v2.0_multifidelity.pt"
MULTIFIDELITY_ONNX = MODEL_DIR / "aero_ai_v2.0_multifidelity.onnx"

PASS_THRESHOLD = 1e-4   # max absolute error across all outputs

DROPOUT         = 0.25
ADAPTER_DROPOUT = 0.10


# ══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE — exact mirror of 02_train_surrogate_multifidelity.py
# ══════════════════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SurrogateNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 256,
                 n_blocks: int = 4, dropout: float = DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


class FrozenBackbone(nn.Module):
    def __init__(self, net: SurrogateNet, x_scaler, y_scaler):
        super().__init__()
        self.net      = net
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.net.requires_grad_(False)

    def forward(self, x_scaled: torch.Tensor):
        hidden = self.net.input_proj(x_scaled)
        hidden = self.net.blocks(hidden)
        output = self.net.head(hidden).squeeze(-1)
        return hidden, output


class RANSAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 256, raw_feat_dim: int = 5,
                 extra_feat_dim: int = 10, dropout: float = ADAPTER_DROPOUT):
        super().__init__()
        in_dim = hidden_dim + raw_feat_dim + extra_feat_dim  # 271
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, hidden_rep, raw_inputs_scaled, extra_feats_scaled):
        x = torch.cat([hidden_rep, raw_inputs_scaled, extra_feats_scaled], dim=1)
        return self.net(x).squeeze(-1)


class MultiFidelityAeroAI(nn.Module):
    def __init__(self, backbone_cl, backbone_cm, backbone_sm, adapter):
        super().__init__()
        self.backbone_cl = backbone_cl
        self.backbone_cm = backbone_cm
        self.backbone_sm = backbone_sm
        self.adapter     = adapter

    def forward(self, x_scaled: torch.Tensor,
                extra_feats_scaled: torch.Tensor) -> torch.Tensor:
        hidden_cl, cl_scaled = self.backbone_cl(x_scaled)
        _,         cm_scaled = self.backbone_cm(x_scaled)
        _,         sm_scaled = self.backbone_sm(x_scaled)
        cd_scaled            = self.adapter(hidden_cl, x_scaled, extra_feats_scaled)
        return torch.stack([cl_scaled, cd_scaled, cm_scaled, sm_scaled], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_backbone(stem: str) -> FrozenBackbone:
    cfg = json.load(open(MODEL_DIR / f"{stem}_config.json"))
    with open(MODEL_DIR / f"{stem}_x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(MODEL_DIR / f"{stem}_y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    net = SurrogateNet(
        n_features=cfg["n_features"],
        hidden=cfg["hidden"],
        n_blocks=cfg["n_blocks"],
        dropout=DROPOUT,
    )
    state = torch.load(MODEL_DIR / f"{stem}_weights.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    backbone = FrozenBackbone(net, x_scaler, y_scaler)
    backbone.eval()
    return backbone


def load_pytorch_model() -> MultiFidelityAeroAI:
    backbone_cl = load_backbone("cl_model")
    backbone_cm = load_backbone("cm_model")
    backbone_sm = load_backbone("stall_margin_model")
    adapter     = RANSAdapter()
    model       = MultiFidelityAeroAI(backbone_cl, backbone_cm, backbone_sm, adapter)
    state       = torch.load(MULTIFIDELITY_PT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def random_inputs(n: int, seed: int = 42):
    """Generate N random physically-plausible scaled inputs.
    Both inputs are zeros here (scaled space) — what matters is
    the numerical agreement between PyTorch and ONNX, not physical realism.
    We add random noise to exercise all neurons."""
    rng = np.random.default_rng(seed)
    x_scaled     = rng.standard_normal((n, 5)).astype(np.float32)
    extra_scaled  = rng.standard_normal((n, 10)).astype(np.float32)
    return x_scaled, extra_scaled


def run_pytorch(model, x_scaled_np, extra_scaled_np):
    with torch.no_grad():
        out = model(
            torch.tensor(x_scaled_np),
            torch.tensor(extra_scaled_np),
        )
    return out.numpy()


def run_onnx(sess, x_scaled_np, extra_scaled_np):
    return sess.run(None, {
        "x_scaled":           x_scaled_np,
        "extra_feats_scaled": extra_scaled_np,
    })[0]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  AERO.AI MVP1 — 04_validate_onnx.py")
    print("  Validating aero_ai_v2.0_multifidelity.onnx")
    print("=" * 60)

    # ── File checks ───────────────────────────────────────────────────────────
    for p in [MULTIFIDELITY_PT, MULTIFIDELITY_ONNX]:
        if not p.exists():
            print(f"  ERROR: {p} not found.")
            sys.exit(1)
    print("  ✓ Both .pt and .onnx files found")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n  Loading PyTorch model ...")
    pt_model = load_pytorch_model()
    print("  ✓ PyTorch model loaded")

    print("  Loading ONNX session ...")
    sess = ort.InferenceSession(str(MULTIFIDELITY_ONNX), providers=["CPUExecutionProvider"])
    print("  ✓ ONNX session loaded")

    results = {}

    # ── Test 1: Input / output names ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Test 1 — Input / output name check")
    input_names  = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    input_shapes  = [i.shape for i in sess.get_inputs()]
    output_shapes = [o.shape for o in sess.get_outputs()]

    print(f"  Inputs  : {input_names}  shapes: {input_shapes}")
    print(f"  Outputs : {output_names}  shapes: {output_shapes}")

    t1 = (input_names == ["x_scaled", "extra_feats_scaled"] and
          output_names == ["outputs"])
    results["test1_names"] = t1
    print(f"  {'✅ PASS' if t1 else '❌ FAIL'} — input/output names correct")

    # ── Test 2: Output shape [N, 4] ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Test 2 — Output shape [N, 4]")
    x_s, e_s = random_inputs(8)
    ort_out   = run_onnx(sess, x_s, e_s)
    t2 = ort_out.shape == (8, 4)
    results["test2_shape"] = t2
    print(f"  ONNX output shape: {ort_out.shape}  (expected (8, 4))")
    print(f"  {'✅ PASS' if t2 else '❌ FAIL'} — output shape correct")

    # ── Test 3: 50-point agreement ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Test 3 — 50-point PyTorch vs ONNX agreement")
    x_s, e_s   = random_inputs(50, seed=0)
    pt_out      = run_pytorch(pt_model, x_s, e_s)
    ort_out     = run_onnx(sess, x_s, e_s)
    abs_err     = np.abs(pt_out - ort_out)
    max_err     = abs_err.max()
    mean_err    = abs_err.mean()
    t3 = max_err < PASS_THRESHOLD
    results["test3_50pt"] = t3
    print(f"  Max  |err| : {max_err:.2e}   (threshold {PASS_THRESHOLD:.0e})")
    print(f"  Mean |err| : {mean_err:.2e}")
    print(f"  {'✅ PASS' if t3 else '❌ FAIL'} — 50-point agreement")

    # Per-output breakdown
    col_names = ["Cl", "Cd", "Cm", "stall_margin"]
    print(f"  Per-output max |err|:")
    for i, name in enumerate(col_names):
        print(f"    {name:<14}: {abs_err[:, i].max():.2e}")

    # ── Test 4: 500-point agreement ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Test 4 — 500-point PyTorch vs ONNX agreement")
    x_s, e_s   = random_inputs(500, seed=99)
    pt_out      = run_pytorch(pt_model, x_s, e_s)
    ort_out     = run_onnx(sess, x_s, e_s)
    abs_err     = np.abs(pt_out - ort_out)
    max_err     = abs_err.max()
    mean_err    = abs_err.mean()
    t4 = max_err < PASS_THRESHOLD
    results["test4_500pt"] = t4
    print(f"  Max  |err| : {max_err:.2e}   (threshold {PASS_THRESHOLD:.0e})")
    print(f"  Mean |err| : {mean_err:.2e}")
    print(f"  {'✅ PASS' if t4 else '❌ FAIL'} — 500-point agreement")

    # Spot-check 5 rows
    print(f"\n  Spot-check (first 5 rows, Cl column):")
    print(f"  {'PyTorch':>12}  {'ONNX':>12}  {'|diff|':>10}")
    for i in range(5):
        print(f"  {pt_out[i,0]:>12.6f}  {ort_out[i,0]:>12.6f}  {abs_err[i,0]:>10.2e}")

    # ── Test 5: Batch size sweep ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Test 5 — Dynamic batch size sweep (N=1, 8, 32, 64)")
    batch_ok = True
    for n in [1, 8, 32, 64]:
        x_s, e_s = random_inputs(n, seed=n)
        pt_out   = run_pytorch(pt_model, x_s, e_s)
        ort_out  = run_onnx(sess, x_s, e_s)
        err      = np.abs(pt_out - ort_out).max()
        ok       = err < PASS_THRESHOLD and ort_out.shape == (n, 4)
        if not ok:
            batch_ok = False
        print(f"  N={n:<4}  shape={str(ort_out.shape):<10}  max|err|={err:.2e}  "
              f"{'✅' if ok else '❌'}")
    results["test5_batch"] = batch_ok
    print(f"  {'✅ PASS' if batch_ok else '❌ FAIL'} — batch size sweep")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    all_pass = all(results.values())
    labels = {
        "test1_names":  "Input/output names",
        "test2_shape":  "Output shape [N,4]",
        "test3_50pt":   "50-point agreement",
        "test4_500pt":  "500-point agreement",
        "test5_batch":  "Batch size sweep",
    }
    for k, label in labels.items():
        icon = "✅" if results[k] else "❌"
        print(f"  {icon}  {label}")

    print()
    if all_pass:
        print("  ✅ ALL 5 TESTS PASSED — ONNX export is valid.")
        print("  → Gate 3 complete. Proceed to building api/ folder.")
        print("  → Next: build api/main.py, api/schemas/, api/services/")
    else:
        n_fail = sum(1 for v in results.values() if not v)
        print(f"  ❌ {n_fail} TEST(S) FAILED — do not build API until all pass.")
        print("  → Re-run 03_export_onnx.py and check for architecture mismatch.")
    print("=" * 60)


if __name__ == "__main__":
    main()
