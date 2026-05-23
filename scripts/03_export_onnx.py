"""
AERO.AI MVP1 — Script 03: ONNX Exporter (Fixed for MultiFidelityAeroAI)
========================================================================
Exports aero_ai_v2.0_multifidelity.pt as a two-input ONNX graph.

ONNX inputs  (both are pre-scaled — scalers are applied in the API, not here):
  x_scaled           : [N, 5]  — [m, p, thickness, aoa, Re], cl_model's x_scaler
  extra_feats_scaled : [N, 10] — [cst_0..7, max_thickness, le_radius], adapter's scaler

ONNX output:
  outputs : [N, 4] — [Cl_scaled, Cd_scaled, Cm_scaled, stall_margin_scaled]
  NOTE: outputs are in SCALED space. The API unscales them with the saved y_scalers.

Output file:
  models/aero_ai_v2.0_multifidelity.onnx

Then run:
  python 04_validate_onnx.py   ← NEVER skip this

Usage:
  conda activate aerocfd
  cd "/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1/scripts"
  python 03_export_onnx.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path("/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1")
MODEL_DIR = BASE / "models"
SCRIPT_DIR = BASE / "scripts"

MULTIFIDELITY_PT   = MODEL_DIR / "aero_ai_v2.0_multifidelity.pt"
MULTIFIDELITY_ONNX = MODEL_DIR / "aero_ai_v2.0_multifidelity.onnx"

# ── Hyperparameter constants (must match 02_train_surrogate_multifidelity.py) ──
DROPOUT         = 0.25
ADAPTER_DROPOUT = 0.10

# ══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE — exact mirror of 02_train_surrogate_multifidelity.py
#  DO NOT MODIFY — must match the saved state_dict exactly.
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SurrogateNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 256,
                 n_blocks: int = 4, dropout: float = DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
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

    def scale_inputs(self, x_raw: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            self.x_scaler.transform(x_raw).astype(np.float32)
        )

    def unscale_output(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


class RANSAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 256, raw_feat_dim: int = 5,
                 extra_feat_dim: int = 10, dropout: float = ADAPTER_DROPOUT):
        super().__init__()
        in_dim = hidden_dim + raw_feat_dim + extra_feat_dim  # 271

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, hidden_rep: torch.Tensor,
                raw_inputs_scaled: torch.Tensor,
                extra_feats_scaled: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden_rep, raw_inputs_scaled, extra_feats_scaled], dim=1)
        return self.net(x).squeeze(-1)


class MultiFidelityAeroAI(nn.Module):
    def __init__(self,
                 backbone_cl: FrozenBackbone,
                 backbone_cm: FrozenBackbone,
                 backbone_sm: FrozenBackbone,
                 adapter:     RANSAdapter):
        super().__init__()
        self.backbone_cl = backbone_cl
        self.backbone_cm = backbone_cm
        self.backbone_sm = backbone_sm
        self.adapter     = adapter

    def forward(self, x_scaled: torch.Tensor,
                extra_feats_scaled: torch.Tensor) -> torch.Tensor:
        """
        x_scaled           : [N, 5]  — scaled backbone inputs
        extra_feats_scaled : [N, 10] — scaled CST + geometry features
        Returns            : [N, 4] — [Cl_scaled, Cd_scaled, Cm_scaled, stall_margin_scaled]
        """
        hidden_cl, cl_scaled = self.backbone_cl(x_scaled)
        _,         cm_scaled = self.backbone_cm(x_scaled)
        _,         sm_scaled = self.backbone_sm(x_scaled)
        cd_scaled            = self.adapter(hidden_cl, x_scaled, extra_feats_scaled)

        return torch.stack([cl_scaled, cd_scaled, cm_scaled, sm_scaled], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD BACKBONE HELPER
# ══════════════════════════════════════════════════════════════════════════════
def load_backbone(stem: str) -> FrozenBackbone:
    cfg_path = MODEL_DIR / f"{stem}_config.json"
    wt_path  = MODEL_DIR / f"{stem}_weights.pt"
    xs_path  = MODEL_DIR / f"{stem}_x_scaler.pkl"
    ys_path  = MODEL_DIR / f"{stem}_y_scaler.pkl"

    for p in [cfg_path, wt_path, xs_path, ys_path]:
        if not p.exists():
            print(f"  ERROR: Missing: {p}")
            sys.exit(1)

    cfg = json.load(open(cfg_path))
    with open(xs_path, "rb") as f:
        x_scaler = pickle.load(f)
    with open(ys_path, "rb") as f:
        y_scaler = pickle.load(f)

    net = SurrogateNet(
        n_features=cfg["n_features"],
        hidden=cfg["hidden"],
        n_blocks=cfg["n_blocks"],
        dropout=DROPOUT,
    )
    state = torch.load(wt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()

    backbone = FrozenBackbone(net, x_scaler, y_scaler)
    backbone.eval()
    return backbone


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  AERO.AI MVP1 — 03_export_onnx.py")
    print("  Exporting MultiFidelityAeroAI → ONNX")
    print("=" * 60)

    # ── Verify model file exists ──────────────────────────────────────────────
    if not MULTIFIDELITY_PT.exists():
        print(f"  ERROR: {MULTIFIDELITY_PT} not found.")
        print("  Run 02_train_surrogate_multifidelity.py first.")
        sys.exit(1)

    # ── Step 1: Reconstruct the full model from parts ─────────────────────────
    # The .pt file is a state_dict, not a pickled model object.
    # We must rebuild the architecture first, then load weights into it.
    print("\n  Step 1 — Reconstructing architecture from backbone files ...")

    backbone_cl = load_backbone("cl_model")
    backbone_cm = load_backbone("cm_model")
    backbone_sm = load_backbone("stall_margin_model")

    print("  ✓ cl_model backbone loaded and frozen")
    print("  ✓ cm_model backbone loaded and frozen")
    print("  ✓ stall_margin_model backbone loaded and frozen")

    adapter = RANSAdapter(
        hidden_dim=256,
        raw_feat_dim=5,
        extra_feat_dim=10,
        dropout=ADAPTER_DROPOUT,
    )

    model = MultiFidelityAeroAI(backbone_cl, backbone_cm, backbone_sm, adapter)

    # ── Step 2: Load the saved state_dict ─────────────────────────────────────
    print("\n  Step 2 — Loading state_dict from aero_ai_v2.0_multifidelity.pt ...")
    state_dict = torch.load(MULTIFIDELITY_PT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("  ✓ State dict loaded cleanly")

    # ── Step 3: Verify frozen/trainable param counts ──────────────────────────
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Param check — frozen: {n_frozen:,}  trainable: {n_trainable:,}")
    assert n_frozen > 0,    "ERROR: No frozen params — backbone freeze failed"
    assert n_trainable > 0, "ERROR: No trainable params — adapter missing"

    # ── Step 4: Sanity forward pass (PyTorch) ─────────────────────────────────
    print("\n  Step 3 — PyTorch sanity forward pass ...")
    dummy_x     = torch.zeros(1, 5,  dtype=torch.float32)
    dummy_extra = torch.zeros(1, 10, dtype=torch.float32)

    with torch.no_grad():
        pt_out = model(dummy_x, dummy_extra)

    assert pt_out.shape == (1, 4), f"Unexpected output shape: {pt_out.shape}"
    print(f"  ✓ Output shape: {list(pt_out.shape)}  values: {pt_out.numpy().ravel()}")

    # ── Step 5: ONNX export ───────────────────────────────────────────────────
    print(f"\n  Step 4 — Exporting to ONNX (opset 17) ...")
    print(f"  Output → {MULTIFIDELITY_ONNX}")

    torch.onnx.export(
        model,
        (dummy_x, dummy_extra),          # two-input tuple
        str(MULTIFIDELITY_ONNX),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x_scaled", "extra_feats_scaled"],
        output_names=["outputs"],        # [N, 4]: [Cl, Cd, Cm, stall_margin] all scaled
        dynamic_axes={
            "x_scaled":           {0: "batch"},
            "extra_feats_scaled": {0: "batch"},
            "outputs":            {0: "batch"},
        },
    )

    size_kb = MULTIFIDELITY_ONNX.stat().st_size / 1024
    print(f"  ✓ Export complete  ({size_kb:.1f} KB)")

    # ── Step 6: onnxruntime validation ────────────────────────────────────────
    print("\n  Step 5 — onnxruntime validation ...")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(MULTIFIDELITY_ONNX),
            providers=["CPUExecutionProvider"]
        )

        # Check input/output names match what we exported
        input_names  = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]
        print(f"  ONNX inputs  : {input_names}")
        print(f"  ONNX outputs : {output_names}")
        assert input_names  == ["x_scaled", "extra_feats_scaled"], \
            f"Input name mismatch: {input_names}"
        assert output_names == ["outputs"], \
            f"Output name mismatch: {output_names}"

        # Run with same dummy inputs
        ort_out = sess.run(
            None,
            {
                "x_scaled":           dummy_x.numpy(),
                "extra_feats_scaled": dummy_extra.numpy(),
            }
        )[0]

        # Compare PyTorch vs ONNX output
        diff = np.abs(pt_out.numpy() - ort_out).max()
        print(f"  PyTorch output : {pt_out.numpy().ravel()}")
        print(f"  ONNX output    : {ort_out.ravel()}")
        print(f"  Max abs diff   : {diff:.2e}")

        if diff < 1e-4:
            print("  ✓ PyTorch vs ONNX agreement within 1e-4  ✅")
        else:
            print(f"  ⚠️  Max diff {diff:.2e} exceeds 1e-4 — investigate before using ONNX")

    except ImportError:
        print("  ⚠️  onnxruntime not installed — skipping runtime check")
        print("      pip install onnxruntime")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  EXPORT COMPLETE")
    print("=" * 60)
    print(f"  ONNX file  : {MULTIFIDELITY_ONNX.name}  ({size_kb:.1f} KB)")
    print(f"  Inputs     : x_scaled [N,5]  |  extra_feats_scaled [N,10]")
    print(f"  Outputs    : outputs [N,4]   (Cl, Cd, Cm, stall_margin — all scaled)")
    print()
    print("  CRITICAL — outputs are in SCALED space.")
    print("  The API must unscale with saved y_scalers:")
    print("    Cl           → backbone_cl  y_scaler")
    print("    Cd           → rans_adapter_cd_y_scaler")
    print("    Cm           → backbone_cm  y_scaler")
    print("    stall_margin → stall_margin y_scaler + corrected_stall_margin()")
    print()
    print("  NEXT STEP:")
    print("    python 04_validate_onnx.py   ← do not skip")
    print("=" * 60)


if __name__ == "__main__":
    main()
