"""
AERO.AI MVP1 — 02b_retrain_stall_margin_bigfoil.py
====================================================
Task 1 (Session B → C): Retrain stall_margin_model using BigFoil data.

Rules:
  - NACA 4-digit airfoils only — non-NACA rows skipped
  - Extract m, p, thickness from airfoil_name string
  - Target column: stall_margin_deg
  - Architecture: SurrogateNet(hidden=128, n_blocks=3) — same as current model
  - Overwrites models/stall_margin_model_weights.pt
  - Backup already exists in models_backup_consider/

Usage:
  conda activate aerocfd
  cd "/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1/scripts"
  python 02b_retrain_stall_margin_bigfoil.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE      = Path("/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1")
PROC      = BASE / "data" / "processed"
MODEL_DIR = BASE / "models"

BIGFOIL_SM = PROC / "bigfoil_stall_margins.csv"

OUT_WEIGHTS = MODEL_DIR / "stall_margin_model_weights.pt"
OUT_CFG     = MODEL_DIR / "stall_margin_model_config.json"
OUT_XSCALER = MODEL_DIR / "stall_margin_model_x_scaler.pkl"
OUT_YSCALER = MODEL_DIR / "stall_margin_model_y_scaler.pkl"

# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS (match 02_train_surrogate.py stall_margin settings)
# ══════════════════════════════════════════════════════════════════════════════
SEED       = 42
BATCH_SIZE = 64
MAX_EPOCHS = 800
PATIENCE   = 100
LR_INIT    = 1e-3
LR_MIN     = 1e-5
T0         = 50
T_MULT     = 2
DROPOUT    = 0.25
HIDDEN     = 128
N_BLOCKS   = 3

FEAT = ["m", "p", "thickness", "aoa", "Re"]   # must match backbone FEAT_CL

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE (mirror of 02_train_surrogate.py — DO NOT CHANGE)
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
    def __init__(self, n_features: int, hidden: int = HIDDEN,
                 n_blocks: int = N_BLOCKS, dropout: float = DROPOUT):
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


# ══════════════════════════════════════════════════════════════════════════════
#  NACA 4-DIGIT PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_naca4(name: str):
    """
    Parse NACA 4-digit airfoil name → (m, p, thickness).
    Returns None if not a valid NACA 4-digit profile.

    Examples:
      'NACA2412' → (2.0, 4.0, 12.0)
      'NACA0012' → (0.0, 0.0, 12.0)
      'naca4415' → (4.0, 4.0, 15.0)
      'Clark Y'  → None  (skipped)
    """
    s = str(name).strip().upper()
    if not s.startswith("NACA"):
        return None
    digits = s[4:].strip()
    if len(digits) != 4 or not digits.isdigit():
        return None
    m         = float(digits[0])        # max camber in %
    p         = float(digits[1])        # camber position (tenths of chord)
    thickness = float(digits[2:4])      # thickness in %
    return m, p, thickness


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD + PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print(f"  Loading {BIGFOIL_SM} ...")
    df = pd.read_csv(BIGFOIL_SM)
    print(f"  Raw rows: {len(df)}")

    # Rename columns to match FEAT names
    # bigfoil cols: airfoil_name, Re, alpha_deg, stall_margin_deg
    df = df.rename(columns={"alpha_deg": "aoa"})

    # Drop rows with missing targets
    df = df.dropna(subset=["stall_margin_deg", "aoa", "Re"])
    df = df[df["stall_margin_deg"] >= 0].reset_index(drop=True)

    # Parse NACA 4-digit geometry
    parsed = df["airfoil_name"].apply(parse_naca4)
    naca_mask = parsed.notna()
    n_skipped = (~naca_mask).sum()
    print(f"  Skipping {n_skipped} non-NACA rows  |  Keeping {naca_mask.sum()} NACA rows")

    df = df[naca_mask].copy().reset_index(drop=True)
    parsed_naca = parsed[naca_mask].reset_index(drop=True)

    df["m"]         = parsed_naca.apply(lambda x: x[0])
    df["p"]         = parsed_naca.apply(lambda x: x[1])
    df["thickness"] = parsed_naca.apply(lambda x: x[2])

    # Clip to training domain
    df["m"]         = df["m"].clip(0, 9)
    df["p"]         = df["p"].clip(0, 9)
    df["thickness"] = df["thickness"].clip(6, 24)
    df["Re"]        = df["Re"].clip(5e4, 5e6)

    # Drop any remaining NaNs in features
    df = df.dropna(subset=FEAT + ["stall_margin_deg"]).reset_index(drop=True)

    print(f"  Final rows after cleaning: {len(df)}")
    print(f"  stall_margin_deg  — mean={df.stall_margin_deg.mean():.2f}°  "
          f"std={df.stall_margin_deg.std():.2f}°  "
          f"min={df.stall_margin_deg.min():.2f}°  max={df.stall_margin_deg.max():.2f}°")

    X = df[FEAT].values.astype(np.float32)
    y = df["stall_margin_deg"].values.astype(np.float32)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train(X: np.ndarray, y: np.ndarray):
    # 80/20 split (stratify not needed for regression)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    print(f"  Train: {len(X_tr)}  |  Val: {len(X_val)}")

    # Scale
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_tr_s  = x_scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = x_scaler.transform(X_val).astype(np.float32)
    y_tr_s  = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)

    # DataLoaders
    tr_ds  = TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_tr_s))
    val_ds = TensorDataset(torch.tensor(X_val_s), torch.tensor(y_val_s))
    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 4, shuffle=False)

    # Model
    net = SurrogateNet(n_features=len(FEAT), hidden=HIDDEN, n_blocks=N_BLOCKS,
                       dropout=DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    optimiser = torch.optim.AdamW(net.parameters(), lr=LR_INIT, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=T0, T_mult=T_MULT, eta_min=LR_MIN
    )
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val   = float("inf")
    best_state = None
    wait       = 0

    print(f"\n  Training for up to {MAX_EPOCHS} epochs (patience={PATIENCE}) ...")
    print(f"  Device: {DEVICE}")
    print()

    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss_fn(net(xb), yb).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
        scheduler.step()

        net.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(loss_fn(net(xb.to(DEVICE)), yb.to(DEVICE)).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 50 == 0:
            print(f"  Epoch {epoch:>4}/{MAX_EPOCHS}  "
                  f"val_loss={val_loss:.6f}  best={best_val:.6f}  "
                  f"lr={optimiser.param_groups[0]['lr']:.2e}")

        if wait >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # Evaluate
    net.load_state_dict(best_state)
    net.eval()
    preds_s = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds_s.append(net(xb.to(DEVICE)).cpu().numpy())
    preds_scaled = np.concatenate(preds_s)
    preds_sm = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    r2 = r2_score(y_val, preds_sm)
    mae = np.mean(np.abs(preds_sm - y_val))
    print(f"\n  Val R²  : {r2:.4f}")
    print(f"  Val MAE : {mae:.3f}°")

    return net, x_scaler, y_scaler, r2, best_state


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════════════
def save(best_state, x_scaler, y_scaler, r2: float):
    torch.save(best_state, OUT_WEIGHTS)
    with open(OUT_XSCALER, "wb") as f:
        pickle.dump(x_scaler, f)
    with open(OUT_YSCALER, "wb") as f:
        pickle.dump(y_scaler, f)

    cfg = {
        "n_features":  len(FEAT),
        "hidden":      HIDDEN,
        "n_blocks":    N_BLOCKS,
        "dropout":     DROPOUT,
        "feat_order":  FEAT,
        "target":      "stall_margin_deg",
        "r2_val":      round(r2, 6),
        "data_source": "BigFoil NACA-only rows from bigfoil_stall_margins.csv",
        "note":        "Retrained Session C Task 1 — overwrites previous XFOIL-only weights",
    }
    with open(OUT_CFG, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n  Saved → {OUT_WEIGHTS.name}")
    print(f"  Saved → {OUT_CFG.name}")
    print(f"  Saved → {OUT_XSCALER.name}")
    print(f"  Saved → {OUT_YSCALER.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  AERO.AI — 02b_retrain_stall_margin_bigfoil.py")
    print("  Task 1: Retrain stall_margin_model on BigFoil data")
    print("=" * 60)
    print(f"  Device : {DEVICE}  |  Seed : {SEED}")
    print(f"  Arch   : SurrogateNet(hidden={HIDDEN}, n_blocks={N_BLOCKS})")
    print()

    if not BIGFOIL_SM.exists():
        print(f"  ERROR: {BIGFOIL_SM} not found.")
        sys.exit(1)

    X, y = load_data()

    print("\n" + "─" * 60)
    print("  TRAINING")
    print("─" * 60)
    net, x_scaler, y_scaler, r2, best_state = train(X, y)

    print("\n" + "─" * 60)
    print("  SAVING")
    print("─" * 60)
    save(best_state, x_scaler, y_scaler, r2)

    print()
    print("=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  stall_margin_model R² : {r2:.4f}")
    if r2 >= 0.85:
        print("  ✅ R² looks good — run 02_train_surrogate_multifidelity.py next")
    else:
        print("  ⚠️  R² below 0.85 — check data quality or increase MAX_EPOCHS")
    print()
    print("  NEXT STEP:")
    print("    python 02_train_surrogate_multifidelity.py")


if __name__ == "__main__":
    main()
