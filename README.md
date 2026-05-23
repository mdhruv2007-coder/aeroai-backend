# AERO.AI MVP1 — Backend

**Company:** APEX.AI  
**Project:** AERO.AI — Aerodynamic Coefficient Prediction Engine  
**Status:** MVP Prototype-1 · CONFIDENTIAL

---

## What Is This?

AERO.AI is a machine-learning surrogate model that predicts aerodynamic coefficients for 2D NACA 4-digit airfoils. It was built for Formula Student / FSAE racing teams who need fast aerodynamic estimates without running full CFD simulations.

Given an airfoil geometry and flow conditions, the model predicts:

- **Cl** — Lift coefficient
- **Cd** — Drag coefficient
- **Cm** — Pitching moment coefficient
- **Stall margin** — How far the airfoil is from stalling (degrees)
- **L/D ratio** and **aero balance** (derived outputs)
- **Uncertainty estimates** via MC Dropout (cl_std, cd_std, confidence tier)

The model is a neural network trained on a mix of CFD (OpenFOAM RANS) and XFoil/database data, exported to ONNX format, and served via a FastAPI REST API.

---

## Folder Structure

```
MVP Prototype-1/
├── README.md                          ← You are here
├── .gitignore
│
├── Documents/
│   ├── AERO_AI_MASTER_SESSION_TRANSFER_v7.md   ← Full project context doc
│   └── AERO_AI_Legacy_Prototype2_Reference.docx
│
├── api/                               ← FastAPI application (main deliverable)
│   ├── README.md                      ← API-specific setup & usage guide
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   ├── .env.example                   ← Template — copy to .env and fill in
│   ├── auth/
│   ├── schemas/
│   ├── services/
│   ├── routers/
│   └── tests/                         ← Empty — reserved for future tests
│
├── models/                            ← Config JSONs only (binaries are gitignored)
│   ├── models_meta.json
│   ├── aero_ai_v2.0_multifidelity_config.json
│   └── *.json
│
├── data/
│   ├── processed/                     ← Audit/manifest files only (CSVs gitignored)
│   └── raw/                           ← Fully gitignored (large datasets)
│
├── cfd_campaign/                      ← OpenFOAM simulation setup
│   ├── base_case/
│   ├── lhs_cases.csv
│   └── scripts/
│
└── scripts/                           ← Data processing & model training pipeline
    ├── README.md                      ← Script run order and descriptions
    └── *.py
```

> **Note:** Model binaries (`.pt`, `.pkl`, `.onnx`) are **not** in git. They live locally at `$BASE/models/`. See the API README for how the API loads them.

---

## Prerequisites

You need **WSL2** on Windows with the `aerocfd` conda environment. This project does **not** run from the Windows command prompt or PowerShell — always use the WSL2 terminal.

### 1. Check your environment

```bash
# Activate the conda environment
conda activate aerocfd

# Verify Python is from the right env (should show /mnt/d/conda_envs/aerocfd/...)
which python3
```

If `conda activate aerocfd` fails, the environment has not been set up yet. Ask the project lead for the environment export file.

### 2. Key package versions

| Package | Version |
|---------|---------|
| Python | 3.11.15 |
| torch | 2.7.1+cu118 |
| fastapi | 0.115.12 |
| pydantic | 2.11.5 |
| onnxruntime | 1.24.4 |

> **CRITICAL:** The base conda environment uses Python 3.13, which is incompatible with `pydantic-core`. Always activate `aerocfd` before running anything. Never `pip install` from the base env.

---

## Running the API

Full instructions are in `api/README.md`. Quick version:

```bash
# 1. Set your project root variable (always do this first)
BASE="/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1"

# 2. Activate environment
conda activate aerocfd

# 3. Set up your .env file (first time only)
cd "$BASE/api"
cp .env.example .env
# Then edit .env and fill in your API keys and model path

# 4. Start the API
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

Test it with:

```bash
curl http://localhost:8000/health
```

Expected response: `{"status": "ok"}`

---

## Making Predictions

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"m":2,"p":4,"t":12,"aoa":6,"re":1000000}'
```

This predicts coefficients for a NACA 2412 airfoil at 6° angle of attack and Re = 1,000,000.

See `api/README.md` for full request/response field documentation.

---

## Cloudflare Tunnel (External Access)

The API runs locally. To expose it externally (e.g. for the web dashboard), use a Cloudflare tunnel. In a **second terminal**:

```bash
cloudflared tunnel --url http://localhost:8000
```

This prints a public URL like `https://xxxx.trycloudflare.com`. That URL changes every session — paste the new one into the dashboard each time you start.

> **Note:** `cloudflared` must be installed. If the command is not found, install it: see [Cloudflare Tunnel docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/).

---

## Model Files (Not in Git)

The following binary files are **not** committed to this repo. They must be present locally for the API to start:

| File | What it is |
|------|------------|
| `models/aero_ai_v2.0_multifidelity.onnx` | Main prediction model (~5.4 MB) |
| `models/cl_model_weights.pt` | Cl backbone weights |
| `models/cm_model_weights.pt` | Cm backbone weights |
| `models/stall_margin_model_weights.pt` | Stall margin backbone weights |
| `models/rans_adapter_weights.pt` | Cd adapter weights |
| `models/*.pkl` | Input/output scalers (one per output) |

These are excluded from git because they are large binary files that change infrequently. Store them in a shared drive location and note the path in your `.env` file.

---

## Known Limitations (MVP1)

1. **Cd monotonicity above stall** — one of five high-AoA test cases fails. Accepted MVP1 limitation. Fix requires more high-AoA CFD data (post-MVP1).
2. **MC Dropout uncertainty** — `cl_std` and `cd_std` currently return `0.0` because ONNX runs dropout in deterministic (eval) mode. Confidence scores are not meaningful yet. Under investigation.
3. **CORS is open** — `allow_origins=["*"]` in `main.py`. Will be locked to the dashboard URL before production.
4. **No automated tests** — `api/tests/` folder is empty. Tests will be added in the next sprint.
5. **Re ceiling at 2M** — Reynolds numbers above 2,000,000 are rejected by the API schema. No extrapolation handling.

---

## Quick Reference

```bash
# Always start here
BASE="/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1"
conda activate aerocfd

# Start API
cd "$BASE/api" && uvicorn main:app --host 0.0.0.0 --port 8000

# Start Cloudflare tunnel (second terminal)
cloudflared tunnel --url http://localhost:8000

# Health check
curl http://localhost:8000/health

# Git status
cd "$BASE" && git status

# See what git would commit (dry run)
git add --dry-run .
```

---

*APEX.AI — AERO.AI MVP1 — CONFIDENTIAL*
