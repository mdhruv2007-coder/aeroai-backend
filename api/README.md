# AERO.AI API — Setup & Usage

This is the FastAPI backend that serves aerodynamic coefficient predictions from the AERO.AI ONNX model.

---

## Setup

### 1. Activate the environment

```bash
conda activate aerocfd
cd "$BASE/api"   # BASE="/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1"
```

### 2. Install dependencies (first time only)

```bash
pip install -r requirements.txt
```

### 3. Create your `.env` file

```bash
cp .env.example .env
```

Then open `.env` and fill in:

```
# Path to the models directory (no trailing slash)
MODELS_DIR=/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1/models

# API keys — change these before sharing with anyone
API_KEY_ADMIN=your-admin-key-here
API_KEY_FRIEND1=your-friend1-key-here
API_KEY_FRIEND2=your-friend2-key-here
```

> **Never commit `.env` to git.** It is listed in `.gitignore`. Only `.env.example` (the blank template) is in the repo.

### 4. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

If it fails at startup, the most likely cause is a wrong `MODELS_DIR` path in `.env` — the app loads the ONNX model and all scalers on startup.

---

## API Reference

### Authentication

All prediction endpoints require an `X-API-Key` header. Keys are defined in your `.env` file.

```
X-API-Key: your-api-key-here
```

Requests without a valid key return `403 Forbidden`.

---

### `GET /health`

No authentication required. Returns `200 OK` if the server is running.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

---

### `POST /v1/predict`

Predict aerodynamic coefficients for a NACA 4-digit airfoil.

**Request headers:**

```
Content-Type: application/json
X-API-Key: your-api-key-here
```

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `m` | int | ✅ | Max camber (first NACA digit, 0–9) |
| `p` | int | ✅ | Max camber position (second NACA digit, 0–9) |
| `t` | int | ✅ | Thickness (last two NACA digits, 1–40) |
| `aoa` | float | ✅ | Angle of attack in degrees (−15.0 to 25.0) |
| `re` | float | ✅ | Reynolds number (50,000 to 2,000,000) |
| `cst_0` … `cst_7` | float | ❌ | CST shape coefficients (optional geometry refinement) |

**Example request — NACA 2412, AoA = 6°, Re = 1M:**

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"m":2,"p":4,"t":12,"aoa":6,"re":1000000}'
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `cl` | float | Lift coefficient |
| `cd` | float | Drag coefficient |
| `cm` | float | Pitching moment coefficient |
| `stall_margin` | float | Degrees from stall (positive = not stalled) |
| `ld_ratio` | float | Lift-to-drag ratio (cl / cd) |
| `aero_balance` | float | Derived aero balance metric |
| `cl_std` | float | Uncertainty on Cl (MC Dropout std dev) |
| `cd_std` | float | Uncertainty on Cd (MC Dropout std dev) |
| `confidence` | float | Aggregate confidence score (0–1) |
| `confidence_tier` | string | `"green"`, `"amber"`, or `"red"` |
| `model_version` | string | Model version string |
| `inference_ms` | float | Server-side inference time in milliseconds |

**Expected response for the example above:**

```json
{
  "cl": 0.899,
  "cd": 0.075,
  "cm": -0.043,
  "stall_margin": 10.0,
  "ld_ratio": 11.99,
  "aero_balance": 0.31,
  "cl_std": 0.0,
  "cd_std": 0.0,
  "confidence": 1.0,
  "confidence_tier": "green",
  "model_version": "aero_ai_v2.0_multifidelity",
  "inference_ms": 12.4
}
```

> **Note on uncertainty fields:** `cl_std`, `cd_std`, and `confidence` are computed via MC Dropout. In the current ONNX export they return `0.0` / `1.0` because ONNX runs dropout layers deterministically. This is a known MVP1 limitation — treat confidence output as placeholder for now.

---

## File Map

```
api/
├── main.py          ← FastAPI app entry point. Lifespan startup loads model + scalers.
│                      CORS config is here — currently open (*), lock before production.
├── config.py        ← All settings (MODELS_DIR, key names, etc.) via pydantic-settings.
│                      Reads from .env automatically.
├── requirements.txt ← pip dependencies
├── .env.example     ← Safe template — commit this. Never commit .env.
│
├── auth/
│   └── keys.py      ← X-API-Key validation. Keys are loaded from .env via config.py.
│
├── schemas/
│   ├── request.py   ← PredictRequest model — 5 required fields + 10 optional CST fields.
│   └── response.py  ← PredictResponse model — all output fields with types.
│
├── services/
│   ├── inference.py ← Core inference logic. Loads ONNX session, runs MC Dropout (30 passes),
│   │                   calls naca4_to_extra_feats(), handles input scaling + output unscaling.
│   └── derived.py   ← Computes L/D ratio and aero_balance from raw model outputs.
│
├── routers/
│   └── predict.py   ← POST /v1/predict route handler — wires request → inference → response.
│
└── tests/           ← Empty — pytest tests to be added in next sprint.
```

---

## Troubleshooting

**API won't start — `FileNotFoundError` on startup**  
→ Check `MODELS_DIR` in your `.env`. The path must point to the folder containing `.onnx` and `.pkl` files.

**`403 Forbidden` on every request**  
→ Your `X-API-Key` header value doesn't match any key in `.env`. Check for typos or trailing spaces.

**`422 Unprocessable Entity`**  
→ A required field is missing or out of range. Check the request body against the field table above.

**`ImportError: pydantic_core` on startup**  
→ You ran `uvicorn` from the base conda env (Python 3.13). Run `conda activate aerocfd` first.

---

*APEX.AI — AERO.AI MVP1 — CONFIDENTIAL*
