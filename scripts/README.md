# AERO.AI — Scripts

These scripts form the data processing and model training pipeline. They are numbered in run order.

> **READ THIS BEFORE RUNNING ANYTHING**  
> Most of these scripts should only ever be run once, or only when intentionally retraining the model. Running them out of order or re-running them accidentally will overwrite validated results and break the model. The API does **not** use these scripts at runtime — they were used to build and train the model.

---

## Run Order

```
00_lock_validation_set.py
        ↓
01_merge_and_clean.py
01_mesh_quality_audit.py      ← run alongside 01_merge_and_clean (independent)
        ↓
02_train_surrogate_multifidelity.py
02b_retrain_stall_margin_bigfoil.py   ← only if retraining stall model
        ↓
03_export_onnx.py
        ↓
04_validate_onnx.py
```

---

## Script Reference

### `00_lock_validation_set.py`

**What it does:** Randomly samples a held-out validation set from the full dataset and locks it by writing `data/processed/validation_lock_manifest.json`. This manifest ensures the same rows are always used for validation, even after the training data is modified.

**When to run:** Once, at the very beginning of the pipeline — before any training. Never run again unless you are intentionally resetting the validation split.

**Output:** `data/processed/validation_lock_manifest.json` (committed to git)

---

### `01_merge_and_clean.py`

**What it does:** Merges raw data from all sources (XFoil polars, BigFoil NACA rows, CFD RANS results) into a single cleaned CSV. Applies outlier filtering and unit normalisation.

**When to run:** Only if raw data has changed. Running this changes the training dataset, which will change model behaviour if you retrain.

**Output:** `data/processed/merged_clean.csv` (gitignored — large file)

**Dependencies:** Raw data files must exist in `data/raw/`. These are gitignored — ask the project lead for access.

---

### `01_mesh_quality_audit.py`

**What it does:** Audits the OpenFOAM CFD mesh quality for all cases. Checks orthogonality, aspect ratio, and skewness. Writes a summary JSON.

**When to run:** After running new CFD cases. Independent of `01_merge_and_clean.py` — can be run in any order alongside it.

**Output:** `data/processed/mesh_quality_summary.json` (committed to git)

---

### `02_train_surrogate_multifidelity.py`

**What it does:** Trains the `MultiFidelityAeroAI` model. This is the main training script. Trains the Cl, Cm, and stall backbone models, then the RANS adapter for Cd. Saves weights and scaler files to `models/`.

**When to run:** Only when intentionally retraining. This is a long-running script (GPU recommended). Running it will overwrite the validated model weights.

**Output:** `models/*.pt`, `models/*.pkl`, `models/*.json` — binaries are gitignored, JSON configs are committed.

> ⚠️ Do not run unless you are certain you want to replace the current trained model.

---

### `02b_retrain_stall_margin_bigfoil.py`

**What it does:** Retrains only the stall margin sub-model using the BigFoil NACA dataset. Used in Session C to improve stall prediction on thick airfoils.

**When to run:** Only if the stall model specifically needs retraining. Replaces `models/stall_margin_model_weights.pt`.

**Output:** `models/stall_margin_model_weights.pt` (gitignored)

> ⚠️ Do not run unless you are specifically targeting stall margin improvement.

---

### `03_export_onnx.py`

**What it does:** Loads the trained PyTorch model and exports it to ONNX format (`aero_ai_v2.0_multifidelity.onnx`). Must be run after training if weights have changed.

**When to run:** After retraining. The ONNX file must be regenerated any time the model architecture or weights change.

**Output:** `models/aero_ai_v2.0_multifidelity.onnx` (~5.4 MB, gitignored)

---

### `04_validate_onnx.py`

**What it does:** Runs 5 validation tests comparing PyTorch vs ONNX output. Checks that the ONNX export is numerically equivalent to the PyTorch model (tolerance: max abs diff < 1e-04). Also runs the physics gate tests.

**When to run:** Always run this after `03_export_onnx.py`. The API should not be used with an unvalidated ONNX file.

**Current result (MVP1):** All 5 ONNX validation tests passed. Max diff = 9.06e-06. Physics gates: 7/8 (Test 8 Cd monotonicity above stall is a known MVP1 limitation).

**Output:** Console report only. No files written.

---

### `bigfoil_parser.py`

**What it does:** Parser utility for the BigFoil NACA aerodynamic database. Reads the raw BigFoil data format and converts it to a pandas DataFrame for use by `01_merge_and_clean.py` and `02b_retrain_stall_margin_bigfoil.py`.

**When to run:** Not run directly — imported as a module by other scripts.

---

### `requirements_bigfoil.txt`

**What it is:** Additional pip dependencies needed specifically for BigFoil data parsing. Install with:

```bash
pip install -r scripts/requirements_bigfoil.txt
```

Only needed if running `02b_retrain_stall_margin_bigfoil.py` or `bigfoil_parser.py`.

---

## Environment

All scripts must be run from within the `aerocfd` conda environment:

```bash
conda activate aerocfd
BASE="/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1"
cd "$BASE"
python scripts/04_validate_onnx.py   # example
```

---

## What You Should NOT Run (Unless Retraining)

| Script | Risk if run accidentally |
|--------|--------------------------|
| `00_lock_validation_set.py` | Resets validation split — invalidates all prior metrics |
| `01_merge_and_clean.py` | Overwrites cleaned training data |
| `02_train_surrogate_multifidelity.py` | Overwrites validated model weights |
| `02b_retrain_stall_margin_bigfoil.py` | Overwrites stall model weights |
| `03_export_onnx.py` | Overwrites ONNX file used by the live API |

Running `04_validate_onnx.py` is always safe — it is read-only.

---

*APEX.AI — AERO.AI MVP1 — CONFIDENTIAL*
