# AERO.AI — audit and fix log (v2.0 → v2.1)

## What was wrong, and the evidence
| # | Finding | Evidence | Fix |
|---|---|---|---|
| 1 | Deployed model's adapter BatchNorm statistics were overwritten after training. MC-dropout test called `adapter.train()` (30 passes) before the unified model was saved. | Script D: `num_batches_tracked` differs by exactly 30 in all 3 BN layers. `verify_root_cause.py`: clean adapter R² 0.9247, deployed 0.2305, deployed weights + clean BN 0.9247. | `models_fixed/aero_ai_v2.1_*` rebuilt with clean BN buffers. Training script now restores BN state and refuses to save if the adapter drifted. |
| 2 | API shape features were a hand-made approximation, not the CST used in training. | Script C: R² −0.10 with approximation. Per-coefficient R² vs the CSV between −9 and 0.07. | `services/cst_features.py` runs the training pipeline's own CST function. Matches all 709 val airfoils to 1e-15. |
| 3 | `naca` stored as float (0012 → 12.0). | 492 of 5,106 val rows (9.6%) have wrong `max_camber`, `max_thickness`, `le_radius`; training parsed `p` wrongly for 453 rows. | Geometry rebuilt with `zfill(4)` in the retrain script. R² impact was small (0.9247 → 0.9219). |
| 4 | Monotonicity penalty never fired (threshold 8.0 on scaled AoA ≈ 43°). | Script E, and the loop in the training script. | Disabled (`MONO_WEIGHT = 0`): applied to raw degrees it would still compare unrelated airfoils. Monotonicity is tested as a property instead. |
| 5 | Uncertainty was fake: ONNX has no dropout, so 30 passes were identical. | `cl_std = 0`, `confidence = 1`. | API returns `null` / `"unavailable"`. |
| 7 | **Validation set was leaked.** The lock split by row; the contamination check compared row indexes, which can never overlap. | Script F: 37.1% of train rows are exact duplicates; 73.3% of val rows also appear in train. R² 0.923 (all val) → 0.860 (rows not in train) → 0.833 (de-duplicated). | `G_make_clean_split.py`: de-duplicate, hold out whole airfoils, re-lock with a documented reason. Retrain on it. |
| 8 | Retrain on the clean split (v2.2): Cd R² **0.887** on 127 held-out airfoils. Physics tests 7/8; Test 8 (Cd monotone above stall) fails for NACA 2418. | `train_v2_1.log` | Shipped as v2.2. Monotonicity is documented as not enforced. |
| 6 | The ONNX validation test checked PyTorch vs ONNX on random noise, not accuracy. | `04_validate_onnx.py`. | `api/tests/test_accuracy.py` scores real validation rows. |

## Honest limits (say these before being asked)
- Honest headline for v2.2: Cd R² 0.887 on 127 held-out airfoils (de-duplicated split). The earlier 0.92 was measured on a leaked split; the v2.1 model scored 0.83 on its de-duplicated rows.
- R² measures agreement with this dataset's Cd labels, not with physics. Median label Cd is 0.064 and the 0–4° band averages 0.0245; XFoil gives about 0.006 for NACA 2412 at 0°. Expect the API to sit well above XFoil (Script B quantifies it).
- Epoch selection used the validation set, and there is no separate test set.
- L/D combines an XFoil-sourced Cl with a CFD-sourced Cd, so absolute L/D is unreliable.
- Cambered training airfoils cover m 1–9 and p 2–7 only; the API warns outside that.
- The 50-case CFD campaign has setups but no results yet, so it is not an independent check.
- Stall margin has not been validated here.

## README — replace "Known limitations" with
Model v2.2: Cd R² 0.887 on held-out airfoils (the original split leaked: 73% of val rows were copies of train rows) (agreement with CFD-labelled drag, not with XFoil or experiment). Deterministic ONNX; no uncertainty estimate (fields are null). Monotonicity in AoA is not enforced. Shape features are computed exactly from m, p, t. Warnings flag inputs outside the training envelope.

## .gitignore — add
models_fixed/*.onnx
models_fixed/*.data
models_fixed/*.pt
models_fixed/*.pkl
models_retrain/
audit_out/
