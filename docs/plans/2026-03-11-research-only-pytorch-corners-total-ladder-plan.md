# Research-only PyTorch corners totals-ladder implementation plan

## 1. Overview

Implement one additive research-only corners challenger that is structurally different from the existing neural residual and PMF paths, but still fits the current corners artifact/eval contract.

- **Recommended path_version:** `totals_surface_neural_ladder_calibrated`
- **Recommended model_version / artifact dir:** `corners_totals_surface_neural_ladder_v1_candidate_20260311`
- **Recommended sidecar:** `neural_total_market_ladder_bundle.pt`
- **What it is:** keep the existing `totals_first` backbone for `total`, `home_share`, `home`, and `away`, then replace only the direct totals override layer with a PyTorch 4-head ladder aligned to `c75/c85/c95/c105`.
- **Why this shape is the conservative next step:** it is additive, totals-only, evaluation-compatible, and leaves `hc*` / `ac*` on the existing backbone rather than introducing a new team-market surface.

### Goals / success criteria
- Train through `src/modeling/v2/families/corners/train_corners.py` with a new `path_version`.
- Predict through `src/modeling/v2/families/corners/predict_corners.py` with the same artifact contract already used by corners challengers.
- Emit the usual artifact files (`artifact_metadata.json`, `model_config.json`, `features.json`, `imputation.json`, `dispersion.json`, `holdout_predictions.csv`) plus one new Torch sidecar.
- Produce OOF-based calibrated/blended probabilities for `c75/c85/c95/c105`.
- Preserve the current fallback derivation for `hc25/hc35/hc45/hc55/ac25/ac35/ac45/ac55`.

### Scope boundaries
- **Included:** new totals-only Torch ladder path, sidecar save/load, OOF calibration/blend, focused tests.
- **Excluded:** changing defaults/aliases, scheduler wiring, market scope changes, PMF refactors, residual refactors, DB/schema changes.

## 2. Prerequisites

- PyTorch must be available for this new path only; keep lazy import behavior so non-Torch corners artifacts still load.
- No feature-contract expansion is needed; reuse `model_v2/feature_contracts/corners.yaml`.
- No migration or evaluation-flow schema change is needed.
- If Torch is not already installed in the runtime, the exact command to request approval for later is: `python -m pip install torch`.

## 3. Implementation Steps

### Step 1: Add the new path identity and keep it totals-only
- **Modify:** `src/modeling/v2/families/corners/train_corners.py`, `src/modeling/v2/families/corners/predict_corners.py`
- Add `totals_surface_neural_ladder_calibrated` to CLI `--path-version` choices.
- Add a dedicated set such as `NEURAL_TOTAL_LADDER_PATHS = {"totals_surface_neural_ladder_calibrated"}`.
- Include that set in `TOTAL_SURFACE_PATHS` and `TOTALS_FIRST_PATHS`, but **not** in `MARKET_HEAD_PATHS` or `PMF_SURFACE_PATHS`.
- Reuse existing total-surface metadata keys where possible:
  - `direct_total_markets = ["c75", "c85", "c95", "c105"]`
  - `direct_team_markets = []`
  - reuse `total_market_head_blend` / `total_market_head_calibrators` naming for downstream compatibility.
- **Testing note:** add a loader test proving legacy `totals_surface_calibrated` artifacts still resolve unchanged.

### Step 2: Extend the Torch helper module instead of creating a new family contract
- **Modify:** `src/modeling/v2/families/corners/neural_residual.py`
- Keep the existing lazy Torch import, normalization, and sidecar save/load pattern.
- Add ladder-specific constants and helpers:
  - `NEURAL_TOTAL_LADDER_PATH_VERSION`
  - `NEURAL_TOTAL_LADDER_BUNDLE_FILENAME`
  - `NEURAL_TOTAL_LADDER_TARGET_MARKETS = ("c75", "c85", "c95", "c105")`
  - `fit_neural_total_ladder_bundle(...)`
  - `predict_neural_total_ladder_probs(...)`
  - `save_neural_total_ladder_bundle(...)`
  - `load_neural_total_ladder_bundle(...)`
- Recommended bundle contents:
  - `state_dict`, `input_mean`, `input_std`, `input_columns`
  - `hidden_dims`, `dropout`, `target_markets`
  - `artifact_format = "torch_bundle_v1"`
  - optional `calibration_hint = "per_market_posthoc"`
- Input surface should be the current selected features plus additive synthetic columns built inside the helper:
  - `__base_total_mu`
  - `prior_c75`, `prior_c85`, `prior_c95`, `prior_c105`
- **Testing note:** keep direct helper tests optional/skip-if-missing-Torch; do not make the main suite depend on Torch being installed.

### Step 3: Add a totals-ladder training path with OOF calibration and blend
- **Modify:** `src/modeling/v2/families/corners/train_corners.py`
- Add a new trainer helper, parallel to `_fit_total_surface_models()`, e.g. `_fit_neural_total_surface_models(...)`.
- Training flow should be:
  1. Fit the existing `total` and `home_share` backbone models exactly as today.
  2. Chronologically sort by `match_datetime_utc`, `fixture_id`.
  3. For each `_walkforward_ranges(...)` fold:
     - fit a fold `total_model`
     - derive fold prior totals with `_derive_total_market_prior_probs(...)`
     - fit one 4-head Torch ladder on fold-train rows
     - score fold-test rows to get raw probabilities for `c75/c85/c95/c105`
     - store OOF raw probs and OOF prior probs market-by-market
  4. Fit the final full-data ladder bundle on all ordered rows.
  5. For each totals market, reuse the existing calibration/blend contract:
     - `evaluate_market_calibration(...)` on OOF raw predictions
     - `apply_binary_calibrator(...)`
     - `_fit_blend_weight(prior_probs=oof_prior[market], head_probs=calibrated, y_true=...)`
- Store outputs on `models` as something like:
  - `neural_total_ladder`
  - `total_market_head_calibrators`
  - `total_market_head_blend`
  - `total_market_head_calibration_report`
- Write `model_config.json` keys for the new sidecar and Torch metadata, for example:
  - `total_market_head_source = "torch_ladder"`
  - `total_market_head_sidecar = "neural_total_market_ladder_bundle.pt"`
- **Testing note:** assert the new path produces calibrated/blended totals-market metadata but no team-market artifacts.

### Step 4: Load and score the ladder at prediction time while preserving team fallback
- **Modify:** `src/modeling/v2/families/corners/predict_corners.py`
- In `load_artifacts(...)`:
  - honor `model_config.json[path_version]` first
  - add fallback sidecar detection when `model_config.json` is missing and `neural_total_market_ladder_bundle.pt` exists beside `total_corners_model.pkl` and `home_share_model.pkl`
  - load the sidecar only for the new path
  - reuse `total_market_head_calibrators.joblib` and `total_market_head_blend` if present
- In `_predict_corner_rates(...)` for the new path:
  - keep `total`, `home_share`, `home`, and `away` from the existing totals-first backbone
  - derive `prior_total_probs = _derive_total_market_prior_probs(total_mu=..., total_r=...)`
  - score `predict_neural_total_ladder_probs(...)`
  - apply the same per-market calibrator + blend logic already used by `totals_surface_calibrated`
  - finish with `_project_total_head_prob_arrays(...)`
  - set `out["total_market_probs"]`, but do **not** set `team_market_probs`
- This preserves `hc*` / `ac*` fallback because `derive_and_validate_corners(...)` will still derive team markets from `home` and `away` when only totals overrides are supplied.
- **Testing note:** add a targeted test that the new path emits direct totals overrides while team markets still come from the backbone path.

### Step 5: Keep artifact/eval handoff unchanged
- **Likely unchanged:** `src/modeling/v2/eval/live_replacement_compare.py`, `src/modeling/v2/run_evaluation_flow.py`
- Do not change `holdout_predictions.csv` schema or market labels.
- Do not introduce a new family or new evaluation output type.
- The challenger should drop into the current `--corners-dir` evaluation flow as a normal corners artifact.

### Step 6: Add focused contract tests
- **Modify:** `tests/v2/test_corners_train_utils.py`, `tests/v2/test_corners_predict_utils.py`
- Add narrow tests for:
  - path detection for `totals_surface_neural_ladder_calibrated`
  - sidecar loader behavior with and without `model_config.json`
  - clear error surfacing when the new path is selected but Torch load fails
  - `_predict_corner_rates(...)` invariants:
    - `home + away == total`
    - `total_market_probs` contains only `c75/c85/c95/c105`
    - `team_market_probs` is absent for the new path
    - `c75 >= c85 >= c95 >= c105`
  - `main()` artifact roundtrip:
    - writes the usual files plus `neural_total_market_ladder_bundle.pt`
    - reloads through `predict_corners.load_artifacts(...)`
    - resolves the new `path_version`
- Keep most tests Torch-independent by monkeypatching the ladder helper functions, mirroring the existing neural residual tests.

## 4. File Changes Summary

### Modified
- `src/modeling/v2/families/corners/neural_residual.py`
- `src/modeling/v2/families/corners/train_corners.py`
- `src/modeling/v2/families/corners/predict_corners.py`
- `tests/v2/test_corners_train_utils.py`
- `tests/v2/test_corners_predict_utils.py`

### Created
- No new code files required if the existing Torch helper module is extended.

### Deleted
- None.

## 5. Testing Strategy

- **Unit tests:** run only the corners-focused tests first:
  - `pytest -q tests/v2/test_corners_train_utils.py`
  - `pytest -q tests/v2/test_corners_predict_utils.py`
  - `pytest -q tests/v2/test_corners_line_monotonicity.py`
- **Integration-ish check:** train one temp artifact dir for `totals_surface_neural_ladder_calibrated` and reload it through `predict_corners.load_artifacts(...)`.
- **Evaluation smoke check:** point the existing evaluation/live-replacement flow at the challenger artifact via `--corners-dir`; success is unchanged schema plus sensible `corners_overlap`/totals rows.
- **Main benchmark question:** does the new path improve totals-market calibration/Brier without dragging down the inherited team-market fallback?

## 6. Rollback Plan

- Stop using `path_version=totals_surface_neural_ladder_calibrated`.
- Delete the research artifact directory if no longer needed.
- Revert the additive path handling and sidecar loader code; no data rollback or schema rollback is required.

## 7. Estimated Effort

- **Effort:** about 1 focused implementation day, plus benchmarking time.
- **Complexity:** medium.
- **Primary risk:** the Torch ladder may beat raw totals priors but still fail to beat `totals_surface_calibrated` after post-hoc calibration/blending; keeping the change totals-only limits blast radius if that happens.

