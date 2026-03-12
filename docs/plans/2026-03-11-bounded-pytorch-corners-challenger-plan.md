# Bounded PyTorch corners challenger implementation plan

## 1. Overview

Implement a first conservative neural corners challenger as a new `train_corners.py` / `predict_corners.py` path version, without changing the active defaults.

### Recommended approach
- **Path/version:** add a new corners derivation path such as `totals_first_neural_share_residual`.
- **Backbone:** keep the current `totals_first` control exactly as-is for `total` and base `home_share`.
- **Neural head:** add a small PyTorch MLP that predicts a **bounded residual on home-share**, not a raw home-corner delta.
- **Why this is best for v1:**
  - preserves total corners exactly by construction
  - is naturally bounded in share space, so it is harder to produce pathological home/away splits
  - keeps the same raw feature surface as `model_v2/feature_contracts/corners.yaml`
  - improves team-market flexibility while leaving the strong totals-first total backbone intact
  - requires less surface area than a dual-head total+share neural challenger

### Goals / success criteria
- Train and predict through the existing corners family entrypoints using a distinct challenger artifact directory.
- Keep artifact outputs compatible with `holdout_predictions.csv`, `artifact_metadata.json`, `model_config.json`, `features.json`, `imputation.json`, and `dispersion.json`.
- Primary validation gate: improve the `corners_overlap` segment in `src/modeling/v2/eval/live_replacement_compare.py` with **no obvious collapse** in per-market corners rows.
- Keep all existing non-neural corners paths working unchanged.

### Scope boundaries
- **Included:** new neural residual path, artifact save/load, targeted tests, smoke validation commands.
- **Excluded for v1:** new raw features, new market heads, replacing the total model, promotion-policy changes, scheduler wiring, aliasing `model_artifacts/v2/corners`.

## 2. Prerequisites

- **PyTorch is required** for this challenger implementation.
- Current repo state:
  - `AGENTS.md` references `pip install -r requirements.txt`, but no root `requirements.txt` is present.
  - `torch` is present only in `autoresearch/pyproject.toml`.
  - the active repo Python environment currently does **not** have `torch` installed.
- For this v1 challenger, treat Torch as an **environment prerequisite**, not a committed manifest change.
  - Exact install command requiring user permission: `python -m pip install torch`
  - Torch is required for:
    - training the neural residual artifact
    - prediction-time loading/inference **only when** the new neural artifact path is selected
  - Torch is **not** required for existing non-neural corners artifacts; those paths must continue working via lazy import.
- No DB migration or schema change is needed.
- No feature-contract expansion is needed; use the existing selected features from `corners.yaml`.

## 3. Neural artifact contract

The new path must be identifiable and loadable without ambiguity.

### Sidecar filename and format
- Exact sidecar filename: `neural_share_residual_bundle.pt`
- Serialization format: `torch.save()` of a single dictionary containing:
  - `state_dict`
  - `input_mean`
  - `input_std`
  - `input_columns`
  - `hidden_dims`
  - `dropout`
  - `delta_bound`
  - `target_kind` = `home_share_residual`
  - `artifact_format` = `torch_bundle_v1`

### Required `model_config.json` keys for the new path
- `path_version`: `totals_first_neural_share_residual`
- `model_type_selected`: the classical backbone selector already written by the trainer
- `neural_residual_kind`: `home_share_residual`
- `neural_residual_sidecar`: `neural_share_residual_bundle.pt`
- `neural_residual_bound`: numeric residual clip/bound
- `neural_hidden_dims`: JSON array of ints
- `neural_dropout`: numeric
- `neural_artifact_format`: `torch_bundle_v1`
- `direct_total_markets`: `[]`
- `direct_team_markets`: `[]`

### Loader detection rule
- Primary rule: honor `model_config.json[path_version]` when present.
- Fallback rule: if `model_config.json` is missing/incomplete but all of these exist:
  - `total_corners_model.pkl`
  - `home_share_model.pkl`
  - `neural_share_residual_bundle.pt`
  then infer `path_version = totals_first_neural_share_residual`.
- The new path must remain in `TOTALS_FIRST_PATHS` only; it must **not** imply direct market-head paths.

### Backward-compatibility rule
- Existing artifacts without the sidecar must continue to resolve exactly as they do today.
- If the new sidecar is selected but Torch is unavailable, prediction must fail with a clear error naming the artifact path and missing dependency.

## 4. Implementation Steps

### Step 1: Introduce a repo-native neural residual helper
- **Create:** `src/modeling/v2/families/corners/neural_residual.py`
- **Description:** isolate all PyTorch-specific code from the large trainer/predictor files.
- **Key details:**
  - define a tiny MLP (`input -> 64 -> 32 -> 1`) with `ReLU`/`GELU`, tiny dropout or no dropout, and a `tanh` output head
  - standardize inputs with stored train means/stds
  - accept feature matrix plus backbone outputs (`base_total_mu`, `base_home_share`) as inputs; this does not expand the external feature contract
  - train with chronological inner validation and early stopping
      - output a bounded residual `delta_share = bound * tanh(raw)`
      - save/load via the exact sidecar contract defined above
- **Testing:** add unit tests for bounded outputs and save/load round-trip

### Step 2: Extend the corners trainer with a new path version
- **Modify:** `src/modeling/v2/families/corners/train_corners.py`
- **Description:** add the new neural path without disturbing existing path behavior.
- **Key details:**
  - add the new path version to `parse_args()` and `TOTALS_FIRST_PATHS`
  - in `_fit_corner_models()`, continue fitting the existing `total` and `home_share` control models first
  - derive base control predictions on the training frame and train the neural residual on **observed home share minus base home share**, clipped to a safe bound (for example `[-0.25, 0.25]`)
  - at `_predict_corner_rates()`, compute corrected share as `clip(base_home_share + delta_share, 0.05, 0.95)` and then call `reconcile_team_means_from_total_share()`
  - keep total-market probabilities coming from the unchanged totals-first backbone/derived lines
      - write `neural_share_residual_bundle.pt` alongside existing joblib artifacts
      - extend `model_config.json` with the required neural keys defined above
  - keep `artifact_metadata.json` contract unchanged except for `path_version`
- **Testing:** extend existing train utils tests with a synthetic example showing total preservation and bounded share correction

### Step 3: Extend prediction-time loading and inference
- **Modify:** `src/modeling/v2/families/corners/predict_corners.py`
- **Description:** support loading the neural challenger artifacts while keeping old artifacts readable.
- **Key details:**
  - add the new path version to the same path sets
  - in `load_artifacts()`, load standard joblib control models exactly as today, then load the neural sidecar only for the new path
  - use lazy `torch` import so non-neural artifact loads still work without torch
  - if a neural artifact is selected but torch is unavailable, raise a clear, path-specific error
      - add the explicit fallback detection rule based on `neural_share_residual_bundle.pt`
  - reuse the existing `resolve_model_identity()` / metadata flow unchanged
- **Testing:** add a predictor artifact-loading test, mirroring `tests/v2/test_scoreline_predict_utils.py`

### Step 4: Keep evaluation handoff unchanged and validate through existing outputs
- **No code change expected:** `src/modeling/v2/eval/live_replacement_compare.py`
- **Description:** the evaluation handoff already consumes `holdout_predictions.csv` plus artifact metadata, so the neural challenger should slot in if the holdout schema stays unchanged.
- **Key details:**
  - do not change the family replacement schema for v1
  - ensure training still emits the same holdout columns and market naming
  - validate against the existing `corners_overlap` and per-market slices
- **Testing:** reuse existing live replacement tests unless implementation reveals a schema gap

### Step 5: Add focused tests
- **Modify:** `tests/v2/test_corners_train_utils.py`
- **Create:** `tests/v2/test_corners_predict_utils.py`
- **Description:** keep tests narrow and contract-focused.
- **Key details:**
  - synthetic fit/predict test for `totals_first_neural_share_residual`
  - assert `home + away == total` within tolerance
  - assert corrected share stays within configured bounds / clipping range
  - assert derived corners market monotonicity still holds
  - artifact load test for the neural sidecar and `model_config.json` path detection
  - trainer artifact round-trip test that:
    - trains the new path into a temp artifact dir
    - asserts presence of `artifact_metadata.json`, `model_config.json`, `features.json`, `imputation.json`, `dispersion.json`, `holdout_predictions.csv`, and `neural_share_residual_bundle.pt`
    - reloads the artifact through `predict_corners.load_artifacts()`
    - verifies the resolved `path_version` is `totals_first_neural_share_residual`
    - verifies prediction-time load succeeds when Torch is available or raises the intended clear error when Torch is intentionally unavailable
  - optional test for clear failure message when torch sidecar is referenced but unavailable

### Step 6: Smoke rollout and challenger validation
- **Files used, not changed:** `src/modeling/v2/families/corners/train_corners.py`, `src/modeling/v2/eval/live_replacement_compare.py`, `docs/v2_evaluation_workflow.md`
- **Description:** validate the challenger in bounded phases.
- **Key details:**
  - run targeted unit tests first
  - if a small smoke dataset is needed, create a trimmed CSV under `artifacts/tmp/` rather than assuming unsupported trainer flags
  - run targeted pytest files
  - run a full candidate train on the same effective feature surface as current `corners.yaml`
  - run `live_replacement_compare.py` with the challenger corners dir and the current bundled scoreline/anytime references
  - review `corners_overlap` first, then per-market corners rows for obvious regressions

## 5. File Changes Summary

### Create
- `src/modeling/v2/families/corners/neural_residual.py`
- `tests/v2/test_corners_predict_utils.py`

### Modify
- `src/modeling/v2/families/corners/train_corners.py`
- `src/modeling/v2/families/corners/predict_corners.py`
- `tests/v2/test_corners_train_utils.py`

### Delete
- None

## 6. Testing Strategy

- **Unit tests:** neural residual helper round-trip, bounded output, train/predict invariants, artifact loading.
- **Integration-ish tests:** trainer emits valid derived markets and predictor can load a neural artifact bundle.
- **Critical contract test:** trainer artifact round-trip through `predict_corners.load_artifacts()`.
- **Manual/smoke checks:**
  - optional trimmed-dataset train with unique artifact dir under `artifacts/tmp/`
  - targeted pytest run for the corners tests
  - `live_replacement_compare.py` run focused on the corners challenger
- **Primary readout:** `corners_overlap` improves; no broad team-market collapse in the per-market report.

## 7. Rollback Plan

- Because this ships behind a new `path_version`, rollback is low risk:
  - stop using the new path/version
  - delete the challenger artifact directory only if desired
  - revert the helper + path-version code changes if the branch is abandoned
- No data rollback or schema rollback is needed.

## 8. Estimated Effort

- **Effort:** about 0.5-1.5 implementation days, depending on dependency setup friction and first-pass test stability.
- **Complexity:** medium.
- **Main uncertainty:** PyTorch dependency management in the main repo runtime is less clear than the code integration itself.

