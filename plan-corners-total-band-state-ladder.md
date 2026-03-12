## 1. Overview

### Recommended challenger
- **Concept:** threshold-aligned total-state mixture with conditional team heads.
- **Suggested `path_version`:** `total_band_state_ladder`
- **Suggested artifact/model version example:** `corners_total_band_state_ladder_v1_candidate_20260311`

### What it is
Keep the existing totals-first backbone (`total_corners_model.pkl` + `home_share_model.pkl`) for stable means/priors, but replace the current PMF/market-head surface logic with a new two-stage market generator:
1. predict a **5-state total-corners band distribution** aligned to the live totals lines;
2. predict **team-market probabilities conditional on those total bands**;
3. marginalize back to the existing `c*`, `hc*`, and `ac*` markets.

### Why this is truly different
- Unlike `pmf_surface_blended`, it does **not** assume independent home/away count PMFs.
- Unlike the neural residual paths, it does **not** make smooth residual corrections to the same totals/share backbone.
- Unlike current direct team heads, it does **not** treat team-corners markets as unconditional; it explicitly conditions them on the predicted total-corners regime.
- It is structurally closest to the anytime family’s `state_ladder`: predict a small set of meaningful states first, then derive downstream markets from a conditional ladder.

### Scope boundaries
- Included: new corners path inside the existing v2 family trainer/predictor/eval contract.
- Excluded: new PIT feature extraction, DB schema changes, scheduler wiring, new external packages.
- Success criteria: clean artifact load/predict flow, coherent monotone market outputs, targeted tests, and a challenger that is meaningfully different enough to justify one more corners cycle.

## 2. Prerequisites

- No new dependencies; reuse `sklearn`, `pandas`, `numpy`, `joblib`, and existing calibration helpers already used in `src/modeling/v2/families/corners/train_corners.py`.
- No migrations or data writes beyond normal artifact outputs under `model_artifacts/v2/...`.
- Continue using the current corners feature contract at `model_v2/feature_contracts/corners.yaml` unless a later bounded rerun intentionally swaps contracts.
- Keep current evaluation flow unchanged: `src/modeling/v2/run_evaluation_flow.py` should consume the new family artifact directory the same way as existing corners challengers.

## 3. Implementation Steps

### Step 1: Add new path plumbing
- **Files:**
  - `src/modeling/v2/families/corners/train_corners.py`
  - `src/modeling/v2/families/corners/predict_corners.py`
- **Do:** add `total_band_state_ladder` to path-version choices and path sets.
- **Key details:** treat it as a totals-first path so it keeps the current backbone/load flow.
- **Testing:** extend artifact-loading tests so `load_artifacts()` recognizes the new sidecars and `model_config.json` fields.

### Step 2: Define the modeled quantities and targets
- **Files:** `src/modeling/v2/families/corners/train_corners.py`
- **Do:** add helpers for the 5 disjoint total bands:
  - `tb_le_7` = `total_corners <= 7`
  - `tb_eq_8` = `total_corners == 8`
  - `tb_eq_9` = `total_corners == 9`
  - `tb_eq_10` = `total_corners == 10`
  - `tb_ge_11` = `total_corners >= 11`
- **Exact trained quantities:**
  1. backbone `E[total_corners | x]`
  2. backbone `E[home_share | x]`
  3. `P(total_band = b | x)` for the 5 bands
  4. for each band `b` and each team market `m` in `hc25..hc55, ac25..ac55`, train `P(m | total_band=b, x, prior)`
- **Testing:** unit-test label creation and the cumulative mapping from band probabilities to `c75/c85/c95/c105`.

### Step 3: Implement total-band classifier and chrono OOF totals calibration
- **Files:** `src/modeling/v2/families/corners/train_corners.py`
- **Do:** fit one multiclass total-band classifier (likely `HistGradientBoostingClassifier`), then convert band probabilities to totals markets:
  - `c75 = p(eq8)+p(eq9)+p(eq10)+p(ge11)`
  - `c85 = p(eq9)+p(eq10)+p(ge11)`
  - `c95 = p(eq10)+p(ge11)`
  - `c105 = p(ge11)`
- **Key details:** fit OOF total-band predictions and reuse `evaluate_market_calibration()` / `apply_binary_calibrator()` / blend-weight logic per total market versus the NB prior from the backbone.
- **Testing:** assert totals are inherently monotone even before projection and remain clipped after calibration/blending.

### Step 4: Implement band-conditional team heads and marginalization
- **Files:** `src/modeling/v2/families/corners/train_corners.py`
- **Do:** for each total band, fit 8 binary heads for the team markets on the subset of rows inside that band.
- **Key details:**
  - use band-specific fallback constants when a band/market has too few rows or only one class;
  - build head inputs from current features plus backbone prior team probs (`_build_team_head_frame(...)`) and optionally the 5 predicted band probs;
  - at inference compute `P(team_market)` as `sum_b P(b|x) * P(team_market|b,x)`.
- **Testing:** add tests that the marginal team outputs are present, in-range, and monotone after `_project_team_head_prob_arrays(...)`.

### Step 5: Keep backbone means for contract compatibility
- **Files:**
  - `src/modeling/v2/families/corners/train_corners.py`
  - `src/modeling/v2/families/corners/predict_corners.py`
- **Do:** keep `home_mu`, `away_mu`, and `total` from the existing totals-first backbone.
- **Key details:** the new path should use direct overrides for totals/team markets while preserving current metadata fields and `derive_and_validate_corners(...)` usage.
- **Testing:** preserve the current invariant that `preds["home"] + preds["away"] == preds["total"]` for the new path.

### Step 6: Persist repo-native artifacts and config
- **Files:**
  - `src/modeling/v2/families/corners/train_corners.py`
  - `src/modeling/v2/families/corners/predict_corners.py`
- **New artifact files:**
  - `total_band_model.pkl`
  - `band_team_market_heads.joblib`
  - `total_band_total_market_calibrators.joblib`
  - `total_band_team_market_calibrators.joblib`
- **`model_config.json` additions:**
  - `path_version: total_band_state_ladder`
  - `total_band_labels`
  - `direct_total_markets`
  - `direct_team_markets`
  - `total_band_total_blend`
  - `total_band_team_blend`
  - `total_band_min_rows`
- **Testing:** verify round-trip save/load and backward-safe path detection in `load_artifacts()`.

### Step 7: Wire prediction-time derivation
- **Files:** `src/modeling/v2/families/corners/predict_corners.py`
- **Do:** add a prediction branch that:
  1. gets backbone means;
  2. predicts total-band probabilities;
  3. derives/calibrates totals-market overrides;
  4. evaluates all band-conditional team heads;
  5. marginalizes to final `hc*` / `ac*` overrides;
  6. sends overrides through `derive_and_validate_corners(...)`.
- **Testing:** extend predict utils tests for metadata, artifact loading, and direct market override behavior.

### Step 8: Add focused validation and smoke evaluation
- **Files:**
  - `tests/v2/test_corners_train_utils.py`
  - `tests/v2/test_corners_predict_utils.py`
  - `tests/v2/test_corners_line_monotonicity.py`
- **Do:** add one synthetic-path test similar to the existing PMF/team-head tests, plus artifact/config tests.
- **Smoke commands:**
  - `pytest -q tests/v2/test_corners_train_utils.py tests/v2/test_corners_predict_utils.py tests/v2/test_corners_line_monotonicity.py`
  - optional small trainer smoke using `--path-version total_band_state_ladder --max-rows ...`
- **Evaluation follow-up:** only after smoke stability, run the normal corners challenger flow via the existing v2 evaluation harness.

## 4. File Changes Summary

### Modified
- `src/modeling/v2/families/corners/train_corners.py`
- `src/modeling/v2/families/corners/predict_corners.py`
- `tests/v2/test_corners_train_utils.py`
- `tests/v2/test_corners_predict_utils.py`
- `tests/v2/test_corners_line_monotonicity.py`

### Created
- artifact directories under `model_artifacts/v2/` for the actual candidate run
- this plan file: `plan-corners-total-band-state-ladder.md`

### Deleted
- none expected

## 5. Testing Strategy

- **Unit tests:** band-target helper, totals-band-to-market mapping, marginal team mixture logic, save/load config round-trip.
- **Integration tests:** one end-to-end synthetic trainer/predictor path analogous to existing PMF and calibrated-team-head tests.
- **Manual/smoke:** small capped train run, then `predict_corners.py` against the emitted artifact dir.
- **Promotion/eval:** reuse current `run_evaluation_flow.py`; no contract changes should be needed outside the family artifact dir.

## 6. Rollback Plan

- Revert the new `path_version` branches in `train_corners.py` and `predict_corners.py`.
- Delete any candidate artifact dirs created for `corners_total_band_state_ladder_v1_candidate_*`.
- No DB rollback required because this path only affects challenger artifacts/predictions.

## 7. Estimated Effort

- **Complexity:** medium
- **Rough estimate:**
  - plumbing + helper functions: 2-3 hours
  - OOF calibration/blend logic: 2-4 hours
  - tests + smoke runs: 2-3 hours
  - first bounded evaluation rerun: 1-2 hours
- **Main uncertainty:** whether band-specific team heads have enough support in `tb_ge_11` and `tb_le_7`; expect shrinkage/fallback design to matter.

## Likely benefits
- Better dependence handling between totals and team-corners markets.
- Stronger alignment with the actual production lines instead of dense count PMFs.
- More meaningful structural contrast than another residual or another independent-PMF tweak.

## Major risks
- Corners may still be fundamentally signal-limited, so even a cleaner structure may not transfer live.
- Sparse band slices could make the conditional heads noisy without aggressive fallback/shrinkage.
- If OOF band predictions are weak, the mixture can amplify totals-model error into team markets.
