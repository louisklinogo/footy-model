## 1. Overview

Implement one additive corners path: `totals_first_market_heads_calibrated`.

- **What it is:** keep the existing `totals_first` backbone for `c75/c85/c95/c105`, keep the existing direct team heads from `totals_first_market_heads`, then add **OOF calibration + prior blending** for `hc*`/`ac*` using the anytime `direct_monotone` pattern.
- **Why this is the narrowest high-signal path:** recent live-replacement runs already show plain `totals_first` is the strongest corners baseline (`auc 0.7099 / brier 0.2143 / log_loss 0.6179`) versus `totals_first_market_heads` (`0.7039 / 0.2166 / 0.6241`) and `totals_first_residual` (`0.6969 / 0.2183 / 0.6262`). So the next step should be a small extension on the strongest backbone, not a new residual branch.
- **Success criteria:**
  - Preserve current train/predict/eval artifact seams.
  - Keep all existing corners path versions working unchanged.
  - Match or stay effectively neutral to `totals_first` on total corners markets.
  - Improve `hc25/hc35/hc45/hc55/ac25/ac35/ac45/ac55` relative to current `totals_first_market_heads`.
- **Out of scope:** refactoring anytime into shared infrastructure, changing market scope, modifying evaluation flow machinery, or replacing `totals_first` as the default before benchmark proof.

## 2. Prerequisites

- No new dependencies, migrations, or DB changes.
- Reuse existing calibration utilities in `src/modeling/v2/calibration/methods.py`.
- Reuse current corners derivation seam in `src/modeling/v2/families/corners/derive_lines.py` (`total_mu_override`, `team_market_overrides`, monotone projection).
- Reuse current benchmark anchors:
  - `model_artifacts/v2/corners_totals_first_v1_candidate_20260309`
  - `model_artifacts/v2/corners_totals_first_market_heads_v1_candidate_20260309`
  - `artifacts/v2/family_replacement/live_replacement_20260309_corners_totals_first_v1`
  - `artifacts/v2/family_replacement/live_replacement_20260309_corners_totals_first_market_heads_v1`

## 3. Implementation Steps

### Step 1: Add the new path as an additive variant
- **Files to modify:** `src/modeling/v2/families/corners/train_corners.py`, `src/modeling/v2/families/corners/predict_corners.py`
- Add `totals_first_market_heads_calibrated` to the corners `path_version` choices/constants.
- Treat it as:
  - a `TOTALS_FIRST_PATHS` member,
  - a `MARKET_HEAD_PATHS` member,
  - but **not** a new artifact family with different output schema.
- Keep `sum_heads`, `totals_first`, `totals_first_residual`, and `totals_first_market_heads` behavior unchanged.
- **Testing note:** add a loader/backward-compatibility test so old artifacts still resolve as before.

### Step 2: Implement calibrated/blended team heads in corners training
- **Files to modify:** `src/modeling/v2/families/corners/train_corners.py`
- Keep current total-corners model + home-share split exactly as in `totals_first`.
- Keep current direct team binary heads exactly as in `totals_first_market_heads`.
- Add the smallest possible local helpers, copied/adapted from anytime `direct_monotone` rather than doing a cross-family refactor:
  - calibrator selection logic matching `_select_binary_calibrator(...)`
  - blend-weight fitting matching `_fit_blend_weight(...)`
- Build a **time-respecting OOF head prediction frame** for the eight team markets, following the anytime pattern:
  - derive prior team probabilities from the totals-first backbone,
  - fit raw direct heads on train slices,
  - score later slices to get OOF raw head probabilities,
  - fit per-market calibrators (`identity`/`sigmoid`/`isotonic`) on OOF predictions,
  - fit per-market blend weights against the totals-first prior market probabilities.
- Save additive metadata/artifacts only:
  - `team_market_head_calibrators.joblib`
  - `model_config.json` entries for `path_version` and `team_market_head_blend`
- **Testing note:** assert models include calibrators/blend only for the new path.

### Step 3: Apply calibrated/blended head overrides at prediction time
- **Files to modify:** `src/modeling/v2/families/corners/predict_corners.py`
- Extend artifact loading to read optional `team_market_head_calibrators.joblib` and blend metadata from `model_config.json`.
- Preserve current auto-detection behavior:
  - config-driven when `model_config.json` exists,
  - file-presence fallback for legacy artifacts.
- For the new path only:
  - derive the totals-first prior team probabilities first,
  - score raw direct team heads,
  - apply calibrators,
  - blend calibrated head outputs back toward the totals-first prior,
  - pass the final values through the existing monotone projection/override path.
- Do **not** change the shape of `holdout_predictions.csv` or downstream evaluation inputs.
- **Testing note:** add a temp-artifact roundtrip test for old vs new artifact layouts.

### Step 4: Reuse existing corners derivation seam instead of changing market math
- **Files to modify:** likely none; only touch `src/modeling/v2/families/corners/derive_lines.py` if a tiny helper extraction makes training code cleaner.
- Keep current total/team probability derivation and monotonicity enforcement intact.
- The new path should supply better `team_market_overrides`; it should not introduce a second derivation path for corners markets.
- **Testing note:** existing monotonicity tests should continue to pass without semantic changes.

### Step 5: Add focused tests for the new path and backward compatibility
- **Files to modify:** `tests/v2/test_corners_train_utils.py`
- **Files to create:** `tests/v2/test_corners_predict_utils.py`
- Add one training test that verifies for `totals_first_market_heads_calibrated`:
  - `home + away == total` still holds,
  - `team_market_probs` exists for all eight team markets,
  - team sequences remain monotone after overrides,
  - calibrator/blend metadata exists.
- Add one loader test modeled after existing predict util tests in v2 families:
  - legacy `totals_first_market_heads` artifact loads without new files,
  - new calibrated artifact loads with calibrators/blend metadata,
  - missing optional calibrator file falls back cleanly.

### Step 6: Benchmark in the smallest useful sequence
- **Files to modify:** none required in evaluation flow.
- Benchmark order:
  1. Use `corners_totals_first_v1_candidate_20260309` as the primary control.
  2. Use `corners_totals_first_market_heads_v1_candidate_20260309` as the ablation control for direct heads without calibration/blending.
  3. Train `corners_totals_first_market_heads_calibrated_v1_candidate_<date>`.
  4. Compare holdout and walkforward metrics by market group:
     - totals: `c75/c85/c95/c105`
     - team: `hc*` and `ac*`
  5. Run the existing evaluation flow to generate the same artifacts now used by corners candidates.
  6. Run the existing live-replacement comparison used by `artifacts/v2/family_replacement/live_replacement_20260309_*`.
- Promotion bar for this path:
  - keep required/core gate passing,
  - beat raw `totals_first_market_heads` on team-market brier/log-loss,
  - avoid material regression versus plain `totals_first` on totals markets and corners-overlap summary.

## 4. File Changes Summary

- **Create:**
  - `tests/v2/test_corners_predict_utils.py`
- **Modify:**
  - `src/modeling/v2/families/corners/train_corners.py`
  - `src/modeling/v2/families/corners/predict_corners.py`
  - `tests/v2/test_corners_train_utils.py`
- **Likely unchanged:**
  - `src/modeling/v2/families/corners/derive_lines.py`
  - `src/modeling/v2/run_evaluation_flow.py`
  - `src/modeling/v2/eval/live_replacement_compare.py`
- **Delete:** none

## 5. Testing Strategy

- **Unit tests:**
  - `pytest -q tests/v2/test_corners_train_utils.py`
  - `pytest -q tests/v2/test_corners_line_monotonicity.py`
  - `pytest -q tests/v2/test_corners_predict_utils.py`
- **Integration-ish validation:**
  - run a corners training job for the new path and confirm artifact set matches existing seams plus the one new optional calibrator file,
  - run the existing v2 evaluation flow for the new candidate,
  - run live replacement compare against current live runtime.
- **Manual checks:**
  - inspect `model_config.json` for `path_version` and `team_market_head_blend`,
  - confirm `holdout_predictions.csv` still contains the same market rows/schema,
  - confirm follow-up failures remain concentrated in corners tail/team markets rather than totals-core.

## 6. Rollback Plan

- Revert the new `path_version` handling in corners train/predict code.
- Stop producing/reading `team_market_head_calibrators.joblib` and `team_market_head_blend`.
- Keep using existing `totals_first` or `totals_first_market_heads` artifacts; no data rollback is needed because artifact changes are additive and file-based.

## 7. Estimated Effort

- **Effort:** 1-2 focused implementation days.
- **Complexity:** Medium.
- **Main risk:** OOF calibration/blending may improve `hc*`/`ac*` while still not beating plain `totals_first` on overall corners summary; that is why the benchmark sequence should compare against both the plain totals-first control and the raw market-heads ablation.
