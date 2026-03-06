## 1. Overview

Implement a conservative v2 calibration stage that runs after `src/modeling/v2/eval/run_walkforward.py` and before `src/modeling/v2/eval/promotion_registry.py`.

Goals:
- Turn calibration into a first-class, auditable step in the v2 evaluation flow.
- Select and evaluate calibration out-of-sample per market/family.
- Preserve current promotion logic shape, but let promotion consume calibrated holdout evidence when available.
- Keep scope minimal: reuse existing `pandas`/`numpy`/`sklearn` stack and current artifact conventions under `model_artifacts/v2/...`.

Success criteria:
- No market is fit and scored on the same calibration rows.
- Selection prefers chronological splitting when possible.
- Promotion output clearly shows raw vs calibrated evidence, selected method, and reasons.
- Targeted tests cover split logic, method selection, flow integration, and promotion behavior.

Out of scope for this phase:
- Applying calibration in live inference/predict scripts.
- Per-league or hierarchical calibrators.
- Baseline registry schema changes or DB migrations.

## 2. Prerequisites

Dependencies/setup:
- Use existing repo stack only: `numpy`, `pandas`, `scikit-learn`.
- Inputs already available from trainers: `<family>/holdout_predictions.csv`, `<family>/metrics_holdout.json`, `<family>/training_report.json`.
- Aggregated input already available from evaluation harness: `model_artifacts/v2/evaluation*/evaluation_report.json`.

Data assumptions:
- `holdout_predictions.csv` contains `market`, `fixture_id`, `y_true`, `p_model`, and sometimes `league_code`.
- If an explicit chronological column is present later (for example `match_datetime_utc`), calibration should use it.
- Minimal implementation should work without trainer changes by using per-market CSV row order as the fallback temporal order.

No migrations/data backfills required.

## 3. Implementation Steps

### Step 1: Add calibration method and split utilities
- Create `src/modeling/v2/calibration/methods.py`.
- Implement a small, explicit method registry with:
  - `identity` (no-op baseline)
  - `platt` / sigmoid calibration using sklearn logistic regression on clipped probabilities or logits
  - `isotonic` using `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`
- Add helpers for:
  - clipping probabilities to current repo convention (`0.001` to `0.999`)
  - fitting/predicting each method
  - chronological train/eval splitting
  - scoring predictions via existing `src/modeling/v2/eval/metrics.py` helpers
- Selection rules should be conservative:
  - split each market holdout into an early fit segment and later eval segment (for example 70/30)
  - require minimum support on both sides (`min_fit_rows`, `min_eval_rows`) and both classes in fit/eval for fitted methods
  - always evaluate `identity` on the eval segment
  - only select a fitted method if it improves calibration quality on eval rows and does not materially harm discrimination
- Suggested method policy:
  - default winner metric: lowest eval `brier`
  - tie-breakers: lower `log_loss`, then lower `ece`
  - hard guard: `auc` cannot drop by more than a small tolerance versus raw/identity eval
  - extra conservatism: only allow isotonic when eval support is comfortably large (for example `>= 200` rows)

Testing considerations:
- Unit-test split non-overlap.
- Unit-test insufficient-data fallback to `identity`.
- Unit-test selection prefers `identity` when a fitted method helps Brier but breaches AUC guard.

### Step 2: Add calibration runner and artifact generation
- Create `src/modeling/v2/calibration/run_calibration.py`.
- Keep the runner minimal by reading the already-built `evaluation_report.json` rather than discovering family dirs itself.
- For each family in `evaluation_report["families"]`:
  - read `holdout_predictions_path`
  - group rows by `market`
  - run split -> fit candidates on fit rows -> score candidates on eval rows -> choose method
  - after method selection, optionally refit the chosen method on all eligible market rows for future reuse, but keep promotion metrics tied to the untouched eval rows
- Write per-family artifacts under each family artifact dir:
  - `calibration_report.json` with per-market selection diagnostics
  - `calibration_models.joblib` with selected fitted calibrators for later inference work
  - optional `calibration_eval_predictions.csv` for debugging/auditability
- Write aggregate artifact under the evaluation dir:
  - `calibration_report.json`

Recommended aggregate shape:
- top-level metadata: `generated_at_utc`, `source_evaluation_report`, `rules`
- `families`: artifact paths and counts
- `selection_by_market`: per-market family, split info, selected method, status, raw eval metrics, candidate eval metrics, selected eval metrics, deltas
- `calibrated_by_market`: flattened per-market eval metrics that promotion can consume directly
- `raw_eval_by_market`: raw/identity eval metrics for comparison
- `missing_markets`: markets with no usable calibration input

Testing considerations:
- Temp-dir runner test modeled after `tests/v2/test_evaluation_harness.py`.
- Assert report contains both raw and calibrated metrics and writes family-level files.

### Step 3: Integrate calibration into the evaluation flow
- Modify `src/modeling/v2/run_evaluation_flow.py`.
- Insert a new step after `eval.walkforward` and before `eval.promotion_registry`:
  - `eval.calibration -> python src/modeling/v2/calibration/run_calibration.py --evaluation-report <evaluation_dir>/evaluation_report.json --output-dir <evaluation_dir>`
- Add a few conservative CLI args that forward to calibration (for example `--calibration-eval-fraction`, `--calibration-min-fit-rows`, `--calibration-min-eval-rows`, `--calibration-auc-drop-tolerance`).
- Keep defaults strict and simple so dry-runs and current tests remain easy to reason about.

Testing considerations:
- Extend `tests/v2/test_evaluation_flow_runner.py` to assert the new step exists in the correct order and forwards calibration args.

### Step 4: Integrate calibration into promotion decisions
- Modify `src/modeling/v2/eval/promotion_registry.py`.
- Add optional `--calibration-report` input.
- Promotion should continue using walk-forward support/fold gates exactly as today.
- For holdout-side comparison, promotion should choose an effective holdout row per market:
  - prefer `calibrated_by_market[market]` from calibration report when present
  - otherwise fall back to existing `evaluation_report["holdout_by_market"][market]`
- Keep baseline comparison minimal and familiar:
  - compare effective holdout `auc`, `brier`, `ece` against baseline thresholds/tolerances
  - include calibration metadata in each market row (`selected_method`, `status`, split counts, raw vs calibrated deltas)
- Add explicit but conservative reasoning:
  - do not fail only because a market fell back to `identity`
  - do fail when the effective calibrated row breaches current gates (`auc_failed`, `brier_failed`, `ece_failed`, etc.)
  - optionally add `calibration_missing` only when a calibration report is supplied but the market has no usable selection entry

Testing considerations:
- Extend `tests/v2/test_evaluation_harness.py` (or add a focused promotion test) with:
  - raw holdout fails but calibrated eval passes -> promotion passes
  - fitted method improves Brier but harms AUC too much -> promotion uses identity/fails per guard

### Step 5: Add focused calibration tests
- Add `tests/v2/test_calibration_methods.py` for split/method/selection behavior.
- Add `tests/v2/test_calibration_runner.py` for aggregate artifact creation and schema checks.
- Keep tests synthetic and file-based; no DB requirements.

## 4. File Changes Summary

Created:
- `src/modeling/v2/calibration/methods.py`
- `src/modeling/v2/calibration/run_calibration.py`
- `tests/v2/test_calibration_methods.py`
- `tests/v2/test_calibration_runner.py`
- `plan-v2-calibration-stage.md`

Modified:
- `src/modeling/v2/run_evaluation_flow.py`
- `src/modeling/v2/eval/promotion_registry.py`
- `tests/v2/test_evaluation_flow_runner.py`
- `tests/v2/test_evaluation_harness.py`

Deleted:
- None

## 5. Testing Strategy

Unit tests:
- split logic is chronological and non-overlapping
- method fit/predict returns clipped probabilities in range
- sparse/single-class markets fall back to `identity`
- selection favors conservative winner under AUC guard

Integration/file-based tests:
- runner reads temp `evaluation_report.json` + family `holdout_predictions.csv` and writes expected reports
- promotion registry prefers calibrated metrics when present
- evaluation flow plan includes calibration step in the middle

Manual verification:
- run `python src/modeling/v2/calibration/run_calibration.py --evaluation-report model_artifacts/v2/evaluation/evaluation_report.json --output-dir model_artifacts/v2/evaluation`
- inspect one market where calibration is selected and one where it falls back to identity
- run promotion registry with the new calibration report and confirm market rows expose the calibration context

## 6. Rollback Plan

- Remove the `eval.calibration` step from `src/modeling/v2/run_evaluation_flow.py`.
- Stop passing `--calibration-report` into promotion.
- Ignore or delete generated calibration artifacts:
  - `<family>/calibration_report.json`
  - `<family>/calibration_models.joblib`
  - `<evaluation_dir>/calibration_report.json`
- No schema or data rollback is needed.

## 7. Estimated Effort

- Rough estimate: 1 to 2 focused engineering days.
- Complexity: medium.
- Main risks:
  - sparse markets causing over-eager fitted calibrators
  - ambiguity in chronology when only CSV row order is available
  - ensuring promotion uses only out-of-sample calibration evidence

Recommended minimal-first posture:
- ship with global-per-market calibrators only
- allow identity fallback freely
- keep promotion unchanged except for consuming calibrated holdout evidence and exposing the audit trail

