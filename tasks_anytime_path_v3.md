# Anytime Path V3 Tasks
Date: 2026-03-05
Owner: Modeling
Goal: Improve `h_1up/a_1up/h_2up/a_2up` with structural path modeling (not feature churn).

## Baseline Reality Check
1. Current anytime v2 holdout (reference) is stronger than naive direct classifiers and naive recalibration attempts.
2. `h_1up/a_1up` still remain weak segments in diagnostics, so we need structural lift.
3. Existing model uses constant-rate match-level CTMC pathing; we will test phase-split rates.

## Scope
1. In-scope:
   1. `src/pricing/markov.py`
   2. `src/modeling/v2/families/anytime/*`
   3. `artifacts/reports/v2_feature_block_experiments/*`
2. Out-of-scope:
   1. Full market-stack rewrite.
   2. Live/in-play modeling.

## Tasks
1. [x] Record baseline and disproven alternatives (direct classifier + simple recalibration).
2. [x] Add phase-split anytime pricer support:
   1. Phase 1 rates (early window proxy).
   2. Phase 2 rates (late window proxy).
3. [x] Add anytime trainer option for path-v3 features and phase rates.
4. [x] Export comparable holdout metrics for:
   1. `h_1up`
   2. `a_1up`
   3. `h_2up`
   4. `a_2up`
5. [x] Run weak-segment comparison report (market + league slices).
6. [x] Decide promotion status for anytime path-v3. (rejected)

## Acceptance Gates
1. Family gate:
   1. Mean AUC delta > 0
   2. Mean Brier delta < 0
2. Weak-market gate:
   1. `h_1up` AUC up and Brier down
   2. `a_1up` AUC up and Brier down
3. Stability gate:
   1. No monotonicity break (`2up <= 1up`) for home/away.

## Outputs
1. `artifacts/reports/v2_feature_block_experiments/<timestamp>/compare_anytime_path_v3.md`
2. `artifacts/reports/v2_feature_block_experiments/<timestamp>/compare_anytime_path_v3.json`

## Outcome
1. `Path V3` did not beat the refreshed anytime baseline and was not promoted.
2. The winning anytime recovery path was richer live-safe snapshot features plus tuned Poisson regularization.
3. Promotion candidate moved to `anytime_rich_snapshot_v1` with report in `artifacts/reports/v2_anytime_improvement/20260305T221536Z/`.
