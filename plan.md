# AUC Lift Plan v1 (Weak-Market First, AUC+Brier Governed)

## Summary
This plan improves model discrimination while preserving probability quality, starting with weakest active markets: `1x2_d`, `dc_12`, `o15`, `u35`, `c75`, `c85`, `c95`, `c105`.
Approach: strengthen data parity, switch to time-series walk-forward evaluation, add market-family feature sets (including missing line-specific odds), and tune per-family models.
Production promotion is hard-gated: **AUC improvement >= 0.02** and **Brier non-worse** vs current production model.

## Scope
1. In scope:
   1. Weak-market retraining and evaluation pipeline upgrades.
   2. Feature pipeline additions required for those weak markets.
   3. Risk/gating alignment so weak/non-improved markets are not tradable.
   4. Artifact and report schema changes for reproducible model selection.
2. Out of scope (Phase 1):
   1. Full portfolio retrain of already-strong markets.
   2. Deep multi-task re-architecture unless Phase 1 fails promotion gates.

## Target Markets and Baseline
1. Priority markets:
   1. `1x2_d`, `dc_12`
   2. `o15`, `u35`
   3. `c75`, `c85`, `c95`, `c105`
2. Baseline source:
   1. `model_artifacts/market_models/metrics_walkforward.json`
   2. `auc.txt`
3. Promotion comparator:
   1. Current production artifacts under `model_artifacts/market_models/`.

## Implementation Plan

### Phase 0: Data and Tradability Integrity (Must pass before retrain)
1. Ensure market availability parity by market and line:
   1. Add daily coverage report script: `src/ingest/report_odds_market_coverage.py`.
   2. Report dimensions: `league_code`, `market_code`, `line_num`, kickoff horizon bucket, coverage %.
   3. Output artifacts:
      1. `artifacts/reports/odds_coverage/YYYY-MM-DD.json`
      2. `artifacts/reports/odds_coverage/YYYY-MM-DD.md`
2. Enforce tradable-market intersection:
   1. `tradable = user_linchpins ∩ gating.eligible ∩ coverage>=threshold`.
   2. Threshold default: `coverage>=0.60` for market+line.
   3. Update `risk_policy.json` generation flow to consume this intersection.

### Phase 1: Evaluation Framework Upgrade (AUC+Brier reliable)
1. Replace single 80/20 split with rolling walk-forward folds.
2. Fold design:
   1. Expanding-train, fixed-horizon test windows.
   2. Default folds: `6`.
   3. Min per-fold test rows per market: `200`; if below, skip fold and log.
3. Metrics per market:
   1. Mean AUC, AUC std
   2. Mean Brier, Brier std
   3. ECE (diagnostic)
4. Artifacts:
   1. `metrics_walkforward_folds.json`
   2. `metrics_summary_phase1.json`

### Phase 2: Market-Family Feature Sets (Core AUC lever)
1. Split feature contracts by family:
   1. `features_1x2_dc.json`
   2. `features_totals_goals.json`
   3. `features_corners_totals.json`
2. Add missing line-specific odds features:
   1. For `u35`: add OU `3.5` odds implied probs and margin features.
   2. For corners totals: add `corners_ou` lines `7.5/8.5/9.5/10.5` implied probs and slope features.
3. Add recency-anchor features:
   1. `season_baseline_xg`
   2. `recent_xg_mean_5`
   3. `recent_vs_baseline_zscore`
   4. `regressed_recent_xg`
4. Missing-data handling:
   1. League-median -> global-median fallback.
   2. Missingness indicators for high-impact odds features.

### Phase 3: Modeling Changes (Medium complexity, high ROI)
1. `1X2/DC` family:
   1. Train one multi-class model (`home/draw/away`).
   2. Derive `dc_1x`, `dc_x2`, `dc_12` from class probs.
2. Goals totals family (`o15/u35`):
   1. Train line-aware binary models with `OU1.5/2.5/3.5` features.
3. Corners totals family:
   1. Train per-line binary models with line-specific odds + corners tempo features.
4. Model candidates:
   1. Baseline: GradientBoostingClassifier.
   2. Candidate: LightGBM/XGBoost if available; else HistGradientBoosting.
   3. Hyperparameter search per family (`<=40` trials).

### Phase 4: Calibration and Promotion Logic
1. Calibration policy:
   1. Calibrate after model selection.
   2. Use time-ordered calibration holdout.
   3. Try isotonic and sigmoid; choose lower holdout Brier.
2. Promotion gate:
   1. `AUC_delta >= +0.02`
   2. `Brier_delta <= 0.00`
   3. `AUC_std` stability check.
3. If gate fails:
   1. Keep current production model.
   2. Keep market `predict_only`.

### Phase 5: Policy/Risk Alignment
1. Update risk defaults:
   1. `hard_precision_gate=true`
   2. Remove non-promoted weak markets from tradable list.
2. Add `market_mode` metadata (`tradable|predict_only`).
3. Tradable requires both promotion and coverage gates.

## Interface / Artifact Changes
1. New CLI flags:
   1. `--markets`
   2. `--family`
   3. `--folds`
   4. `--promotion-baseline-path`
2. New artifacts:
   1. `metrics_walkforward_folds.json`
   2. `metrics_summary_phase1.json`
   3. `model_selection_registry.json`
   4. Daily odds coverage artifacts.
3. Metadata additions:
   1. `market_mode`
   2. `model_family`
   3. `promotion_batch_id`

## Test Plan
1. Unit tests:
   1. OU3.5 and corners line extraction.
   2. DC derivation identity checks.
   3. Tradable intersection logic.
2. Integration tests:
   1. End-to-end retrain writes new artifacts.
   2. Risk assessment consumes `market_mode` correctly.
3. Regression tests:
   1. Non-target markets unaffected.
   2. Prediction upsert compatibility preserved.
4. Acceptance tests:
   1. Promotion decision emitted for each target market.
   2. Promoted markets satisfy gate.

## Rollout and Monitoring
1. Sequence:
   1. Offline dry-run retrain.
   2. Report-only week.
   3. Enable tradability per-market after pass.
2. Monitoring:
   1. Weekly AUC/Brier/ECE/coverage/fallback usage per market.
   2. Drift alerts for AUC and coverage drops.
3. Auto fallback:
   1. Revert model artifact.
   2. Set `market_mode=predict_only`.

## Task List (Execution Order)
1. Implement coverage report and tradable intersection.
2. Add walk-forward fold evaluation artifacts.
3. Build market-family feature contracts.
4. Add OU3.5 + corners line-specific features.
5. Implement 1X2/DC multiclass path and derivations.
6. Add candidate search + selection registry.
7. Implement chrono calibration holdout.
8. Add promotion-gate enforcement.
9. Update risk policy integration and `market_mode`.
10. Add/expand unit+integration+acceptance tests.
11. Run Phase 1 retrain and produce promotion report.
12. Apply per-market rollout decisions.
