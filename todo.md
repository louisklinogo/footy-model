# Project Todo - Soccer Predictive Model

Updated: 2026-02-26

Legend:
- [x] done
- [ ] pending
- [ ] in progress (marked inline)

## Quick File Reference Map
- Layer 2 training: `src/modeling/layer2_situational/train_situational_residual.py`
- Layer 2 inference: `src/modeling/layer2_situational/predict_situational_residual.py`
- Feature health audit: `src/modeling/layer2_situational/audit_feature_health.py`
- Ablation table: `src/modeling/layer2_situational/run_ablation_table.py`
- Monitoring: `src/modeling/layer2_situational/monitor_layer2_signals.py`
- Leakage audit: `src/db/audit_layer2_situational_leakage.py`
- Calibration check: `src/modeling/evaluation/calibration_check.py`
- Market predictor: `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py`
- Market exporter: `src/modeling/export/export_market_outcomes_fixtures_first.py`
- Market scorer: `src/modeling/evaluation/score_market_outcomes_fixtures_first.py`
- Historical odds backfill: `src/ingest/backfill_sofascore_odds_markets_v1.py`
- Prematch odds polling: `src/jobs/sofascore_odds_polling.py`
- Validation loop scheduler wrapper: `scripts/run_validation_loop_scheduler.cmd`
- Validation loop task registration helper: `scripts/register_validation_loop_scheduler.cmd`
- Tick job (Sofa-first settlement, orchestration wiring pending cleanup): `src/jobs/tick_due_fixtures_v1.py`
- Sofa stale/FT reconciliation: `scripts/reconcile_stale_ft_matches.py`
- Shared script logger: `src/common/script_logger.py`
- DB functions to migrate: `migrations/002_db_functions.sql`
- Sofa settlement helper migration: `migrations/011_sofa_settlement_helpers.sql`
- Poisson / Dixon-Coles engine: `src/pricing/poisson.py`

## Current Snapshot (Reality Check)
- Core Layer 2 guardrails are implemented (chronology fail-fast, rolling temporal CV, odds leakage guard, reproducibility sidecar).
- Biggest blockers are still data coverage, not missing modeling code.
- `odds_model_gap_home` is still sparse in the latest health report (non-missing ~16.2%).
- Upcoming `player_availability` coverage was 0% in latest monitor snapshot.
- Settlement migration has materially progressed: stale FT reconciliation is now Sofa-native and successful in live runs.
- Validation loop automation is now in place (manual runner + scheduled task every 6 hours).
- Operational gap remains in orchestration path alignment (tick still references moved/renamed scripts in some phases).

## NOW (Execution Blockers)

### 0) Tick Orchestration Path Alignment (Critical)
- [ ] in progress - Align `src/jobs/tick_due_fixtures_v1.py` subprocess paths with current repo layout.
  - Canonical paths now expected:
    - `src/ingest/scrapers/premium_enricher_v4.js`
    - `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py`
    - `src/modeling/export/export_market_outcomes_fixtures_first.py`
    - `src/modeling/evaluation/score_market_outcomes_fixtures_first.py`
  - DoD:
    - one live tick run completes settle + predict + export + score without missing-file errors.
    - pipeline run status for `tick_due_fixtures_v1` and phase jobs is `success`.

- [x] Resolve model identity contract between predict/export/score.
  - Canonical identity:
    - `model_name='market_outcome_gbm'`
    - `model_version='fixtures_first_prematch_v1'`

### 1) Historical + Prematch Odds Coverage (Highest ROI)
- [ ] in progress - Backfill historical Sofascore odds into `fixture_odds_markets` (closing snapshots)
  - Goal: provide point-in-time market signal for `odds_model_gap_*` features.
  - Script: `src/ingest/backfill_sofascore_odds_markets_v1.py`
  - Source logic: Sofascore `/event/{sofascore_id}/odds/1/all`, provider id default `1` (Bet365).
  - Feature logic: parse fractional/decimal -> implied probabilities -> compute `odds_model_gap_home/draw/away` vs Poisson outcome probs.
  - Leakage rule: training/inference joins only rows with `snapshot_time_utc <= match_datetime_utc`.
  - DoD:
    - `odds_model_gap_home` coverage materially above current baseline.
    - no post-kickoff odds rows consumed in train/predict.

- [ ] in progress - Run recurring prematch odds polling for upcoming fixtures
  - Script: `src/jobs/sofascore_odds_polling.py`
  - Cadence logic in code:
    - every 2 hours until T-3h
    - every 30 minutes in last 3 hours
  - Writes `latest_pre_match` snapshots into `fixture_odds_markets`.
  - DoD:
    - active upcoming fixtures get fresh pre-kickoff snapshots.
    - polling job is repeatable and idempotent via unique constraint.

- [ ] Verify odds signal health before every retrain
  - Scripts: `src/modeling/layer2_situational/audit_feature_health.py`, `src/modeling/layer2_situational/monitor_layer2_signals.py`
  - Checks:
    - `odds_model_gap_*` not constant
    - drift check available (not blocked by insufficient rows)
    - expected distribution range (no obvious parsing artifacts)

### 2) Player Availability Coverage (2nd Highest ROI)
- [ ] Raise historical `player_availability` coverage to target >70% before production-weighting injury features.
  - Ingestion script: `src/ingest/ingest_sofascore_availability.py`
  - Coverage report: `src/ingest/report_player_availability_coverage.py`
  - Leakage audit: `src/db/audit_layer2_situational_leakage.py` (`recorded_at <= kickoff`)
  - DoD:
    - per-league coverage report produced for target seasons.
    - timing violations remain zero (or explicitly investigated).

### 3) Validation Loop After Each Backfill Batch
- [x] Add automation wrapper for validation loop with lock/stale-lock/logging.
  - Script: `scripts/run_validation_loop_scheduler.cmd`
  - Outputs:
    - `artifacts/reports/layer2_feature_health/<run_ts>/...`
    - `artifacts/reports/leakage_audit/leakage_audit_<run_ts>.json`
    - `artifacts/reports/monitoring/<run_ts>/...`
- [x] Register and verify recurring Task Scheduler run for validation loop.
  - Suggested task name: `FootyLayer2ValidationLoop`
  - Registration helper: `scripts/register_validation_loop_scheduler.cmd`
  - Cadence: every 6 hours
- [ ] Run feature health audit and archive outputs.
- [ ] Run leakage audit and archive outputs.
- [ ] Run monitor report for drift/coverage/same-kickoff groups.
- [ ] Retrain only when coverage gates improve enough to make re-train informative.

Suggested batch loop command set:
1. `python src/modeling/layer2_situational/audit_feature_health.py`
2. `python src/db/audit_layer2_situational_leakage.py --days 14 --limit 500`
3. `python src/modeling/layer2_situational/monitor_layer2_signals.py --days 14`

### 4) Sofa-Native Settlement Migration (Operational)
- [x] Refactor DB functions to prioritize/return Sofa identifiers where appropriate.
  - Current functions: `get_stale_fixtures()`, `get_ft_without_results()` in `migrations/002_db_functions.sql`
- [ ] in progress - Refactor tick settlement path away from Flashscore-only JSON enrichment path.
  - Current file: `src/jobs/tick_due_fixtures_v1.py`
- [x] Add Sofa-native stale FT reconciliation script.
  - Target script: `scripts/reconcile_stale_ft_matches.py`
- [x] Keep controlled fallback behavior during migration (no blind cutover).

### 5) Runtime Logging + Observability (Operational)
- [x] Add file logging + DB run metadata to tick orchestration.
  - `src/jobs/tick_due_fixtures_v1.py`
  - `pipeline_runs.details_json.log_file`
- [x] Add file logging + DB run metadata to stale reconcile script.
  - `scripts/reconcile_stale_ft_matches.py`
- [x] Add file logging + DB run metadata to prematch odds polling.
  - `src/jobs/sofascore_odds_polling.py`
- [ ] pending - Standardize this logger pattern across remaining operational jobs.

## NEXT (After Coverage Improves)

### 6) Retrain + Evaluate
- [ ] Retrain Layer 2 situational residual model and refresh deployment artifacts.
  - Script: `src/modeling/layer2_situational/train_situational_residual.py --train`
  - Expected outputs:
    - `model_artifacts/situational_model/situational_model.pkl`
    - `model_artifacts/situational_model/layer2_deployment_policy.json`
    - `model_artifacts/situational_model/situational_model.meta.json`

- [ ] Run ablation table and compare stage lifts.
  - Script: `src/modeling/layer2_situational/run_ablation_table.py`
  - Stages to keep:
    1. baseline (Layer 1 only)
    2. + standings/points
    3. + schedule
    4. + player impact
    5. + odds gap

- [ ] Re-run calibration diagnostics on prediction outputs.
  - Script: `src/modeling/evaluation/calibration_check.py`
  - Outputs: ECE + reliability diagrams.

### 7) Risk Segmentation + Deployment Governance
- [ ] Add explicit cold-start segment (`either team <6 prior games`) to evaluation output.
- [ ] Validate per-league deployment policy gates after each retrain.
- [ ] Track enabled leagues over time and compare out-of-sample stability.

## LATER (R&D / Expansion)

### 8) Modeling Enhancements
- [ ] Investigate strict Dixon-Coles normalization in `src/pricing/poisson.py` (marginal mean preservation).
- [ ] Add H1 vs H2 dynamics features (fatigue, xG momentum, halftime response).
- [ ] Explore tactical archetype / style-on-style features (`style_delta`) under strict point-in-time controls.

### 9) Data Scope Expansion
- [ ] Expand `team_rivalries` coverage where useful (current seed: `src/ingest/seed_rivalries.py`).
- [ ] Include domestic cups where they improve congestion/fatigue signals.
- [ ] Add `shots_inside_box` into engineered training features only if lift is proven.
- [ ] Tactical formation tracking:
  - [ ] create normalized `fixture_formations` table
  - [ ] preserve formation keys in Sofascore ingestion path

## Evidence Notes (Feb 2026)

### Why some features were near-zero importance
- Odds signal was sparse in training windows, so expected impact was muted.
- Rare flags (`is_derby`, lame duck) have low prevalence and are hard to split on with current sample sizes.
- Injury impact features need stronger historical population before expecting stable lift.

### Keep these guardrails permanently
- [x] strict chronology separation (`max(train_dt) < min(test_dt)`)
- [x] rolling temporal CV (3 folds)
- [x] odds snapshot timing guard (`snapshot_time_utc <= kickoff`)
- [x] leakage audit for availability timing
- [x] reproducibility sidecar (feature hash, params, git hash)

## Implemented (Do Not Re-Plan)
- [x] Calibration tooling (ECE + reliability diagram): `src/modeling/evaluation/calibration_check.py`
- [x] Time-respecting split + chronology fail-fast: `src/modeling/layer2_situational/train_situational_residual.py`
- [x] Rolling temporal CV in Layer 2 training.
- [x] Odds leakage guard in train/predict joins.
- [x] Reproducibility sidecar output (`situational_model.meta.json`).
- [x] Leakage audit checks for availability timing.
- [x] Layer 2 monitor for odds drift, availability coverage, same-kickoff grouping.
- [x] Layer 2 ablation runner scaffold.
- [x] Key player absence feature path.
- [x] European competitions integrated into fixtures-first flow.
- [x] Sofa settlement helper migration and v2 stale/FT helper functions.
- [x] Sofa-native stale FT reconciliation script (`scripts/reconcile_stale_ft_matches.py`).
- [x] Operational log file plumbing + `pipeline_runs.details_json.log_file` for tick/reconcile/odds polling.
