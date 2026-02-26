# Project Todo - Soccer Predictive Model

Updated: 2026-02-26

Legend:
- [x] done
- [ ] pending
- [ ] in progress (marked inline)

## Program Override (Active)
- [ ] in progress - Layer 2 Reconciliation Freeze Program (cross-cutting, decision-critical).
- Source of truth: `docs/plans/2026-02-26-layer2-reconciliation-source-of-truth.md`
- Unified architecture design (decision locked: `override`):
  - `docs/plans/2026-02-26-unified-model-architecture-override-design.md`
- Rule: until this program is resolved, do not proceed with unrelated roadmap tasks.
- Allowed work: Layer 2 data audits, feature audits, train/predict reconciliation, controlled Layer 2 experiments, governance verification.
- Exit gate: all criteria in the source-of-truth document must pass before normal task sequencing resumes.
- Program checklist:
  - [x] Step 1: English Layer 2 charter locked.
  - [x] Step 2: Layer 1 residual anatomy audit completed.
    - Report: `docs/plans/2026-02-26-layer1-residual-anatomy-audit.md`
    - Status: Layer 1 timing lineage repaired (`feature_asof_utc` now anchoring latest lambda pairs pre-kickoff).
  - [x] Step 3: Data source audit across all Layer 2 dependencies.
    - Report: `docs/plans/2026-02-26-layer2-data-source-audit-review.md`
    - Current high-impact findings:
      - `predictions(lambda_xgb)` timing lineage repaired to decision-grade.
      - `player_availability` timing backfilled and clamped to pre-kickoff anchor for historical rows.
      - `team_rivalries` prevalence too low for stable global signal.
      - `fixture_player_stats.expected_goals` missingness remains high.
    - Repair design: `docs/plans/2026-02-26-timestamp-lineage-repair-design.md`
  - [x] Step 4: Feature-by-feature audit ledger and reconciliation matrix (post-repair refresh).
    - Review note: `docs/plans/2026-02-26-layer2-feature-reconciliation-review.md`
    - Ledger artifacts:
      - `artifacts/reports/layer2_reconciliation/layer2_feature_ledger.json`
      - `artifacts/reports/layer2_reconciliation/layer2_feature_ledger.md`
    - Reconciliation matrix artifacts:
      - `artifacts/reports/layer2_reconciliation/layer2_feature_reconciliation_matrix.json`
      - `artifacts/reports/layer2_reconciliation/layer2_feature_reconciliation_matrix.md`
    - Current split (latest refresh):
      - keep candidates: 2
      - conditional candidates: 23
      - drop candidates: 9
      - lineage-blocked (deferred): 2
  - [x] completed - Step 5: Train/predict parity remediation and controlled reruns.
    - Implemented immediate lineage guardrails:
      - `src/modeling/layer1_poisson/predict_lambda.py`: stop overwriting `predictions.created_at` on conflict upsert.
      - `src/ingest/ingest_sofascore_availability.py`: stop overwriting `player_availability.recorded_at` on conflict upsert.
    - Implemented schema + DB lineage work:
      - migration applied: `migrations/013_prediction_availability_lineage.sql`
      - lineage triggers active for `predictions` and `player_availability`
      - one-time lambda lineage backfill: `feature_asof_utc = kickoff - 60m` for legacy `lambda_xgb` rows
      - audit transition completed (`residual anatomy`, `source audit`, `leakage audit`)
    - parity report: `docs/plans/2026-02-26-layer2-train-predict-parity-audit.md`
    - Availability timing repair completed:
      - script: `scripts/repair_player_availability_event_timing.py`
      - repair artifact: `artifacts/reports/layer2_reconciliation/player_availability_event_timing_repair_20260226T171652Z.md`
      - result: `rows_after_kickoff 66026 -> 0`, `fixtures_with_bad_timing 1508 -> 0`
    - Post-repair source evidence:
      - `artifacts/reports/layer2_reconciliation/layer2_source_timing_checks.json`
      - availability historical FT timing now `0` late rows (`0.0%`)
      - lambda latest-pair late rows `0`, odds late chosen snapshots `0`
    - Controlled rerun comparison completed:
      - `artifacts/reports/layer2_reconciliation/layer2_postrepair_full_vs_keyabs_comparison.md`
      - full (36f): home/away RMSE `1.1910 / 1.0783`, enabled leagues `E1, RO1`
      - keyabs-only (33f): home/away RMSE `1.1924 / 1.0762`, enabled leagues `RO1`

- [x] completed - Step 6/7 bridge: Feature intent vs implementation introspection (code + DB).
  - Intent contract:
    - `docs/plans/2026-02-26-layer2-feature-intent-contract.md`
  - Audit review note:
    - `docs/plans/2026-02-26-layer2-feature-intent-implementation-audit.md`
  - Per-feature outputs:
    - `artifacts/reports/layer2_reconciliation/layer2_feature_intent_implementation_audit.json`
    - `artifacts/reports/layer2_reconciliation/layer2_feature_intent_implementation_audit.md`
  - Current audit snapshot:
    - features audited: `36`
    - implemented well: `1`
    - implemented with issues: `35`
  - Top issue classes:
    - mixed incremental value: `23`
    - drop candidates: `9`
    - lineage/source-blocked decisions: `2`
  - [x] Step 9: Final keep/fix/drop/defer sign-off pass completed.
    - Artifact: `artifacts/reports/layer2_reconciliation/layer2_feature_final_signoff.md`
    - Artifact: `artifacts/reports/layer2_reconciliation/layer2_feature_final_signoff.json`
    - Final split: keep `2`, fix `23`, drop `9`, defer `2` (total `36`).
  - [x] Step 8: Final recommendation memo drafted.
    - Memo: `docs/plans/2026-02-26-layer2-final-recommendation-memo.md`
    - Current recommendation: keep `baseline_full_features` as active candidate; keep freeze active for final calibration/serving-readiness sign-off.
  - [x] Step 10: End-to-end player-impact assumption validation run completed.
    - Script: `src/modeling/layer2_situational/validate_player_impact_assumptions.py`
    - Artifacts:
      - `artifacts/reports/layer2_reconciliation/player_impact_assumption_validation.json`
      - `artifacts/reports/layer2_reconciliation/player_impact_assumption_validation.md`
    - Gate status:
      - correctness: `pass`
      - football_logic: `warn`
      - predictive_value: `warn`
      - operational_readiness: `fail`
      - overall: `fail`
    - Core findings:
      - train/predict feature parity is exact (`0` mismatches) and timing-late rows are `0`.
      - player-impact directional logic is unstable (`corr(xg_lost,residual)=+0.0217`, key-absent residual delta `+0.0080`).
      - best predictive variant (`key_absent_only`) gives only marginal lift (triggered `+0.07%`, global `+0.01%`).
      - operational gate fails mainly on upcoming availability coverage (`0/622`) and high `fixture_player_stats.expected_goals` null rate (`71.67%`).

## Layer 1 Stabilization (Active Sub-Track)
- [x] completed - Stabilize Layer 1 before further Layer 2 expansion.
  - Review note: `docs/plans/2026-02-26-layer1-stabilization-sweep-review.md`
  - Sweep evidence:
    - `artifacts/reports/layer1_sweep/layer1_feature_sweep.md`
    - `artifacts/reports/layer1_sweep/layer1_hyperparam_sweep_full.md`
  - Decisions locked:
    - time-decay default: `xi=0.001`
    - XGBoost defaults promoted: `learning_rate=0.025`, `max_depth=3`, `min_child_weight=25`, `subsample=0.7`, `colsample_bytree=0.7`
    - `shots_inside_box` not promoted to Layer 1 production features yet (no lift in sweep)
  - Applied:
    - `src/modeling/layer1_poisson/train_lambda.py` now exposes core training hyperparameters (`--xi`, `--learning-rate`, `--max-depth`, `--min-child-weight`, `--subsample`, `--colsample-bytree`, `--num-boost-round`, `--early-stopping-rounds`)
    - artifact path alignment fixed: trainer now writes model files to `model_artifacts/poisson_model/` (canonical load path for `predict_lambda.py`)
    - Layer 1 confirmatory retrain run completed with promoted defaults
    - partial lambda refresh run completed: `predict_lambda.py --limit 4000 --batch-size 400` (`7092` rows saved)
    - full lambda refresh completed: `predict_lambda.py --batch-size 500` (`20578` rows saved)
    - post-refresh anatomy coverage improved: pre-kickoff pair fixtures `6906 -> 7137`
    - post-stabilization anatomy metrics now near-neutral:
      - full pre-kickoff: home bias `+0.0149`, away bias `-0.0080`, RMSE home/away `1.1959 / 1.0756`
      - recent pre-kickoff (365d): home bias `+0.0108`, away bias `+0.0101`, RMSE home/away `1.1750 / 1.0653`
  - Operational note:
    - one transient DB disconnect occurred during a large-batch refresh; rerun with smaller batch size completed successfully (idempotent upsert path).

## Quick File Reference Map
- Layer 1 sweep (time-decay + shots test): `src/modeling/layer1_poisson/run_layer1_feature_sweep.py`
- Layer 2 training: `src/modeling/layer2_situational/train_situational_residual.py`
- Layer 2 inference: `src/modeling/layer2_situational/predict_situational_residual.py`
- Layer 2 feature intent contract: `docs/plans/2026-02-26-layer2-feature-intent-contract.md`
- Layer 2 feature intent vs implementation audit: `docs/plans/2026-02-26-layer2-feature-intent-implementation-audit.md`
- Layer 2 final recommendation memo: `docs/plans/2026-02-26-layer2-final-recommendation-memo.md`
- Layer 2 final feature sign-off: `artifacts/reports/layer2_reconciliation/layer2_feature_final_signoff.md`
- Layer 2 pruned-vs-full comparison: `artifacts/reports/layer2_reconciliation/layer2_pruned_drop9_vs_full_comparison.md`
- Layer 2 rule-layer segmented backtest: `src/modeling/layer2_situational/backtest_rule_layer_overrides.py`
- Layer 2 global variant comparison runner: `src/modeling/layer2_situational/compare_global_model_variants.py`
- Layer 2 rule-layer policy recommendation: `artifacts/reports/layer2_reconciliation/layer2_three_way_policy_recommendation.md`
- Layer 2 rule-layer engine: `src/modeling/layer2_situational/rule_layer.py`
- Layer 2 player-impact variant sweep runner: `src/modeling/layer2_situational/sweep_player_impact_variants.py`
- Daily orchestrator rule-layer flags: `src/pipelines/daily_pipeline.py`
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
- Availability polling ingester: `src/ingest/ingest_sofascore_availability.py`
- Validation loop scheduler wrapper: `scripts/run_validation_loop_scheduler.cmd`
- Validation loop task registration helper: `scripts/register_validation_loop_scheduler.cmd`
- Availability scheduler wrapper: `scripts/run_availability_scheduler.cmd`
- Availability task registration helper: `scripts/register_availability_scheduler.cmd`
- Deep availability scheduler wrapper: `scripts/run_availability_deep_scheduler.cmd`
- Deep availability task registration helper: `scripts/register_availability_deep_scheduler.cmd`
- Tick job (Sofa-first settlement, orchestration wiring pending cleanup): `src/jobs/tick_due_fixtures_v1.py`
- Sofa stale/FT reconciliation: `scripts/reconcile_stale_ft_matches.py`
- Shared script logger: `src/common/script_logger.py`
- DB functions to migrate: `migrations/002_db_functions.sql`
- Sofa settlement helper migration: `migrations/011_sofa_settlement_helpers.sql`
- Poisson / Dixon-Coles engine: `src/pricing/poisson.py`

## Current Snapshot (Reality Check)
- Core Layer 2 guardrails are implemented (chronology fail-fast, rolling temporal CV, odds leakage guard, reproducibility sidecar).
- Biggest blockers are now model quality and signal stability, not missing modeling code.
- Layer 1 lambda lineage is now repaired to decision-grade timing (`feature_asof_utc` active).
- Layer 1 stabilization is now explicitly active (`xi` sweep complete, `shots_inside_box` tested and deferred).
- `odds_model_gap_home` coverage materially improved in latest health report (non-missing ~89.2%, n=6979/7824).
- Upcoming `player_availability` coverage was 0% in latest monitor snapshot.
- Settlement migration has materially progressed: stale FT reconciliation is now Sofa-native and successful in live runs.
- Validation loop automation is now in place (manual runner + scheduled task every 6 hours).
- Operational gap remains in orchestration path alignment (tick still references moved/renamed scripts in some phases).
- Program state update: Layer 2 reconciliation is now the active blocker and source-of-truth workflow.
- Latest Layer 2 retrain (post-repair, 2026-02-26) is positive but modest:
  - test RMSE vs Layer 1 baseline: home `1.1910` vs `1.1963` (+0.44% lift), away `1.0783` vs `1.0854` (+0.66% lift)
  - deployment policy currently enables leagues: `E1`, `RO1`
  - controlled variant tradeoff:
    - keyabs-only improves away RMSE but worsens home RMSE and removes `E1` from enabled leagues.

## NOW (Execution Blockers)

Global note for this section: items below are paused unless they are directly required by the active Layer 2 reconciliation source-of-truth workflow.

### Immediate Next Tasks (Post Sign-Off)
- [x] Build next global Layer 2 candidate by dropping final-signoff `drop` features (9 features) and retrain in isolated artifacts dir.
  - Source sign-off: `artifacts/reports/layer2_reconciliation/layer2_feature_final_signoff.md`
  - Candidate artifacts: `model_artifacts/situational_model_pruned_drop9/`
  - Comparison: `artifacts/reports/layer2_reconciliation/layer2_pruned_drop9_vs_full_comparison.md`
  - Compare candidate vs current full model on:
    - holdout RMSE (home/away)
    - per-league enablement (`layer2_deployment_policy.json`)
    - temporal CV stability
- [x] Design and implement v1 situational rule-layer override (deterministic, capped adjustments, full logging).
  - Code:
    - `src/modeling/layer2_situational/rule_layer.py`
    - `src/modeling/layer2_situational/predict_situational_residual.py` (`--enable-rule-layer`, metadata logging)
  - Default production-ready config staged:
    - `model_artifacts/situational_model/rule_layer_config.json` (`keyabs_home_only_minus0.120`, supersedes `home_only_v1`)
  - Initial triggers:
    - `home_upcoming_tier` / `away_upcoming_tier`
    - `home_key_absent` / `away_key_absent`
    - `congestion_flag`, extreme `rest_delta`
  - Guardrails:
    - max absolute lambda adjustment cap
    - no override when confidence/coverage gates fail
    - always log fired rules and net adjustment
- [x] Run segmented backtest for rule-layer on triggered fixtures only.
  - Evaluate directional correctness and ROI-like hit metrics by trigger type.
  - Keep global RMSE as secondary check; primary metric is triggered-segment improvement.
- [x] Decide policy from three-way comparison:
  - Layer 1 only
  - Layer 1 + global Layer 2 (pruned candidate)
  - Layer 1 + global Layer 2 + rule-layer overrides
  - Final decision artifact:
    - `artifacts/reports/layer2_reconciliation/layer2_three_way_policy_recommendation.md`
  - Decision:
    - keep `baseline_full_features` as global Layer 2 (pruned candidate not promoted).
    - promote `keyabs_home_only_minus0.120` deterministic rule-layer on top of full Layer 2.
  - Promotion evidence (2026-02-26):
    - sweep artifact: `artifacts/reports/layer2_reconciliation/layer2_rule_layer_key_absent_sweep_step1_6.md`
    - confirmatory backtest: `artifacts/reports/layer2_reconciliation/layer2_rule_layer_segmented_backtest_full_keyabs_home_minus012_v1.md`
    - metric delta vs prior `home_only_v1`:
      - global home lift: `0.20%` vs `0.18%`
      - triggered home lift: `2.45%` vs `1.60%`
      - global away lift unchanged at `0.06%`
- [ ] Keep freeze active until:
  - calibration sample size reaches decision-grade threshold
  - upcoming availability serving coverage gate improves from current `8.14%` (`50/614`)
  - final policy is approved and written to deployment docs.

### 0b) Unified Backbone + Market Orchestration (Override Track)
- [x] Implement overlap-family override semantics in Layer 2 serving path.
  - First overlap family: `key_absent`.
  - Rule: when overlap rule fires for a side, do not double-count equivalent global contribution for that side.
  - File: `src/modeling/layer2_situational/predict_situational_residual.py`
  - DoD:
    - metadata includes explicit `overlap_mode='override'`.
    - side-level attribution is logged (`lambda_before_layer2`, `lambda_after_layer2`, `lambda`, `overlap_family`).
  - Completed: `2026-02-26`
  - Evidence:
    - constrained serving run: `python src/modeling/layer2_situational/predict_situational_residual.py --days 2 --league E1 --enable-rule-layer --rule-overlap-mode override`
    - output: `Upserted 44 records (11 fixtures with adjusted lambdas) for 11 fixtures.`
    - output: `Overlap overrides applied: home=0, away=0`

- [x] Mirror overlap override semantics in backtest evaluation path (train/eval/serve parity).
  - File: `src/modeling/layer2_situational/backtest_rule_layer_overrides.py`
  - Completed: `2026-02-26`
  - Evidence:
    - command: `python src/modeling/layer2_situational/backtest_rule_layer_overrides.py --rule-overlap-mode override --output-stem layer2_rule_layer_segmented_backtest_override_parity_2026_02_26`
    - artifact: `artifacts/reports/layer2_reconciliation/layer2_rule_layer_segmented_backtest_override_parity_2026_02_26.json`
    - artifact: `artifacts/reports/layer2_reconciliation/layer2_rule_layer_segmented_backtest_override_parity_2026_02_26.md`
    - key diagnostics: `n_test=1426`, `overlap_mode=override`, `override_home=50`, `override_away=0`

- [ ] Unify tick predict path with canonical backbone graph.
  - Ensure tick runs:
    1. `predict_lambda.py`
    2. `predict_situational_residual.py --enable-rule-layer`
    3. `predict_market_outcomes_fixtures_first.py`
    4. `export_market_outcomes_fixtures_first.py`
  - File: `src/jobs/tick_due_fixtures_v1.py`
  - DoD:
    - one live run completes full graph with success statuses in `pipeline_runs`.
    - no divergence in model path between `daily_pipeline.py` and tick.

- [ ] Add backbone outputs to market-model feature contract.
  - Add features:
    - `lambda_home_l1`, `lambda_away_l1`
    - `adj_lambda_home_final`, `adj_lambda_away_final`
    - `rule_fired_home`, `rule_fired_away`
  - Files:
    - `src/modeling/layer2_markets/market_outcome_calibrator.py`
    - `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py`
  - DoD:
    - train/predict feature parity check passes.
    - leakage guards remain pass.

- [ ] Implement explicit market fallback path.
  - If market artifact for a market is missing/invalid, fallback to backbone/Poisson-derived probability.
  - File: `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py`
  - DoD:
    - prediction metadata carries `fallback_used`.
    - scorer/export remain stable with mixed artifact availability.

- [ ] Run champion-challenger comparison after unification.
  - Champion: current market stack.
  - Challenger: unified stack + backbone-enhanced market features.
  - Artifacts:
    - holdout metrics
    - calibration deltas
    - triggered segment diagnostics
  - Promotion gate:
    - non-negative global performance
    - no away degradation beyond tolerance
    - calibration sample gate met.

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
- [ ] Raise upcoming scheduled-fixture `player_availability` coverage before production-weighting injury features in serving.
  - Ingestion script: `src/ingest/ingest_sofascore_availability.py`
  - Active automation:
    - hourly near-term window: `scripts/run_availability_scheduler.cmd` (`0h..72h`, nearest kickoff first, 404 cooldown)
    - daily deep-future sweep: `scripts/run_availability_deep_scheduler.cmd` (`72h..336h`)
  - Coverage report: `src/ingest/report_player_availability_coverage.py`
  - Leakage audit: `src/db/audit_layer2_situational_leakage.py` (`recorded_at <= kickoff`)
  - DoD:
    - upcoming-window coverage report produced by league and kickoff horizon.
    - timing violations remain zero (or explicitly investigated).

### 2b) Player-Impact Assumption Remediation (Active)
- [x] Run implementation-assumption checks on player impact construction.
  - Compare current `xg_lost` formula vs alternatives:
    - season-to-date rolling `xG90` baseline
    - recency-weighted `xG90` baseline
    - starts-minutes weighted baseline
  - Sweep artifact:
    - `artifacts/reports/layer2_reconciliation/player_impact_variant_sweep_step2_4.json`
    - `artifacts/reports/layer2_reconciliation/player_impact_variant_sweep_step2_4.csv`
    - `artifacts/reports/layer2_reconciliation/player_impact_variant_sweep_step2_4.md`
  - Result:
    - runs tested: `36`
    - passing runs (logic + predictive): `0`
    - best run: `season_xg90` with `xg_share>=0.12`, `minutes_share>=0.07`
    - best triggered/global lift: `+0.09% / +0.02%` (below promotion threshold)
- [x] Rebuild key-player definition and verify stability.
  - Test thresholds for "key absent" (`xG share`, starts share, minutes share) per league.
  - Current sweep-tested grid:
    - `xg_share_thresholds`: `0.08, 0.10, 0.12`
    - `minutes_share_thresholds`: `0.07, 0.09, 0.11`
  - League sanity artifact for best run:
    - `artifacts/reports/layer2_reconciliation/player_impact_variant_best_league_breakdown_step3.json`
    - `artifacts/reports/layer2_reconciliation/player_impact_variant_best_league_breakdown_step3.md`
  - Decision: no threshold combination is decision-grade yet.
- [x] Keep player-impact block conditional in serving until gates pass.
  - Required to promote globally:
    - football logic gate `pass`
    - predictive gate above minimum meaningful lift
    - operational upcoming availability coverage above threshold.
  - Confirmatory run (no promotion applied):
    - `artifacts/reports/layer2_reconciliation/player_impact_assumption_validation_step6_confirmatory.md`
    - status unchanged: `correctness=pass`, `football_logic=warn`, `predictive_value=warn`, `operational_readiness=fail`, `overall=fail`

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
- [x] Run feature health audit and archive outputs.
- [x] Run leakage audit and archive outputs.
- [x] Run monitor report for drift/coverage/same-kickoff groups.
- [ ] Retrain only when coverage gates improve enough to make re-train informative.

Latest validation evidence (2026-02-26, run `20260226_212123`):
- Feature health: `python src/modeling/layer2_situational/audit_feature_health.py`
  - `artifacts/reports/layer2_feature_health/20260226_212123/feature_health.json`
  - `odds_model_gap_*` non-missing: `89.2%` (`6987/7832`)
  - binary prevalence check: `home_lame_duck`/`away_lame_duck` remain zero-variance.
- Leakage audit: `python src/db/audit_layer2_situational_leakage.py --days 14 --limit 500`
  - report: `artifacts/reports/leakage_audit/leakage_audit_20260226_212123.json`
  - availability timing check: `pass` (0 violations)
  - snapshot sample check: `pass` (0 violations)
  - lambda timing check: `info` (`late_rows=0`)
- Monitoring: `python src/modeling/layer2_situational/monitor_layer2_signals.py --days 14`
  - `artifacts/reports/monitoring/20260226_212123/layer2_monitor_2026-02-26.json`
  - drift check: `drift_flag=false` for odds gap
  - upcoming availability coverage: `8.14%` (`50/614`) -> improved, but still below production-weighting target.

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
- [x] Retrain Layer 2 situational residual model and refresh deployment artifacts.
  - Script: `src/modeling/layer2_situational/train_situational_residual.py --train`
  - Expected outputs:
    - `model_artifacts/situational_model/situational_model.pkl`
    - `model_artifacts/situational_model/layer2_deployment_policy.json`
    - `model_artifacts/situational_model/situational_model.meta.json`

- [x] Run ablation table and compare stage lifts.
  - Script: `src/modeling/layer2_situational/run_ablation_table.py`
  - Stages to keep:
    1. baseline (Layer 1 only)
    2. + standings/points
    3. + schedule
    4. + player impact
    5. + odds gap

- [x] Re-run calibration diagnostics on prediction outputs.
  - Script: `src/modeling/evaluation/calibration_check.py`
  - Outputs: ECE + reliability diagrams.
  - Note: latest run had only 10 scored rows per market, so all market-level calibration bins were skipped by min-sample gate.

Latest evidence refresh (2026-02-26):
- Retrain run:
  - `python src/modeling/layer2_situational/train_situational_residual.py --train`
  - artifacts refreshed in `model_artifacts/situational_model/`
- Ablation run:
  - `python src/modeling/layer2_situational/run_ablation_table.py`
  - `artifacts/reports/ablation/ablation_results.md`
  - stage readout:
    - `standings_gaps` alone: slight negative vs baseline
    - `+schedule`: near-flat home, slight away improvement
    - `+player`: negative on both home and away
    - `+odds`: recovers strong away lift; home remains slightly negative in this staged path
- Calibration run:
  - `python src/modeling/evaluation/calibration_check.py`
  - found `220` scored predictions total, but only `10` per market
  - all markets skipped by min-sample gate, output remained empty (`data/v1/calibration/calibration_results.json`)

Provisional feature-block decision from this cycle:
- Keep: odds gap block (`odds_model_gap_*`, `odds_opening_gap_*`) as primary Layer 2 lift source.
- Keep (monitor): schedule block (`rest_delta`, `congestion_flag`, `*_upcoming_tier`) with modest contribution.
- Rework/defer: player-availability impact block (`home_xg_lost`, `away_xg_lost`, `home_key_absent`, `away_key_absent`, `injury_impact`) before production weighting.

Post-repair controlled comparison (current reference):
- Comparison report:
  - `artifacts/reports/layer2_reconciliation/layer2_postrepair_full_vs_keyabs_comparison.json`
  - `artifacts/reports/layer2_reconciliation/layer2_postrepair_full_vs_keyabs_comparison.md`
- Baseline full-feature model (`model_artifacts/situational_model/`):
  - home/away RMSE: `1.1910 / 1.0783`
  - enabled leagues: `E1`, `RO1`
- Candidate keyabs-only model (`model_artifacts/situational_model_keyabs_only/`):
  - home/away RMSE: `1.1924 / 1.0762`
  - enabled leagues: `RO1`
- Decision frame:
  - full model is preferred if we prioritize home-side accuracy and broader league enablement.
  - keyabs-only is preferred only if we prioritize away-side RMSE improvement and accept losing `E1`.

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
