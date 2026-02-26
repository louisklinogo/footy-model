# Layer 2 Reconciliation Source of Truth

Status: active  
Effective date: 2026-02-26  
Owner: In-house Senior Data Science review loop (you + assistant)  
Scope: Layer 2 situational residual model only

## Purpose
This document is the single source of truth for resolving why Layer 2 is underperforming and for deciding what Layer 2 should do that Layer 1 does not do.

While this document is active, it overrides normal task sequencing.

## Core Problem Statement
Layer 2 now shows modest aggregate lift vs baseline, but feature governance and deployment tradeoffs are not fully resolved.

Observed from latest run:
- Full model lift vs Layer 1 baseline:
  - home: `+0.44%` (`1.1963 -> 1.1910`)
  - away: `+0.66%` (`1.0854 -> 1.0783`)
- Controlled `full` vs `keyabs_only` tradeoff:
  - `keyabs_only` improves away RMSE further but worsens home RMSE and narrows enabled leagues.
- Calibration: still sample-limited for market-level decision-grade diagnostics.

## Layer 2 Charter (English, non-technical contract)
Layer 2 exists to make small, point-in-time, context-aware corrections to Layer 1 where Layer 1 has systematic residual error.

Layer 2 must not:
- leak post-kickoff or future information
- rely on sparse/noisy features that degrade holdout performance
- apply globally when only some leagues pass stability gates

Layer 2 must:
- improve out-of-sample residual RMSE vs zero-correction baseline
- remain stable across time splits
- pass per-league governance gates before being enabled

## Stop-The-Line Rule
Until this program is complete, do not proceed with unrelated roadmap tasks.

Allowed work during freeze:
- data audits needed for Layer 2 diagnosis
- feature-level audits
- train/predict code reconciliation
- controlled Layer 2 experiments
- governance and policy verification

Not allowed during freeze:
- new modeling expansion unrelated to Layer 2 diagnosis
- deployment broadening without passing gates
- roadmap items not required for this decision

## Decision Framework (must follow in order)
1. Define Layer 2 target behavior in plain English.
2. Audit Layer 1 residual error anatomy (where and when Layer 1 is wrong).
3. Audit every upstream data source used by Layer 2 features.
4. Audit each feature end-to-end:
   - business/statistical hypothesis
   - exact formula and timing semantics
   - coverage and missingness behavior
   - leakage risk
   - incremental value over baseline
5. Audit implementation parity between train and predict paths.
6. Build reconciliation matrix (`intended` vs `implemented` vs `observed` vs `decision`).
7. Run controlled experiments based on that matrix.
8. Decide final keep/fix/drop/defer set and update policy.

No step may be skipped.

## Required Deliverables
1. Layer 2 English charter (this document, kept current).
2. Layer 1 residual anatomy report (segment-level error map).
3. Data source audit table for all dependencies.
4. Feature audit ledger (all Layer 2 features).
5. Train/predict parity audit report.
6. Reconciliation matrix with signed decisions.
7. Experiment summary table with pass/fail against gates.
8. Final recommendation memo: production policy update or continued freeze.

## Exit Criteria (freeze can end only if all pass)
- Charter accepted and unchanged contradictions resolved.
- All data sources audited with timing and quality sign-off.
- All features classified: keep, fix, drop, or defer.
- Train/predict implementation parity confirmed.
- Controlled rerun shows non-negative aggregate lift and acceptable per-league stability.
- Calibration has adequate sample size for decision-grade diagnostics.
- Deployment policy reflects new evidence and is explicitly approved.

## Working Principles
- Point-in-time correctness beats feature quantity.
- Incremental value beats intuition.
- Per-league governance beats global rollout.
- Reproducible evidence beats one-off wins.
- If evidence conflicts with assumptions, assumptions are changed.

## Immediate Next Sequence
1. Lock this charter as source of truth.
2. Build Layer 1 residual error anatomy report.
3. Start source-by-source and feature-by-feature audit.
4. Reconcile code with charter.
5. Review together before any further roadmap execution.

## Progress Checkpoint (2026-02-26)
- Step 2 executed: Layer 1 residual anatomy audit completed.
- Audit note: `docs/plans/2026-02-26-layer1-residual-anatomy-audit.md`
- Updated residual anatomy state after lineage repair:
  - `6906/6906` latest lambda pairs are now pre-kickoff (`0` post-kickoff latest pairs).
  - latest-pair lineage source is now `feature_asof_utc` for all covered fixtures.
- Step 3 executed: Layer 2 source-by-source data audit completed.
- Source audit note: `docs/plans/2026-02-26-layer2-data-source-audit-review.md`
- Timestamp lineage repair design drafted:
  - `docs/plans/2026-02-26-timestamp-lineage-repair-design.md`
- Current critical sources from Step 3 rerun:
  - `team_rivalries` prevalence utility risk
- High-risk source from Step 3:
  - `fixture_player_stats` (`expected_goals` missingness high)
- Post-Step-3 repair update:
  - historical `player_availability` timing repaired (`rows_after_kickoff = 0` on FT set)
  - evidence: `artifacts/reports/layer2_reconciliation/player_availability_event_timing_repair_20260226T171652Z.md`
  - remaining availability concern is upcoming-fixture population coverage (serving readiness), not historical timing leakage.
- Step 4 started: feature-by-feature ledger and reconciliation matrix built.
  - Review note: `docs/plans/2026-02-26-layer2-feature-reconciliation-review.md`
  - Ledger:
    - `artifacts/reports/layer2_reconciliation/layer2_feature_ledger.json`
    - `artifacts/reports/layer2_reconciliation/layer2_feature_ledger.md`
  - Reconciliation matrix:
    - `artifacts/reports/layer2_reconciliation/layer2_feature_reconciliation_matrix.json`
    - `artifacts/reports/layer2_reconciliation/layer2_feature_reconciliation_matrix.md`
  - Current matrix split (latest refresh):
    - `keep_candidate`: 2
    - `conditional_candidate`: 23
    - `drop_candidate`: 9
    - `fix_lineage_before_judgment`: 2
- Step 5 completed:
  - Migration `013_prediction_availability_lineage.sql` applied to active DB.
  - Writers updated to honor lineage semantics:
    - `src/modeling/layer1_poisson/predict_lambda.py`
    - `src/ingest/ingest_sofascore_availability.py`
  - Leakage/source/residual audits now read lineage-safe timing fields.
  - One-time `lambda_xgb` lineage backfill executed (`feature_asof_utc = kickoff - 60m` where null).
  - Train/predict parity report delivered:
    - `docs/plans/2026-02-26-layer2-train-predict-parity-audit.md`
  - Historical availability timing repair executed with clamp/update script and verified clean.
- Step 6/7 advanced:
  - feature intent contract + implementation audit completed.
  - post-repair controlled comparison completed (`full` vs `keyabs_only`).
- Step 8 drafted:
  - final recommendation memo created:
    - `docs/plans/2026-02-26-layer2-final-recommendation-memo.md`
  - recommendation: keep full-feature candidate as active policy reference; retain freeze until calibration and serving-readiness gates are satisfied.
- Step 9 completed:
  - final feature governance sign-off generated:
    - `artifacts/reports/layer2_reconciliation/layer2_feature_final_signoff.md`
    - split: keep `2`, fix `23`, drop `9`, defer `2`.
- Validation and calibration checkpoint refreshed:
  - feature health rerun completed (`odds_model_gap_*` non-missing `6979/7824`, lame-duck flags still zero-variance).
  - leakage audit rerun passed (`0` availability timing violations, `0` snapshot violations, `0` lambda late rows in sample).
  - monitor rerun: odds drift flag `false`; upcoming availability coverage improved to `50/614` (`8.14%`) in run `20260226_212123`.
  - calibration rerun remains sample-limited (`220` scored predictions total, only `10` per market; all markets skipped).
- Step 10 completed (global candidate pruning test):
  - trained pruned global candidate dropping 9 sign-off `drop` features:
    - `model_artifacts/situational_model_pruned_drop9/`
  - comparison artifact:
    - `artifacts/reports/layer2_reconciliation/layer2_pruned_drop9_vs_full_comparison.md`
  - result:
    - pruned candidate improved away RMSE but materially worsened home RMSE and removed `E1` from enabled leagues.
    - decision: keep full global model as baseline policy candidate.
- Step 11 completed (deterministic rule-layer pilot + segmented backtest):
  - implementation:
    - `src/modeling/layer2_situational/rule_layer.py`
    - `src/modeling/layer2_situational/predict_situational_residual.py` (`--enable-rule-layer`)
    - `src/pipelines/daily_pipeline.py` (`--enable-situational-rule-layer`, `--situational-rule-layer-config`)
  - segmented backtest artifacts:
    - `artifacts/reports/layer2_reconciliation/layer2_rule_layer_segmented_backtest_full.md`
    - `artifacts/reports/layer2_reconciliation/layer2_rule_layer_segmented_backtest_full_home_only_v2.md`
  - selected policy config:
    - `model_artifacts/situational_model/rule_layer_config.json` (`home_only_v1`, initial pilot)
  - result:
    - home-only overrides improved home triggered/global metrics and avoided away global degradation.
- Step 12 completed (three-way policy recommendation):
  - artifact:
    - `artifacts/reports/layer2_reconciliation/layer2_three_way_policy_recommendation.md`
  - recommendation:
    - use full global Layer 2 model.
    - do not promote pruned-drop9 global candidate.
    - enable deterministic rule-layer on top of global Layer 2.
  - freeze remains active pending calibration sample and upcoming-availability serving readiness gates.
- Step 12a completed (key-absence deterministic refinement):
  - key-absence-only sweep executed:
    - `src/modeling/layer2_situational/sweep_rule_layer_key_absent.py`
    - artifact: `artifacts/reports/layer2_reconciliation/layer2_rule_layer_key_absent_sweep_step1_6.md`
  - selected config promoted:
    - `model_artifacts/situational_model/rule_layer_config.json` (`keyabs_home_only_minus0.120`)
  - confirmatory backtest:
    - `artifacts/reports/layer2_reconciliation/layer2_rule_layer_segmented_backtest_full_keyabs_home_minus012_v1.md`
  - result vs prior `home_only_v1`:
    - improved global home lift (`0.20%` vs `0.18%`)
    - improved triggered home lift (`2.45%` vs `1.60%`)
    - global away lift unchanged (`0.06%`)
