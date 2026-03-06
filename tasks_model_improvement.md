# Model Improvement Sprint Tasks
Date: 2026-03-05
Owner: Modeling
Goal: Improve predictive quality (AUC/Brier) before betting-policy changes.

## Strategy Decision (Honest Answer)
1. Is broad feature churn the best way? `No`.
2. Is the next approach (path-aware structural modeling + segment-aware calibration + missingness audit) the best validated next move? `Partially`.
3. Why:
   1. Direct classifier benchmark for `h_1up/a_1up` underperformed current anytime head.
   2. Simple isotonic / league-logistic recalibration for `h_1up/a_1up` also underperformed.
   3. Pure phase-split anytime (`Path V3`) also underperformed refreshed baseline.
   4. The validated anytime bottleneck was underused DB state, not lack of complexity: richer snapshot attack/defense features lifted the family.

## Current State Snapshot
1. [x] Retrain all v2 families and produce before/after compare report.
2. [x] Confirm scoreline improved, corners mixed, anytime mixed.
3. [x] Confirm edge-bucket report coverage and weak buckets.

## Active Todo List
1. [x] Add automatic model-type selection (`histgb_poisson` vs `poisson_glm`) for scoreline.
2. [x] Add automatic model-type selection (`histgb_poisson` vs `poisson_glm`) for corners.
3. [x] Add automatic model-type selection (`histgb_poisson` vs `poisson_glm`) for anytime.
4. [x] Add focused unit tests for model-selection scoring helpers.
5. [x] Retrain v2 families using `--model-type auto`.
6. [x] Produce updated compare report and summarize metric movement by family.

## Acceptance
1. Family trainers support `--model-type auto`.
2. Training reports include requested vs selected model type and selection summary.
3. Tests pass for new helper logic.
4. Compare report updated with new run artifacts.

## Next Cycle Todos (2026-03-05)
1. [x] Set `--model-type auto` as default for v2 family trainers.
2. [x] Rebuild `metrics_baseline_v2.json` from latest holdout artifacts and validate coverage.
3. [x] Run corners feature ablation and propose feature pruning changes.
4. [x] Run anytime feature ablation and propose feature/target redesign priorities.
5. [x] Apply safe contract-level feature pruning based on ablation evidence and retrain.
6. [x] Publish updated before/after compare report after pruning pass.

## Active Todos (Current)
1. [x] Build reusable edge-bucket override recommender from report artifacts.
2. [x] Run recommender on latest risk-edge report and apply any strict improvements.
3. [x] Re-run risk assessment dry-run and confirm action mix shift for affected market buckets. (no policy change detected; action mix unchanged in this cycle)

## AUC Lift Sprint (Now)
1. [x] Build scored-prediction diagnostic slice report (market, league, odds band, kickoff horizon, edge band).
2. [x] Run diagnostic on latest 180-day window and identify bottom-decile AUC segments with sufficient sample.
3. [x] Define candidate feature blocks for weak segments and map each to existing DB fields.
4. [x] Run first feature-block experiment (small, high-signal set) and produce before/after deltas.

## Next Execution Queue
1. [x] Run Experiment 2: scoreline draw/variance feature block (`1x2_d`, `dc_12` recovery focus). (rejected: focus markets worsened)
2. [x] Run Experiment 3: anytime momentum-path feature block (`h_1up`, `a_1up` recovery focus). (near-flat lift; hold)
3. [x] Promote only if family-level and weak-segment deltas are both positive (AUC up, Brier down). (no promotion from Exp2/Exp3)

## Next Execution Queue (Revised)
1. [x] Build anytime `Path V3` (phase-split rates) for `h_1up/a_1up/h_2up/a_2up`. (rejected: weaker than refreshed baseline)
2. [ ] Design draw-specific model branch/calibrator for `1x2_d` + `dc_12` rather than broad scoreline feature expansion.
3. [ ] Build league-segmented calibration pass for weak leagues (`BR1`, `JP1`, `SC2`, `SC3`).
4. [ ] Audit and improve scored-row odds/edge availability for corners and anytime markets to reduce `missing` slice dominance.

## Immediate Next Tasks (Execution Order)
1. [x] Create `tasks_anytime_path_v3.md` and run benchmark baseline snapshot (`anytime_reality_check_20260305.md`).
2. [x] Implement phase-split anytime pricer and training/eval entrypoint.
3. [x] Compare Path V3 vs current anytime on weak segments and overall family metrics. (failed promotion)
4. [x] Promote only if both family-level and weak-segment deltas are positive. (no promotion for Path V3)

## Anytime Recovery (Current Winner)
1. [x] Audit richer live-safe DB snapshot features already present in the dataset.
2. [x] Build enriched anytime contract using `xgot`, `xa`, `box_touches`, `big_chances`, `crosses`, `goals_prevented`, and possession features.
3. [x] Add per-side Poisson regularization controls for anytime training.
4. [x] Retrain anytime with enriched contract and compare against refreshed baseline.
5. [x] Promote enriched anytime candidate (`anytime_rich_snapshot_v1`) on aggregate lift and refresh `model_artifacts/v2/anytime`.

## Current Next Queue
1. [ ] Diagnose and recover `a_1up` AUC specifically; it is the only anytime market that did not improve on AUC. Initial weak-league diagnostic points to `RO1`, `SP1`, `P1`, `D1`, and `SP2`.
2. [ ] Design draw-specific model branch/calibrator for `1x2_d` + `dc_12`.
3. [ ] Build league-segmented calibration pass for weak leagues (`BR1`, `JP1`, `SC2`, `SC3`).
4. [ ] Audit and improve scored-row odds/edge availability for corners and anytime markets to reduce `missing` slice dominance.

## Strategy Pivot (2026-03-06)
1. [x] Confirm existing accumulator backtester already supports walk-forward ticket construction, bankroll paths, and correlation controls.
2. [x] Decide not to build a new betting engine from scratch; build a strategy layer on top of existing backtester instead.
3. [x] Define first small-bankroll accumulator universe and policy artifacts.
4. [x] Run the first `100 cedis` small-bankroll campaign backtest and milestone simulation.
5. [x] Use campaign evidence to decide whether remaining model fixes (`a_1up`, draw branch, weak-league calibration) are trading-critical or can stay deferred.

## Betting-Driven Reprioritization
1. [x] Identify real trading blocker: usable scored odds coverage is effectively limited to `o15` in the current backtestable universe.
2. [x] Confirm first working campaign uses `o15` doubles only; broader corners/anytime accas are not yet backtestable with current odds coverage.
3. [ ] Prioritize odds coverage expansion for corners, anytime, and non-draw scoreline legs before additional feature churn on those markets.
4. [ ] Revisit `a_1up` recovery only after its odds coverage makes it relevant to the betting layer.

## Runtime Repair Sprint (2026-03-06)
1. [x] Make `C:\Developer\soccer\footy-model` the canonical runtime source for WSL schedulers.
2. [x] Repoint WSL scheduler wrappers to execute the canonical repo instead of the stale WSL repo copy.
3. [x] Fix score selection so scoreable rows are processed before rows missing corners/incidents.
4. [x] Make settled scoring repair rows with existing `prediction_scores` but null odds.
5. [x] Re-run scoring/backfill, then refresh odds-backed coverage and accumulator backtest results.

## Runtime Repair Outcome
1. [x] WSL schedulers now execute the canonical Windows repo using the WSL venv.
2. [x] Canonical `tick_due_fixtures_v1.py` now supports `--skip-settle/--skip-predict/--skip-score`, matching live scheduler usage.
3. [x] Manual 365-day score repair pass completed successfully with zero skip waste across leagues.
4. [x] Historical scored odds coverage expanded materially:
   1. `1x2_h/d/a`: ~4.9k odds-backed rows each
   2. `dc_1x/x2/12`: ~4.8k odds-backed rows each
   3. `o15`: ~7.1k odds-backed rows
   4. `c85`: 551 odds-backed rows
5. [x] Default small-bankroll tradable core is now `dc_1x`, `dc_x2`, `o15` instead of `o15` only.
