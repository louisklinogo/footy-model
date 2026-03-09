# Current Repo State (Canonical)

Updated: 2026-03-09
Owner: Augment Agent
Status: Canonical current-state summary

## What is live right now

- Active scheduler wrapper: `scripts/run_tick_scheduler.cmd`
- Active job entrypoint: `src/jobs/tick_due_fixtures_v1.py`
- Active predict path:
  1. `src/features/build_team_premium_snapshots_v1.py`
  2. `src/modeling/layer1_poisson/predict_lambda.py`
  3. `src/modeling/layer2_situational/predict_situational_residual.py`
  4. `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py`
  5. `src/modeling/evaluation/assess_prediction_risk.py`
  6. `src/modeling/export/export_market_outcomes_fixtures_first.py`

## Active model identities

- Final live market model:
  - `model_name = market_outcome_gbm`
  - `model_version = fixtures_first_prematch_v1`
- Supporting live upstream models:
  - `lambda_xgb / v1`
  - `situational_xgb / v2`

These are the identities currently seen in the live DB tables (`predictions`, `prediction_risk_assessments`, `pipeline_runs`) at material volume.

## What v2 is today

- `src/modeling/v2/*` is a real challenger/evaluation stack.
- Family predictors exist for:
  - `scoreline_v2`
  - `corners_v2`
  - `anytime_v2`
- Current repo evidence indicates v2 family predictions are being run manually / experimentally, not through the active scheduled prediction path.
- Current leading scoreline challenger reference is:
  - artifact: `model_artifacts/v2/scoreline_v21_total_intensity_snap_20260308/`
  - passing evaluation: `model_artifacts/v2/evaluation_scoreline_v21_total_intensity_snap_20260308_eps/`
  - pinned named baseline snapshot: `model_artifacts/v2/baselines/metrics_baseline_v2_scoreline_v21_total_intensity_snap_20260308_eps.json`
- This does **not** mean v2 is the active live runtime yet; it means this is the current repo-backed scoreline challenger reference for future v2 comparisons.
- Current leading anytime challenger reference is:
  - artifact: `model_artifacts/v2/anytime_direct_monotone_v1_candidate_20260308/`
  - evaluation: `model_artifacts/v2/evaluation_anytime_direct_monotone_v1_candidate_20260308/`
  - live-overlap comparison: `artifacts/v2/family_replacement/live_replacement_20260308_anytime_direct_monotone_v1/`
- Anytime status nuance: the earlier `anytime_h1up_rich_snapshot_v1_candidate_20260308` contract-only challenger was the best bounded model, and the full `state_ladder` research branch (`v1` through `v5`) was scientifically useful but repeatedly lost on matched live-overlap, especially on home ladders. The first `direct_monotone` candidate changes that: `anytime_direct_monotone_v1_candidate_20260308` keeps the leader-style constant-goals backbone, adds direct monotone heads with chrono OOF blending, improves `h_1up` and `a_1up`, and preserves `h_2up` / `a_2up` while also improving the aggregate anytime matched live-overlap benchmark versus the prior leader.
- Current corners status: the bounded `corners_attack_block_candidate_20260308` rerun is recorded as a stop signal, not a promotion path. Corners should be revisited later as a deeper totals-first structural sprint.
- Current repo-backed bundled challenger reference is:
  - scoreline artifact: `model_artifacts/v2/scoreline_v21_total_intensity_snap_20260308/`
  - corners artifact: `model_artifacts/v2/live_verification_corners_20260306/`
  - anytime artifact: `model_artifacts/v2/anytime_direct_monotone_v1_candidate_20260308/`
  - bundled evaluation: `model_artifacts/v2/evaluation_bundle_refresh_anytime_direct_monotone_v1_20260309/`
  - bundled live-overlap comparison: `artifacts/v2/family_replacement/live_replacement_20260309_bundle_anytime_direct_monotone_v1/`
- Bundle status nuance: this is still a challenger bundle, not the active live runtime. It passes the current required scoreline-core promotion gate (`22/22` required markets passed, `75/76` scoped markets passed) and modestly improves the matched full-bundle overlap versus the prior anytime-led bundle; the gain comes from the new anytime family while scoreline and corners remain unchanged.

## Canonical source-of-truth hierarchy

1. Live DB + active entrypoint code
   - `pipeline_runs`
   - `predictions`
   - `prediction_risk_assessments`
   - `src/jobs/tick_due_fixtures_v1.py`
   - `scripts/run_tick_scheduler.cmd`
2. Live runtime docs
   - `docs/current_state.md`
   - `docs/ops_due_fixtures_job_v1.md`
3. V2 challenger docs
   - `docs/v2_upgrade_tracker.md`
   - `docs/v2_evaluation_workflow.md`

## Non-canonical docs

The following have been moved to `docs/archive/` and should be treated as planning/history unless explicitly refreshed:

- `docs/archive/progress.md`
- `docs/archive/ROADMAP.md`
- `docs/archive/todo.md`
- `docs/archive/tasks_v2.md`
- `docs/archive/tasks_anytime_path_v3.md`
- `docs/archive/SYS.MD`
- `docs/archive/plan.md`
- `docs/archive/plan-v2-calibration-stage.md`

## Serving decision rule

Do not treat “v2” as one atomic production switch. Promote family-by-family only when:

- it beats the current live path on the relevant markets,
- it is wired into risk/export/runtime safely,
- and it has a clear rollback path.