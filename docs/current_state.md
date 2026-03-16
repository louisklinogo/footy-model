# Current Repo State (Canonical)

Updated: 2026-03-15
Owner: Augment Agent
Status: Canonical current-state summary

## What is live right now

- Active scheduler wrapper: `scripts/run_tick_scheduler.cmd`
- Active WSL auxiliary scheduler wrappers:
  - `scripts/run_predict_scheduler_hybrid.sh`
  - `scripts/run_score_scheduler_hybrid.sh`
  - `scripts/run_external_context_scheduler.sh`
- Active job entrypoint: `src/jobs/tick_due_fixtures_v1.py`
- Active settle/runtime repair behavior:
  - the settle phase now includes a bounded post-match repair pass for recent `ft` fixtures that are still missing `fixture_stats_premium` corners stats or `fixture_incident_lead_states`
  - this repair path reruns SofaScore stats/incidents for targeted fixture IDs and then rebuilds incident lead-state features before the normal score phase
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

## Recent runtime repair note

- Hybrid scoring is now fully scoreable on finished fixtures when upstream post-match data exists.
- `score_market_outcomes_fixtures_first.py` now supports explicit `--model` / `--version` selection, which is how `market_outcome_v2 / hybrid_v1` is backfilled and scored.
- Scoreline family exotics (`mg_*`, `hmg_*`, `amg_*`, `ms_*`) are now part of the scorer's supported market set.
- External context collection is now also wired into WSL as a separate hourly auxiliary job:
  - script: `scripts/run_external_context_scheduler.sh`
  - current cron line: `40 * * * * /home/louis/developer/footy-model/scripts/run_external_context_scheduler.sh >> /home/louis/developer/footy-model/artifacts/logs/cron_external_context.log 2>&1`
  - current operational scope: scheduled/upcoming fixtures only, with a `0h` to `72h` horizon
  - current persisted context types from the first real load:
    - team: `standings_total`, `standings_home`, `standings_away`, `team_overview`, `league_stats`
    - match: `team_streaks`, `h2h_results`
    - player: `player_overview`, `attributes`, `league_stats`

## What v2 is today

- `src/modeling/v2/*` is a real challenger/evaluation stack.
- Family predictors exist for:
  - `scoreline_v2`
  - `corners_v2`
  - `anytime_v2`
- Current repo evidence indicates v2 family predictions are being run manually / experimentally, not through the active scheduled prediction path.
- Current leading scoreline challenger reference is:
  - artifact: `model_artifacts/v2/scoreline_external_context_v1_candidate_20260316/`
  - evaluation: `model_artifacts/v2/evaluation_scoreline_external_context_v1_candidate_20260316/`
  - live-overlap comparison: `artifacts/v2/family_replacement/live_replacement_20260316_scoreline_external_context_v1/`
  - pinned named baseline snapshot still used for scoreline promotion decisions:
    - `model_artifacts/v2/baselines/metrics_baseline_v2_scoreline_v21_total_intensity_snap_20260308_eps.json`
- This does **not** mean v2 is the active live runtime yet; it means this is the current repo-backed scoreline challenger reference for future v2 comparisons.
- Current leading anytime challenger reference is:
  - artifact: `model_artifacts/v2/anytime_direct_monotone_v1_candidate_20260308/`
  - evaluation: `model_artifacts/v2/evaluation_anytime_direct_monotone_v1_candidate_20260308/`
  - live-overlap comparison: `artifacts/v2/family_replacement/live_replacement_20260308_anytime_direct_monotone_v1/`
- Anytime status nuance: the earlier `anytime_h1up_rich_snapshot_v1_candidate_20260308` contract-only challenger was the best bounded model, and the full `state_ladder` research branch (`v1` through `v5`) was scientifically useful but repeatedly lost on matched live-overlap, especially on home ladders. The first `direct_monotone` candidate changes that: `anytime_direct_monotone_v1_candidate_20260308` keeps the leader-style constant-goals backbone, adds direct monotone heads with chrono OOF blending, improves `h_1up` and `a_1up`, and preserves `h_2up` / `a_2up` while also improving the aggregate anytime matched live-overlap benchmark versus the prior leader.
- Current leading corners challenger reference is:
  - artifact: `model_artifacts/v2/corners_totals_first_signal_expanded_league_style_v1_candidate_20260310/`
  - evaluation: `model_artifacts/v2/evaluation_corners_totals_first_signal_expanded_league_style_v1_candidate_20260310/`
  - live-overlap comparison: `artifacts/v2/family_replacement/live_replacement_20260310_corners_totals_first_signal_expanded_league_style_v1/`
- Corners status nuance: the 2026-03-10 totals-first structural sprint tested allocation redesigns, direct team-market heads, calibrated totals surfaces, a league-share residual split path, and a coherent PMF surface. None beat live or passed promotion. The current evidence says corners is now signal/support-limited more than architecture-limited under the valid pre-match PIT setup; see `docs/plans/2026-03-10-corners-signal-limit-review.md`.
- Corners next-step plan: the earlier signal-acquisition roadmap remains recorded in `docs/plans/2026-03-10-corners-new-signal-acquisition-plan.md`, but the currently justified bounded follow-ups have now been executed. Treat corners as parked rather than as an active feature-engineering queue.
- Corners Phase A status (2026-03-10): the first contract-only signal acquisition pass was executed and did not beat the current leader. The full bundle (`corners_signal_phasea_v1_candidate_20260310`) regressed materially; the bounded `xgot/xa + big_chances` salvage variant (`corners_signal_phasea_core_v1_candidate_20260310`) improved on the full bundle but still lost to `corners_totals_first_signal_expanded_league_style_v1_candidate_20260310`. If corners work continues, the next justified step is Phase B PIT extraction of defensive-pressure signals, not more small contract reshuffles.
- Corners Phase B status (2026-03-10): the defensive-pressure extraction pass was executed in full and also failed clearly. `corners_signal_phaseb_pressure_v1_candidate_20260310` added `rolling_errors_lead_to_shot` and `rolling_tackles_pct` (plus missingness indicators) through both PIT and live prediction seams, but matched live `corners_overlap` regressed to `ΔAUC -0.017072`, `ΔBrier +0.004888`, `Δlog-loss +0.012008`, materially worse than the current leader. Corners should now be treated as parked / blocked until upstream support materially improves (especially team-corners odds, denser totals-corners odds, stronger style-cluster PIT coverage, or genuinely populated availability/lineup data).
- Corners final bounded closeout (2026-03-10): a DB-audit-driven snapshot-quality pass (`corners_signal_final_snapshot_quality_v1_candidate_20260310`) added `rolling_rest_days`, `fidelity_score`, and compact lead-rate/rest/fidelity derived features through both PIT and live paths. It failed decisively, worsening matched live `corners_overlap` to `ΔAUC -0.01609`, `ΔBrier +0.00459`, `Δlog-loss +0.01161` and failing all `12/12` corners markets. A bounded autoresearch-style repo/DB audit after that found no hidden odds-table rescue or team-corners market seam; the only remaining repo-backed seam is situational context features, which looks incremental at best and not sufficient to justify another corners cycle. Practical conclusion: corners is parked until upstream support materially improves.
- Current repo-backed bundled challenger reference is:
  - scoreline artifact: `model_artifacts/v2/scoreline_external_context_v1_candidate_20260316/`
  - corners artifact: `model_artifacts/v2/live_verification_corners_20260306/`
  - anytime artifact: `model_artifacts/v2/anytime_direct_monotone_v1_candidate_20260308/`
  - bundled evaluation: `model_artifacts/v2/evaluation_scoreline_external_context_v1_candidate_20260316/`
  - bundled live-overlap comparison: `artifacts/v2/family_replacement/live_replacement_20260316_scoreline_external_context_v1/`
- Bundle status nuance: this is still a challenger bundle, not the active live runtime. It passes the current required scoreline-core promotion gate (`22/22` required markets passed, `59/76` scoped markets passed) and materially improves the matched full-bundle overlap versus the current repo-backed bundle baseline, driven primarily by the new scoreline family.

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
