# DB-First SofaScore Expansion Plan

Status owner: Codex  
Source of truth: live Postgres tables plus active code paths

## Goal

Improve the model stack by reusing already-populated SofaScore-derived DB surfaces first, before adding new ingestion/storage.

## Rollout Order

1. Anytime
2. Scoreline
3. Corners
4. Sparse-field parser repair / backfill
5. New tables only if DB-reuse plateaus

## Phase 0: DB Audit And Backfill Gate

- [x] Add a live SQL audit report generator
- [x] Add a feature eligibility registry with statuses:
  - `ready_now`
  - `ready_after_backfill`
  - `missing_requires_new_table`
  - `diagnostics_only`
- [x] Add sparse premium field diagnosis for:
  - `ball_recoveries`
  - `yellow_cards`
  - `red_cards`
- [x] Generate first audit artifacts from the live DB

Artifacts:
- `artifacts/reports/db_reuse_surface/20260315T084020Z/db_audit_report.json`
- `artifacts/reports/db_reuse_surface/20260315T084020Z/feature_eligibility_registry.json`
- `artifacts/reports/db_reuse_surface/20260315T084020Z/sparse_premium_field_diagnosis.json`

Observed DB diagnosis:
- `yellow_cards`: `ready_now` after bounded backfill
- `red_cards`: `ready_now` after bounded backfill
- `ball_recoveries`: `missing_requires_new_table` under current stored payload coverage

## Phase 1: Anytime Wave

- [x] Add DB-reuse feature layer for:
  - missing-player market value burden
  - positional missing-value burden
  - top-1 / top-2 missing player value
  - missing-value share of listed squad value
  - lineup completeness / known-share context
  - attack concentration from prior player xG / xA history
  - bench attack fallback context
- [x] Integrate those features into the anytime family builder
- [x] Add experimental anytime contract:
  - `model_v2/feature_contracts/experiments/anytime_dbreuse_player_context_v1.yaml`
- [x] Emit feature coverage snapshots from anytime training
- [x] Add focused tests
- [x] Run smoke anytime training with the new contract
- [x] Train real anytime challenger artifact:
  - `model_artifacts/v2/anytime_dbreuse_player_context_v1_candidate_20260315/`
- [x] Run offline challenger evaluation artifacts:
  - `model_artifacts/v2/evaluation_anytime_dbreuse_player_context_v1_candidate_20260315/`
- [x] Run matched live replacement compare:
  - `artifacts/v2/family_replacement/live_replacement_20260315_anytime_dbreuse_player_context_v1/`
- [ ] Advance Wave 1 to new anytime leader

Implementation files:
- `src/modeling/v2/db_reuse_features.py`
- `src/modeling/v2/families/anytime/features.py`
- `src/modeling/v2/families/anytime/train_anytime.py`
- `model_v2/feature_contracts/experiments/anytime_dbreuse_player_context_v1.yaml`

Verification:
- Focused tests passed
- Smoke artifact written to:
  - `artifacts/tmp/anytime_dbreuse_player_context_smoke/`

Observed Wave 1 result:
- Offline family gate did not beat the current anytime leader cleanly.
- Holdout comparison versus `anytime_direct_monotone_v1_candidate_20260308` stayed flat-to-worse on all four markets:
  - `a_1up` AUC `0.5987` vs `0.6008`
  - `a_2up` AUC `0.6188` vs `0.6255`
  - `h_1up` AUC `0.6086` vs `0.6111`
  - `h_2up` AUC `0.6081` vs `0.6101`
- Walk-forward improved slightly on `a_1up` / `a_2up`, stayed flat on `h_1up`, and regressed on `h_2up`.
- Matched live replacement versus the live stack was positive on the anytime slice:
  - `ΔAUC +0.00240`
  - `ΔBrier -0.00094`
  - `ΔLogLoss -0.00330`
- Per-market live result:
  - `a_1up`: positive
  - `a_2up`: positive
  - `h_1up`: better Brier/log loss, essentially flat AUC
  - `h_2up`: negative
- Decision: useful no-go iteration, not the new anytime leader under the rollout policy because offline and live did not both clear cleanly.
- Important evaluation nuance: the full bundle promotion output for this run is not the decision source for Phase 1 because the currently checked-out scoreline artifact has drifted from the frozen named baseline snapshot. The Wave 1 decision should be read from the anytime family comparisons above, not the top-level bundle promotion headline.

## Phase 2: Scoreline Wave

- [x] Add standings + player-context feature hooks into shared DB-reuse layer
- [x] Add experimental scoreline contract:
  - `model_v2/feature_contracts/experiments/scoreline_dbreuse_team_player_context_v1.yaml`
- [x] Add contract-selection test coverage
- [x] Integrate the full scoreline challenger training/evaluation run
- [x] Confirm live challenger behavior against the current scoreline leader
- [ ] Advance Wave 2 to new scoreline leader

Implementation files:
- `src/modeling/v2/db_reuse_features.py`
- `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py`
- `model_v2/feature_contracts/experiments/scoreline_dbreuse_team_player_context_v1.yaml`

Artifacts:
- `model_artifacts/v2/scoreline_dbreuse_team_player_context_v1_candidate_20260315/`
- `model_artifacts/v2/evaluation_scoreline_dbreuse_team_player_context_v1_candidate_20260315/`
- `artifacts/v2/family_replacement/live_replacement_20260315_scoreline_dbreuse_team_player_context_v1/`

Observed Wave 2 result:
- Offline family gate failed decisively versus the frozen scoreline leader `scoreline_v21_total_intensity_snap_20260308`.
- Formal evaluation against the pinned named baseline stayed at:
  - `29 / 76` scoped markets passed
  - `2 / 22` required markets passed
- The only required markets that passed were:
  - `o15`
  - `u35`
- Directional/core scoreline markets remained broadly negative in the offline gate:
  - `1x2_*`
  - `dc_*`
  - `ah2_*`
  - `eh3_*`
- Matched live replacement versus the live stack was mildly positive:
  - full overlap:
    - `delta_auc +0.00105`
    - `delta_brier -0.00047`
    - `delta_log_loss -0.00103`
  - scoreline directional overlap:
    - `delta_auc +0.00113`
    - `delta_brier -0.00045`
    - `delta_log_loss -0.00070`
- Important nuance:
  - the live directional slice is only partly true scoreline-vs-scoreline because about `70%` of `scoreline_directional_overlap` rows are still fallback rows in the live replacement harness for this cohort.
  - the nonfallback directional slice was stronger, but the offline gate remains the controlling evidence under the rollout policy.
- Decision: useful no-go iteration, not the new scoreline leader under the rollout policy because live improvement did not transfer into the formal offline gate against the pinned leader.

## Phase 3: Corners Wave

- [ ] Add premium territory/contact/transition extras to the corners feature surface
- [ ] Add explicit ablation blocks for those extras
- [ ] Train challenger against the current best corners research leader
- [ ] Validate targeted seam:
  - `c95`
  - `c105`
  - `ac25`
  - `ac35`
  - `ac55`

## Phase 4: Parser Repair / Backfill

- [x] Confirm yellow/red cards are present in stored raw payloads
- [x] Patch parser maps for yellow/red cards
- [x] Add schema columns for:
  - `h_yellow_cards`
  - `a_yellow_cards`
  - `h_red_cards`
  - `a_red_cards`
- [x] Apply schema bootstrap to live DB
- [x] Extend offline backfill selector to include missing card fields
- [x] Run bounded historical backfill for card fields
- [x] Re-run DB audit after backfill

Implementation files:
- `src/ingest/ingest_sofascore_stats.py`
- `src/ingest/offline_stats_backfill.py`
- `src/ingest/bootstrap_fixtures_schema_v1.py`

Artifacts:
- `artifacts/reports/db_reuse_surface/20260315T141011Z/db_audit_report.json`
- `artifacts/reports/db_reuse_surface/20260315T141011Z/feature_eligibility_registry.json`
- `artifacts/reports/db_reuse_surface/20260315T141011Z/sparse_premium_field_diagnosis.json`

Observed Phase 4 result:
- The bounded offline backfill updated `3,712` `fixture_stats_premium` rows.
- Card coverage moved materially:
  - yellow cards: `55 -> 3,640` fully populated home/away rows
  - red cards: `12 -> 1,183` fully populated home/away rows
- `yellow_cards_surface` now audits as:
  - `ready_now`
  - coverage ratio `0.3353`
- `red_cards_surface` now audits as:
  - `ready_now`
  - coverage ratio `0.1090`
- `ball_recoveries_surface` stayed:
  - `missing_requires_new_table`
  - diagnosis `payload_absent_likely`
- Remaining selector-eligible rows fell from `8,231` to `7,750`, which is consistent with the script only backfilling rows whose stored raw payloads actually contain the mapped fields.
- Decision: Phase 4 is complete for cards. Yellow/red cards are now legitimate ready-now premium surfaces; ball recoveries is still not.

## Phase 5: New Table Expansion

- [x] Write Phase 5 design doc:
  - `docs/plans/2026-03-15-external-context-phase5-design.md`
- [x] Add new external-context tables:
  - `team_external_context`
  - `match_external_context`
  - `player_external_context`
- [x] Align bootstrap with the live provider-ID columns needed by the new context layer
- [x] Add external-context ingestion scaffolding:
  - `src/ingest/ingest_external_context.py`
- [x] Add focused tests for normalization and schema presence
- [x] Verify schema bootstrap and dry-run ingestion smoke pass
- [x] Run non-dry external-context ingestion at useful coverage
- [x] Audit live row coverage / freshness for the new tables
- [x] Decide which context types become model-eligible first

Implementation files:
- `src/ingest/bootstrap_fixtures_schema_v1.py`
- `src/ingest/ingest_external_context.py`
- `tests/test_ingest_external_context.py`
- `tests/test_external_context_schema.py`

Observed Phase 5 initial result:
- All three new tables are now part of the bootstrap schema.
- The bootstrap was also aligned with the live DB's provider-ID columns and `players` table so the new context layer has a consistent base schema.
- Focused validation passed:
  - `7 passed`
- Dry-run smoke pass succeeded:
  - `python src/ingest/ingest_external_context.py --context-type all --limit 1 --dry-run`
- The dry run proved all three context families can fetch and normalize at least one row path, with expected provider variability:
  - `team_overview`, `league_stats`, `team_streaks`, `h2h_results`, `player_overview`, `attributes`, and `league_stats` all normalized successfully
  - some optional endpoints returned `404` on the sample entity (`performance_graph`, `pre_match_form`, `win_probability`), which is acceptable and already handled as sparse provider availability rather than a hard failure
- Real bounded ingestion now also completed:
  - `python src/ingest/ingest_external_context.py --context-type all --limit 5`
- Post-ingestion audit written to:
  - `artifacts/reports/db_reuse_surface/20260315T203159Z/external_context_audit.json`
- First persisted coverage snapshot:
  - `team_external_context`: `18` rows, `5` context-linked teams, `5` context types, `0` null canonical IDs
  - `match_external_context`: `12` rows, `5` context-linked fixtures, `2` context types, `0` null canonical IDs
  - `player_external_context`: `18` rows, `5` context-linked players, `3` context types, `0` null canonical IDs
- First eligible context types based on actual persisted rows:
  - `team_external_context`: `standings_total`, `standings_home`, `standings_away`, `team_overview`, `league_stats`
  - `match_external_context`: `team_streaks`, `h2h_results`
  - `player_external_context`: `player_overview`, `attributes`, `league_stats`
- First non-eligible context types based on real endpoint behavior:
  - `performance_graph`
  - `pre_match_form`
  - `win_probability`
  - These returned `404` in the bounded live ingestion pass and should stay outside model planning until provider availability is demonstrated at useful coverage.
- Decision: Phase 5 initial implementation is complete. The next Phase 5 step, if we continue, is a broader coverage pass or targeted scheduler wiring for the context types that actually persist, not more schema work.
- Full external-context audit update after broad backfill + recurring WSL collection:
  - `artifacts/reports/db_reuse_surface/20260316T014439Z/db_audit_report.json`
  - `artifacts/reports/db_reuse_surface/20260316T014439Z/feature_eligibility_registry.json`
- Audited model-eligible external context surfaces are now all `ready_now`:
  - `external_team_standings`
  - `external_team_overview`
  - `external_team_league_stats`
  - `external_team_performance_graph`
  - `external_match_pre_match_form`
  - `external_match_team_streaks`
  - `external_match_h2h_results`
  - `external_player_overview`
  - `external_player_attributes`
  - `external_player_league_stats`
- First external-context scoreline wave is now implemented in code as a narrow team+match challenger path:
  - shared PIT-safe external joins added to `src/modeling/v2/db_reuse_features.py`
  - new experiment contract:
    - `model_v2/feature_contracts/experiments/scoreline_external_context_v1.yaml`
  - scoreline train/predict paths now support the external-context feature block without changing market outputs
  - focused validation passed:
    - `20 passed`
  - smoke artifacts written:
    - `artifacts/tmp/scoreline_external_context_v1_smoke/`
- Full challenger artifacts now complete:
  - candidate artifact:
    - `model_artifacts/v2/scoreline_external_context_v1_candidate_20260316/`
  - evaluation:
    - `model_artifacts/v2/evaluation_scoreline_external_context_v1_candidate_20260316/`
  - live replacement:
    - `artifacts/v2/family_replacement/live_replacement_20260316_scoreline_external_context_v1/`
- Observed external-context scoreline result:
  - formal policy-backed recommendation:
    - `promote`
    - scope: `required_markets_only`
  - required/core markets:
    - `22 / 22` passed
  - full scope:
    - `59 / 76` passed
  - failed follow-up markets are concentrated in:
    - non-core multigoals / multiscore tails
    - anytime ladders (`h_1up`, `a_1up`, `h_2up`, `a_2up`) included in the combined family evaluation surface
  - matched live replacement versus the current repo-backed bundle was strongly positive:
    - full overlap:
      - `delta_auc +0.02075`
      - `delta_brier -0.00874`
      - `delta_log_loss -0.02280`
    - scoreline directional overlap:
      - `delta_auc +0.03321`
      - `delta_brier -0.01506`
      - `delta_log_loss -0.03912`
    - scoreline directional nonfallback overlap:
      - `delta_auc +0.04808`
      - `delta_brier -0.01856`
      - `delta_log_loss -0.04576`
- Decision status:
  - implementation complete
  - evaluation complete
  - this is a live positive scoreline challenger and a policy-backed `required_markets_only` promotion candidate
  - `external_player_league_stats`
- Audited live coverage at that point:
  - team standings canonical entities: `527`
  - team overview canonical entities: `559`
  - team league stats canonical entities: `559`
  - team performance graph canonical entities: `418`
  - match pre-match form canonical entities: `2752`
  - match team streaks canonical entities: `5000`
  - match h2h canonical entities: `5000`
  - player overview canonical entities: `10000`
  - player attributes canonical entities: `7630`
  - player league stats canonical entities: `9017`
- Practical implication: Phase 5 has moved from “schema/scaffolding” to “real reusable model surface”. The next justified modeling step is a first external-context challenger, starting with scoreline.

- Scoreline directional-shrink follow-up (2026-03-17):
  - implemented `one_sided_anchor` as a bounded post-holdout calibration candidate for the home-leaning directional seam
  - scope was intentionally narrow:
    - `1x2_h`
    - `dc_x2`
    - `ah2_home_m05`
    - `ah2_away_p05`
    - `ah2_home_m15`
    - `ah2_away_p15`
    - `eh3_0_1_home`
    - `eh3_0_1_away`
  - focused tests added:
    - `tests/v2/test_calibration_methods.py`
  - variant artifact:
    - `model_artifacts/v2/scoreline_external_context_directional_shrink_v1_candidate_20260317/`
  - evaluation:
    - `model_artifacts/v2/evaluation_scoreline_external_context_directional_shrink_v1_candidate_20260317/`
  - observed result:
    - useful no-go
    - targeted shrink did not improve the intended seam on the current scoreline leader
    - resulting calibrator set matched the existing `scoreline_external_context_v1_candidate_20260316` directional seam outcome
    - promotion registry stayed unchanged at `59 / 76` scoped markets passed
  - practical implication:
    - the remaining scoreline log-loss issue is not fixed by a simple one-sided calibration shrink
    - the next justified scoreline follow-up, if we continue, needs a stronger directional residual hypothesis

## Acceptance Policy

A wave is only considered complete when:

- offline challenger evaluation is run
- live confirmation is run where applicable
- the new block improves probability quality, not just hit rate

## Current State Summary

- Phase 0: complete
- Phase 1 Anytime: implemented, fully evaluated, not advanced
- Phase 2 Scoreline: implemented, fully evaluated, not advanced
- Phase 3 Corners: pending
- Phase 4 Parser repair: complete for cards
- Phase 5 New tables: implemented and broadly populated; first model wave can start
