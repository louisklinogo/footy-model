# premium_first_data_architecture

core principle: one fixture lifecycle, id-first (`flashscore_id`), no fuzzy joins.

## 1) final table set (from scratch)

### `leagues`
- purpose: league reference data
- key columns: `league_id` (pk), `league_code` (unique), `league_name`, `country`

### `teams`
- purpose: canonical team entities
- key columns: `team_id` (pk), `team_name` (unique), `league_code` (nullable)

### `team_aliases`
- purpose: map provider names to canonical teams
- key columns: `alias_id` (pk), `provider` (`flashscore`), `alias_name`, `team_id`
- unique constraint: (`provider`, `alias_name`)

### `fixtures`
- purpose: one row per match across all statuses
- key columns:
  - `fixture_id` (pk)
  - `flashscore_id` (unique, required)
  - `league_code`, `home_team_id`, `away_team_id`, `match_datetime_utc`
  - `status` (`scheduled`, `live`, `ft`, `postponed`, `cancelled`)
  - `created_at`, `updated_at`

### `fixture_results`
- purpose: final result ledger (separate from fixture metadata)
- key columns:
  - `fixture_id` (pk/fk -> fixtures)
  - `home_goals`, `away_goals`, `result_status`
  - `result_source` (`flashscore`)
  - `settled_at`

### `fixture_odds_snapshots`
- purpose: preserve full odds ladder snapshots over time
- key columns:
  - `snapshot_id` (pk)
  - `fixture_id` (fk)
  - `snapshot_time_utc`
  - `snapshot_type` (`opening`, `latest_pre_match`, `live`, `closing`)
  - `ou_json` (jsonb full ladder)
  - `ah_json` (jsonb full ladder)
  - `one_x_two_json` (jsonb, optional)
- index: (`fixture_id`, `snapshot_time_utc`)

### `fixture_stats_premium`
- purpose: final premium match stats (post-match)
- key columns:
  - `fixture_id` (pk/fk)
  - rich stats: xg, xgot, xa, box_touches, crosses, possession, etc.
  - `fidelity_score`
  - `ingested_at`

### `team_form_snapshots`
- purpose: general rolling pre-match features by team
- key columns:
  - `fixture_id`, `team_id`, `is_home` (composite key)
  - rolling goals/sot/corners/points/rest

### `team_premium_snapshots`
- purpose: lagged premium rolling features by team
- key columns:
  - `fixture_id`, `team_id`, `is_home` (composite key)
  - rolling xg/xgot/box_touches/big_chances/crosses/etc.
  - built from prior matches only

### `predictions`
- purpose: store model outputs for each market
- key columns:
  - `prediction_id` (pk)
  - `fixture_id`
  - `market_code` (e.g., `o15`, `o25`, `u35`, `btts_yes`, `c85`, `ah_home`, `ah_away`)
  - `model_name`, `model_version`
  - `p_model`, `p_final`
  - `created_at`

### `prediction_scores`
- purpose: post-settlement scoring and betting analytics
- key columns:
  - `score_id` (pk)
  - `prediction_id` (fk)
  - `actual`
  - `brier`, `log_loss`, `hit`
  - `odds_used`, `edge`, `roi_unit`
  - `scored_at`

### `pipeline_runs`
- purpose: job observability/audit
- key columns:
  - `run_id` (pk)
  - `job_name` (`discover`, `resolve`, `settle`, `predict`, `score`)
  - `started_at`, `ended_at`, `status`, `message`

### `data_quality_runs`
- purpose: enforce QA gates before prediction/training
- key columns:
  - `dq_id` (pk)
  - `run_time`
  - coverage/fidelity/null metrics
  - `passed` boolean

## 2) minimum viable subset (v1 launch)

if you want the smallest safe build first, start with:
- `leagues`, `teams`, `team_aliases`
- `fixtures`, `fixture_results`
- `fixture_odds_snapshots`, `fixture_stats_premium`
- `team_premium_snapshots`
- `predictions`, `prediction_scores`

everything else can be added after stable daily operations.

## 3) daily operational flow

1. discover/update fixtures (`flashscore_id` required)
2. snapshot odds into `fixture_odds_snapshots`
3. settle ft fixtures -> write `fixture_results` + `fixture_stats_premium`
4. rebuild `team_premium_snapshots`
5. predict upcoming fixtures
6. score settled predictions
7. write `data_quality_runs` + `pipeline_runs`

## 4) non_negotiable rules

- `flashscore_id` is the business key across ingestion/resolution/settlement
- no same-match leakage in feature generation
- premium model training defaults to high fidelity rows (`fidelity_score >= 0.9`)
- unresolved past fixtures are marked and excluded from prediction queue
- market promotion to live requires per-league quality gates

### odds snapshot as-of and leakage guard
- pre-match odds features must satisfy `snapshot_time_utc <= kickoff`
- `src/modeling/layer2_markets/market_outcome_calibrator.py` and `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py` hard-fail on any post-kickoff odds snapshot
- for backfills, ingest `snapshot_type='closing'` snapshots anchored to kickoff when possible
- for upcoming fixtures, the tick job must refresh premium json pre-kickoff to capture true `latest_pre_match`

## 5) market set (target)

- `o15`
- `o25`
- `u35`
- `btts_yes`
- `c85`
- `ah_home`
- `ah_away`

note: if btts odds are not available in the odds source, keep `btts_yes` as shadow scoring until odds are ingested.
