# Sofascore Odds Schema Design

Date: 2026-02-25
Owner: DS/DB

## Summary

We will keep legacy Flashscore odds as read-only history and introduce a new normalized odds table for Sofascore. The new schema supports multiple markets, lines, providers, and snapshot types without forcing odds into fixed JSON columns. Modeling will use Sofascore odds only, with strict as-of joins.

## Goals

- Make Sofascore the canonical odds provider for modeling.
- Preserve legacy Flashscore odds for audit and backtest continuity.
- Support multi-market odds without schema churn.
- Enforce snapshot semantics and leakage guards.

## Non-Goals

- Backfill all markets on day one.
- Use legacy Flashscore odds in new models.

## Current State

- `fixture_odds_snapshots` contains legacy odds from Flashscore only.
- `one_x_two_json` is empty across existing snapshots.
- Snapshot taxonomy is defined in docs but partially mismatched in legacy views and functions.
- Sofascore odds are not yet ingested.

## Proposed Schema

### New Table: fixture_odds_markets

Purpose: store odds by provider, market, and line with snapshot semantics.

Columns:
- `odds_id` BIGSERIAL PRIMARY KEY
- `fixture_id` BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE
- `provider` TEXT NOT NULL
- `provider_id` INT NULL
- `market_code` TEXT NOT NULL
- `line_num` NUMERIC(6,3) NULL
- `line_text` TEXT NULL
- `odds_json` JSONB NOT NULL
- `snapshot_time_utc` TIMESTAMPTZ NOT NULL
- `snapshot_type` TEXT NOT NULL CHECK (snapshot_type IN ('opening', 'latest_pre_match', 'live', 'closing'))
- `created_at` TIMESTAMPTZ NOT NULL DEFAULT NOW()

Constraints:
- Unique constraint on (fixture_id, provider, provider_id, market_code, line_num, line_text, snapshot_time_utc, snapshot_type)

Indexes:
- (fixture_id, snapshot_time_utc DESC)
- (fixture_id, provider, market_code, line_num, snapshot_time_utc DESC)
- Optional partial index for snapshot_type = 'latest_pre_match'

### Optional Reference Tables

- `odds_providers` (provider_id, provider_name)
- `odds_markets` (market_code, description, line_required)

## Data Model Semantics

- `provider` is the data source (e.g., sofascore, flashscore).
- `provider_id` is the book or sub-provider (e.g., Bet365 = 1).
- `market_code` is a stable code (1x2, ou, ah, btts, dnb, dc).
- `line_num` is used for numeric lines; `line_text` is a fallback for non-numeric lines.
- `odds_json` stores raw provider payload plus normalized fields:
  - raw payload under `raw`
  - opening prices under `prices_opening`
  - latest/closing prices under `prices_latest`
  - implied probabilities under `implied_prob`

## Migration Plan

1. Create `fixture_odds_markets` and indexes.
2. Leave `fixture_odds_snapshots` untouched and treat as legacy read-only.
3. Add a new ingestion path for Sofascore odds into `fixture_odds_markets`.
4. Update modeling joins to read Sofascore odds from `fixture_odds_markets` only.
5. Optionally add views for research that join new odds data.

## Modeling Usage

- Layer 2 will join `fixture_odds_markets` with filters:
  - provider = 'sofascore'
  - market_code = '1x2'
  - snapshot_type in ('latest_pre_match', 'closing')
  - snapshot_time_utc <= match_datetime_utc
- Compute market implied probabilities from normalized `odds_json`.
- Use `odds_model_gap_*` as the delta between market implied and Poisson.

## Testing and Verification

- Schema tests: table exists, constraints and indexes present.
- Ingestion tests: insert sample Sofascore 1x2 odds; verify unique constraint.
- Leakage test: no snapshot with snapshot_time_utc > match_datetime_utc used.
- Feature test: odds_model_gap_* is non-constant when Sofascore odds exist.

## Rollout Notes

- Keep Flashscore odds for audit only; do not mix providers in training.
- Backfills should use `snapshot_type='closing'` anchored to kickoff time.
- Future work: add market taxonomy and provider reference tables if needed.
