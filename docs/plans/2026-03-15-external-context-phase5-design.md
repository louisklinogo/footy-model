# Phase 5 External Context Design

Status owner: Codex  
Date: 2026-03-15  
Source of truth: live Postgres schema plus active SofaScore ingestion patterns

## Goal

Add a provider-agnostic external-context layer for missing SofaScore-derived context that does not exist in the current DB surface.

## Table Names

- `team_external_context`
- `match_external_context`
- `player_external_context`

These names are intentionally not SofaScore-specific so the schema can absorb other external providers later.

## Entity Linkage Rule

Every context row stores:

- canonical repo entity ID for downstream joins
- provider entity ID for ingestion traceability
- provider name
- snapshot timestamp
- raw upstream payload

Canonical IDs are the primary join path for modeling and analytics. Provider IDs are preserved for reconciliation and re-fetching.

## Table Boundaries

### `team_external_context`

Purpose:
- team-level snapshots not tied to one fixture

Primary keys for uniqueness:
- `provider`
- `context_type`
- `provider_team_id`
- `league_code`
- `season_label`
- `snapshot_time_utc`

Expected context types:
- `standings_total`
- `standings_home`
- `standings_away`
- `team_overview`
- `league_stats`
- `performance_graph`

Normalized fields:
- standings rank / points / played / wins / draws / losses / goals for / goals against / goal diff
- recent form summaries
- selected league-stat summaries
- selected performance-graph summaries

### `match_external_context`

Purpose:
- fixture-level prematch snapshots

Primary keys for uniqueness:
- `provider`
- `context_type`
- `provider_fixture_id`
- `snapshot_time_utc`

Expected context types:
- `pre_match_form`
- `team_streaks`
- `h2h_results`
- `win_probability`

Normalized fields:
- home/away average rating, position, and points-like value
- home/away recent form sequence and points
- selected streak counts
- selected H2H summary counts
- optional monitoring-only win probability columns

### `player_external_context`

Purpose:
- player-level competition/season snapshots

Primary keys for uniqueness:
- `provider`
- `context_type`
- `provider_player_id`
- `league_code`
- `season_label`
- `snapshot_time_utc`

Expected context types:
- `attributes`
- `league_stats`
- `player_overview`

Normalized fields:
- position group
- attribute overview metrics
- rating / minutes / goals / assists / xg / xa
- market value snapshot

## Storage Policy

- bias toward a modest stable normalized column set
- keep full upstream payload in `raw_json`
- never overwrite history-only rows with “latest only” logic
- upsert on the full snapshot key only

## Ingestion Order

1. `team_external_context`
2. `match_external_context`
3. `player_external_context`

All three tables are created in Phase 5, but ingestion should still stabilize in that order.

## Provider And Mapping Policy

- source provider is currently SofaScore
- rows store `provider = 'sofascore'`
- ingestion uses SofaScore IDs to fetch payloads
- ingestion resolves those to canonical IDs before storing
- if canonical mapping is unavailable, retain the provider ID and raw payload rather than silently dropping the row

## Initial Implementation Scope

Schema:
- add the three new external-context tables
- align bootstrap with the live DB columns already used by SofaScore ingestion (`sofascore_*` columns and `players` table)

Ingestion scaffolding:
- add one CLI module for external-context ingestion
- support `team`, `match`, `player`, and `all` modes
- support `--league`, `--limit`, and `--dry-run`
- write normalized fields plus raw payloads

Not in this phase:
- no model feature wiring yet
- no scheduler wiring yet
- no promotion/evaluation tied to these tables yet

## Validation

Required after implementation:

- schema bootstrap idempotency
- normalization helper tests
- mapping tests for canonical/provider IDs
- one dry-run ingestion smoke pass
- row-count / freshness audit for the new tables

## Success Criteria

Phase 5 initial implementation is successful when:

- all three tables exist in the live DB
- the bootstrap script is idempotent
- ingestion scaffolding can fetch and write or dry-run all three context types
- stored rows preserve both canonical IDs and provider IDs
- snapshot timestamps make the new tables PIT-queryable later
