# Timestamp Lineage Repair Design

Date: 2026-02-26  
Status: implemented for schema + writers + audits; monitoring period active for new-data accrual  
Scope: Layer 2-critical upstream tables with mutable timing fields

## Problem
Two upstream sources currently use mutable timestamp fields that are overwritten on upsert:
- `predictions.created_at` for Layer 1 lambda rows
- `player_availability.recorded_at`

This prevents trustworthy point-in-time audits and makes leakage checks ambiguous.

## Design Goals
1. Preserve immutable first-seen timestamp.
2. Preserve mutable last-refresh timestamp separately.
3. Preserve semantic event timestamp (when known) separately from ingestion timestamp.
4. Keep backward compatibility with existing readers during migration.

## Proposed Schema Changes

### A) `predictions`
Add:
- `first_created_at TIMESTAMPTZ NULL`
- `last_refreshed_at TIMESTAMPTZ NULL`
- `feature_asof_utc TIMESTAMPTZ NULL`

Backfill:
- `first_created_at = created_at` for existing rows
- `last_refreshed_at = created_at` for existing rows

Write policy:
- On insert:
  - `created_at = NOW()` (legacy)
  - `first_created_at = NOW()`
  - `last_refreshed_at = NOW()`
  - `feature_asof_utc = derived decision timestamp when available`
- On conflict update:
  - never mutate `first_created_at`
  - mutate `last_refreshed_at = NOW()`
  - mutate prediction payload fields
  - do not use `created_at` as lineage anchor going forward

Read policy for Layer 2 audits:
- prefer `feature_asof_utc` if populated
- else fallback to `first_created_at`
- never use mutable refresh time for PIT integrity checks

### B) `player_availability`
Add:
- `first_recorded_at TIMESTAMPTZ NULL`
- `last_refreshed_at TIMESTAMPTZ NULL`
- `event_recorded_at TIMESTAMPTZ NULL` (source event/observation time when available)

Backfill:
- `first_recorded_at = recorded_at`
- `last_refreshed_at = recorded_at`
- `event_recorded_at = COALESCE(event_recorded_at, first_recorded_at, recorded_at)` for migration-time continuity

Write policy:
- On insert:
  - `recorded_at` retained for compatibility
  - `first_recorded_at = NOW()`
  - `last_refreshed_at = NOW()`
  - `event_recorded_at = parsed source event timestamp if present else observation/ingest timestamp`
- On conflict update:
  - never mutate `first_recorded_at`
  - mutate `last_refreshed_at = NOW()`
  - mutate status/reason fields
  - do not overwrite `event_recorded_at` with ingest time

Read policy for leakage checks:
- prefer `event_recorded_at` for true PIT validation
- if NULL, treat row timing certainty as unknown (not automatically pass/fail)

## Application Changes Required
1. Update Layer 1 writer:
   - `src/modeling/layer1_poisson/predict_lambda.py`
2. Update availability ingester:
   - `src/ingest/ingest_sofascore_availability.py`
3. Update leakage auditors to use new lineage columns.
4. Update residual anatomy audit to anchor on `feature_asof_utc` / `first_created_at`.

## Migration Strategy
1. Add columns with nullable defaults.
2. Backfill columns from legacy fields.
3. Ship writer updates.
4. Run dual-read period (legacy + new columns).
5. Switch audits to new columns.
6. Deprecate reliance on mutable legacy timestamps.

Migration artifact prepared:
- `migrations/013_prediction_availability_lineage.sql`

Migration and DB application status:
- `013_prediction_availability_lineage.sql` applied successfully to active DB on 2026-02-26.
- Trigger functions are active:
  - `trg_set_predictions_lineage_columns`
  - `trg_set_player_availability_lineage_columns`
- One-time backfill applied:
  - `predictions(lambda_xgb).feature_asof_utc = kickoff - 60 minutes` where null.

## Risk Notes
- Historical rows still cannot recover true source event time where it was never stored.
- This design improves future correctness and reduces ambiguity for all subsequent runs.

## Acceptance Criteria
1. New writes preserve immutable first timestamp.
2. Refreshes no longer overwrite lineage anchor.
3. Layer 2 audits report decision-grade timing integrity for new data.
4. Documentation and audit scripts explicitly use lineage-safe fields.

## Implementation Snapshot
Writer and audit implementation completed:
- `src/modeling/layer1_poisson/predict_lambda.py`
  - no longer overwrites `created_at` on conflict
  - writes lineage fields when available (`first_created_at`, `last_refreshed_at`, `feature_asof_utc`)
- `src/ingest/ingest_sofascore_availability.py`
  - no longer overwrites `recorded_at` on conflict
  - writes lineage fields when available (`first_recorded_at`, `last_refreshed_at`, `event_recorded_at`)
- audit readers now use lineage-safe timing fields with fallback logic:
  - `src/modeling/layer2_situational/audit_layer1_residual_anatomy.py`
  - `src/modeling/layer2_situational/audit_layer2_data_sources.py`
  - `src/db/audit_layer2_situational_leakage.py`

Remaining practical blocker:
- historical `player_availability` timing remains mostly post-kickoff in legacy rows; improvement now depends on accruing fresh pre-kickoff data under repaired writes.
