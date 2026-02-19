# Fixture Lifecycle Contract

## Scope

This contract defines canonical fixture lifecycle behavior for ingestion, prediction eligibility, and post-match scoring.

## Identity And Canonical Row Rule

- Business key: `flashscore_id`.
- Exactly one canonical `fixtures` row exists per `flashscore_id`.
- Reschedules and status changes update that same row; they do not create a new fixture row.

## Canonical Datetime Rule (UTC)

- Canonical kickoff field: `match_datetime_utc`.
- Storage type: `timestamptz` (UTC canonical).
- All provider/local kickoff times must be converted to UTC before persistence.
- Unknown kickoff is stored as `NULL` and is treated as not eligible for prediction.

## Allowed Statuses

`status` must be one of:

- `scheduled`
- `live`
- `ft`
- `postponed`
- `cancelled`
- `abandoned`

## Status Transition Contract

Allowed transitions:

- `scheduled` -> `live`, `postponed`, `cancelled`, `abandoned`
- `live` -> `ft`, `abandoned`, `cancelled`
- `postponed` -> `scheduled`, `cancelled`, `abandoned`
- `cancelled` -> terminal
- `abandoned` -> terminal
- `ft` -> terminal

Notes:

- `postponed` can return to `scheduled` only when a new kickoff is confirmed.
- Any kickoff update must preserve `flashscore_id` and update `match_datetime_utc` in-place.

## Reschedule Semantics

- A reschedule is any change to kickoff time/date for an existing `flashscore_id`.
- On reschedule:
  - update `match_datetime_utc` on the existing row,
  - update `status` according to provider truth (`postponed` if no new kickoff yet, `scheduled` when new kickoff is known),
  - update `updated_at`.
- Historical odds/stats/predictions remain linked through the same fixture identity.

## Prediction Eligibility Contract

Generate predictions only when all conditions are true:

1. `status = 'scheduled'`
2. `match_datetime_utc IS NOT NULL`
3. kickoff is in the future at prediction runtime

Do not generate new predictions for fixtures in `live`, `ft`, `postponed`, `cancelled`, or `abandoned`.

## Scoring And Invalidation Contract

- Scoring is allowed only for fixtures with `status = 'ft'` and a valid final result.
- Fixtures with `status` in (`cancelled`, `postponed`, `abandoned`) are non-scorable.
- Invalidation behavior for existing predictions on non-scorable fixtures:
  - no row is written to `prediction_scores`,
  - prediction records remain for audit,
  - downstream aggregates must exclude invalidated/non-scorable predictions from hit-rate, ROI, Brier, and log-loss metrics.

## Operational Invariants

- No fuzzy joins are allowed for fixture identity; joins must use `flashscore_id` (or resolved `fixture_id`).
- Lifecycle updates must be idempotent: replaying the same provider event cannot create duplicate fixtures or duplicate scoring rows.
