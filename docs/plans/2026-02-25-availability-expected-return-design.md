# Availability Expected Return Parsing Design

Date: 2026-02-25
Owner: DS/DB

## Summary

We will normalize `expected_return` values from Sofascore so they always resolve to a valid `datetime` or `None`. This prevents ingestion failures and keeps `player_availability.expected_return` consistent.

## Goals

- Accept expected return values in common Sofascore formats.
- Store a timezone-aware `datetime` in the DB or `None` when unavailable.
- Avoid breaking changes to the existing ingestion flow.

## Non-Goals

- Changing the backfill control flow or batch behavior.
- Rewriting availability ingestion logic or schema.

## Supported Input Formats

- Unix timestamps in seconds or milliseconds.
- ISO-8601 strings, including `Z` suffix.
- `None` or empty values.

## Implementation Plan

- Add `parse_expected_return` in `src/ingest/ingest_sofascore_availability.py`.
- Use it before inserting `expected_return` in `_upsert_availability_rows`.
- Add a small unit test to validate parsing behavior.

## Testing

- Unit test covers: seconds, milliseconds, ISO string, invalid string, and `None`.

## Rollout Notes

- No DB migrations required.
- Safe to deploy without downtime; no schema changes.
