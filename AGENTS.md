# AGENTS.md

This repository is a fixtures-first, DB-first football modeling and ingestion pipeline.
Production runtime is Python. JS Flashscore scrapers still exist but are legacy and not in the active scheduler chain.

## Quick Orientation

- Main phaseable job: `src/jobs/tick_due_fixtures_v1.py`
- Scheduler scripts (active):
  - `scripts/run_settle_scheduler.sh`
  - `scripts/run_predict_scheduler.sh`
  - `scripts/run_score_scheduler.sh`
  - `scripts/run_availability_scheduler.sh`
  - `scripts/run_odds_polling_scheduler.sh`
  - `scripts/run_validation_loop_scheduler.sh`
- DB run logging: `src/common/pipeline_logging.py` (`pipeline_runs` table)
- Tests: `tests/` (pytest; many are DB-backed)

## Source-of-Truth Policy

- Active enrichment and settlement path is SofaScore-first.
- Active schedulers should not call Flashscore scripts.
- `fixture_results.result_source` for active settlement flow should be `sofascore`.
- Flashscore code under `src/ingest/scrapers/*` is legacy/backfill unless explicitly scheduled.

## Setup

### Environment variables

- `DATABASE_URL` is required for most scripts.
- Fallbacks used in code: `DEV_DATABASE_URL`, `PROD_DATABASE_URL`.

### Python environment

- Create venv:
  - `python -m venv .venv`
  - `source .venv/bin/activate` (bash) or `./.venv/Scripts/activate` (Windows)
- Install deps:
  - `pip install -r requirements.txt`
- Note:
  - `pyarrow` is required for validation/monitoring parquet reads.

## Operational Runs

### Manual phase runs (single command)

- Settle only:
  - `python src/jobs/tick_due_fixtures_v1.py --skip-predict --skip-score --max-settle 100`
- Predict only:
  - `python src/jobs/tick_due_fixtures_v1.py --skip-settle --skip-score --max-predict 100 --predict-days 3`
- Score only:
  - `python src/jobs/tick_due_fixtures_v1.py --skip-settle --skip-predict --max-score 300 --score-since-days 30`
- Full legacy tick (manual only if needed):
  - `python src/jobs/tick_due_fixtures_v1.py`

### Scheduler scripts (recommended)

- `scripts/run_settle_scheduler.sh`
- `scripts/run_predict_scheduler.sh`
- `scripts/run_score_scheduler.sh`
- `scripts/run_availability_scheduler.sh`
- `scripts/run_odds_polling_scheduler.sh`
- `scripts/run_validation_loop_scheduler.sh`

All scheduler scripts:
- Use atomic `flock` lock files in `artifacts/locks/`
- Write stdout logs to `artifacts/logs/*_scheduler.stdout.log`
- Support optional `MAX_RUNTIME_MINUTES`

### Current cron shape (reference)

- `*/20 * * * *` settle
- `5,35 * * * *` predict
- `15 * * * *` score
- `0 * * * *` availability
- `*/20 * * * *` odds polling
- `30 5 * * *` validation loop

## Expected Runtime Behavior

- Settle can legitimately process `targets=0` during quiet windows.
- Score can legitimately report many `skipped` rows when fixtures are not yet fully settled/scorable.
- Availability can return lineup `404` before lineups are published; cooldown defers re-tries.
- Occasional upstream API timeout is expected; scheduler should continue and next cycle retries.

## Logs and Debugging

- Check scheduler logs first:
  - `artifacts/logs/settle_scheduler.stdout.log`
  - `artifacts/logs/predict_scheduler.stdout.log`
  - `artifacts/logs/score_scheduler.stdout.log`
  - `artifacts/logs/availability_scheduler.stdout.log`
  - `artifacts/logs/odds_polling_scheduler.stdout.log`
  - `artifacts/logs/validation_loop_scheduler.stdout.log`
- Check run-state in DB:
  - `pipeline_runs` for `tick_due_fixtures_v1`, `tick_due_fixtures_v1.settle`, `tick_due_fixtures_v1.predict`, `tick_due_fixtures_v1.score`, `sofascore_odds_polling`

## Testing

- Run all tests:
  - `pytest -q`
- Run focused tests:
  - `pytest -q tests/test_smoke_minipipeline.py`
  - `pytest -q -k leakage_guard`

DB-backed tests will skip when DB env vars are missing.

## Coding Guardrails

- Use parameterized SQL (`%s`) with psycopg2.
- Close DB connections in `finally`.
- Keep scheduler scripts idempotent and lock-protected.
- Do not reintroduce Flashscore calls into active scheduler chain unless explicitly requested.
- Keep outputs and reports under existing `artifacts/` and `storage/` structure.