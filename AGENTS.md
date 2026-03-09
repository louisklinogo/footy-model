# AGENTS.md

This repository is a fixtures-first, DB-first football modeling and ingestion pipeline.
Production runtime is Python. JS Flashscore scrapers still exist but are legacy and not in the active scheduler chain.

## Quick Orientation

- Main phaseable job: `src/jobs/tick_due_fixtures_v1.py`
- Canonical current-state doc: `docs/current_state.md`
- Canonical live runtime runbook: `docs/ops_due_fixtures_job_v1.md`
- Canonical v2 challenger docs:
  - `docs/v2_upgrade_tracker.md`
  - `docs/v2_evaluation_workflow.md`
- Archived historical/planning docs: `docs/archive/`
- Scheduler scripts (active in this workspace):
  - `scripts/run_tick_scheduler.cmd`
  - `scripts/run_availability_scheduler.cmd`
  - `scripts/run_odds_polling_scheduler.cmd`
  - `scripts/run_validation_loop_scheduler.cmd`
- DB run logging: `src/common/pipeline_logging.py` (`pipeline_runs` table)
- Tests: `tests/` (pytest; many are DB-backed)

## Documentation Policy

- Treat live DB tables plus active entrypoint code as the highest source of truth.
- Use `docs/current_state.md` for the current runtime/model summary.
- Use `docs/v2_upgrade_tracker.md` and `docs/v2_evaluation_workflow.md` for v2 challenger work only.
- Treat files under `docs/archive/` as historical context, not current truth.

## V2 Evaluation / Promotion Guardrails

- Treat `model_artifacts/v2/baselines/metrics_baseline_v2.json` as a mutable registry, not unquestioned frozen truth.
- Before interpreting a v2 promotion failure, validate that the baseline registry covers all scoped markets and all required markets for the active promotion policy.
- Prefer explicit named frozen baseline snapshots for important promotion decisions; generic baseline files can drift after scope/contract changes.
- If baseline coverage is incomplete, rebuild or replace the baseline intentionally before doing any further model diagnosis.

## Current V2 Challenger References

- Current leading anytime challenger artifact: `model_artifacts/v2/anytime_direct_monotone_v1_candidate_20260308/`
- Current repo-backed bundled challenger reference:
  - scoreline: `model_artifacts/v2/scoreline_v21_total_intensity_snap_20260308/`
  - corners: `model_artifacts/v2/live_verification_corners_20260306/`
  - anytime: `model_artifacts/v2/anytime_direct_monotone_v1_candidate_20260308/`
  - bundled evaluation: `model_artifacts/v2/evaluation_bundle_refresh_anytime_direct_monotone_v1_20260309/`
  - bundled live compare: `artifacts/v2/family_replacement/live_replacement_20260309_bundle_anytime_direct_monotone_v1/`
- Stable v2 alias behavior:
  - `model_artifacts/v2/anytime` should mirror the current anytime leader for default v2 bundle/prediction/evaluation flows.
  - Explicit `--anytime-dir` overrides are still supported for challenger experiments.
- Important scope note:
  - The active scheduled production runtime is not the v2 bundle path; these references are for challenger/evaluation flows unless explicitly wired into runtime.

## Source-of-Truth Policy

- Active enrichment and settlement path is SofaScore-first.
- Active schedulers should not call Flashscore scripts.
- `fixture_results.result_source` for active settlement flow should be `sofascore`.
- Flashscore code under `src/ingest/scrapers/*` is legacy/backfill unless explicitly scheduled.

## Setup

### Environment variables

- `DATABASE_URL` is required for most scripts.
- Fallbacks used in code: `DEV_DATABASE_URL`, `PROD_DATABASE_URL`.
- `src/db/db_utils.py` calls `load_dotenv()` on import, so DB access can still work from `.env` even when the interactive shell appears to have no DB env vars set.
- DB resolution order in code is: `DATABASE_URL` -> `DEV_DATABASE_URL` -> `PROD_DATABASE_URL`.

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

- `scripts/run_tick_scheduler.cmd`
- `scripts/run_availability_scheduler.cmd`
- `scripts/run_odds_polling_scheduler.cmd`
- `scripts/run_validation_loop_scheduler.cmd`

All scheduler scripts:
- Use lock files in `artifacts/locks/`
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
  - `artifacts/logs/tick_scheduler.stdout.log`
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
- Put ad hoc smoke/self-check outputs under `artifacts/tmp/` (local scratch; gitignored) rather than creating new root-level scratch files or mixing temporary outputs into repo-backed benchmark/artifact directories.