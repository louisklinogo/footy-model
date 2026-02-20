# AGENTS.md (Draft for repo root)

This repository is a fixtures-first, DB-first football modeling + scraping pipeline.
Primary runtime is Python; JS is used for Flashscore discovery/enrichment.

If you need this at repo root, copy this file to `AGENTS.md`.

## Quick Orientation

- Python entrypoints: `daily_pipeline.py`, `jobs/`, `scrapers/*.py`, `models/*.py`, `webapp/main.py`
- JS entrypoints (CommonJS): `scrapers/*.js` (notably `scrapers/run_seed_all.js`, `scrapers/premium_enricher_v4.js`)
- Tests: `tests/` (pytest; DB-backed integration/regression style)
- Data outputs: `data/v1/` (discovery, ids, premium payloads, daily exports)

## Setup

### Environment variables

- `DATABASE_URL` is required for most DB-backed scripts/tests.
  - Fallbacks: `DEV_DATABASE_URL`, `PROD_DATABASE_URL` (see `db_utils.py`).

### Python env (recommended)

- Create a venv and install dev deps:
  - `python -m venv .venv`
  - `./.venv/Scripts/activate` (Windows) or `source .venv/bin/activate` (bash)
  - `pip install -r requirements-dev.txt`
- Note: runtime deps are not fully pinned in a root `requirements.txt`.
  - If a script fails with `ModuleNotFoundError`, add the missing package to your env.

### JS env (README says Bun)

- Install JS deps:
  - `bun install`
- `package.json` is CommonJS (`"type": "commonjs"`). Keep new JS in CJS unless you migrate intentionally.

## Build / Run Commands

### DB bootstrap

- Bootstrap v1 schema (README):
  - `python scrapers/bootstrap_fixtures_schema_v1.py`

### Daily pipeline

- Run the full daily pipeline:
  - `python daily_pipeline.py`
- Incremental mode (example from README):
  - `python daily_pipeline.py --mode incremental --days 3`
- Dry-run (logs commands without executing):
  - `python daily_pipeline.py --dry-run`

### Tick job

- Frequent tick (example from README):
  - `python jobs/tick_due_fixtures_v1.py --leagues E0`

### Web app (FastAPI)

- Run dev server:
  - `python -m uvicorn webapp.main:app --reload`

### Scrapers / discovery / enrichment

- Discover/seed fixture IDs (node):
  - `node scrapers/run_seed_all.js --mode seed`
  - `node scrapers/run_seed_all.js --mode incremental --leagues E0,E1`
- Enrich premium payloads (Playwright via Crawlee):
  - `node scrapers/premium_enricher_v4.js E0 --ids-root data/v1/ids --out-root data/v1/premium`
- If Playwright browsers are missing:
  - `bunx playwright install` (or equivalent for your environment)

## Test Commands (pytest)

pytest config lives in `pytest.ini`:
- `testpaths = tests`
- `pythonpath = .`

Run all tests:
- `pytest -q`

Run one test file:
- `pytest -q tests/test_smoke_minipipeline.py`

Run one test function:
- `pytest -q tests/test_smoke_minipipeline.py::test_smoke_minipipeline_snapshot_predict_export`

Run by substring match:
- `pytest -q -k leakage_guard`

DB note:
- Tests use a DB fixture (`tests/conftest.py`) and will `pytest.skip(...)` if `DATABASE_URL` (or fallbacks)
  are not set. Expect many tests to be skipped without DB access.

## Lint / Format / Type Checking

- No repo-level formatter/linter configs were found (.editorconfig/ruff/black/isort/mypy/eslint/prettier).
- Python files commonly include inline Pyright suppression headers (e.g. `# pyright: ...`).
  - Treat these headers as the project’s current “type checking policy”.
  - If you introduce new Pyright errors, prefer fixing types; only relax rules narrowly.

Pragmatic checks you can run if you have tools installed:
- Syntax sanity: `python -m compileall .`
- Type check (if you use pyright locally): `pyright` (expect existing suppressions).

If you decide to add lint/format tooling, do it explicitly and repo-wide (avoid partial enforcement).

## Code Style Guidelines (follow existing patterns)

### Python

- Prefer `from __future__ import annotations` in new/edited modules (common across repo).
- Imports:
  - Standard library first, then third-party, then local imports.
  - Use explicit imports (avoid wildcard imports).
- Types:
  - Use modern typing (`list[str]`, `dict[str, object]`, unions with `|`).
  - Keep types honest; don’t add `Any` unless the boundary truly is untyped.
- Paths and IO:
  - Prefer `pathlib.Path` (see `daily_pipeline.py`).
  - Read/write text with explicit encoding (`utf-8`).
- Error handling:
  - Use `ValueError` for invalid CLI args / inputs (see `daily_pipeline.py:parse_league_list`).
  - Use `RuntimeError` for pipeline abort / impossible states.
  - When running subprocesses, capture stdout/stderr and surface failures with context (see `daily_pipeline.py:run_step`).
- Database:
  - Use parameterized SQL (`%s`) with psycopg2; never format SQL with f-strings.
  - Close connections in `finally` (see `pipeline_logging.py`).
  - Prefer idempotent upserts with `ON CONFLICT ... DO UPDATE` when appropriate (see `tests/conftest.py`).

### Tests

- Tests are integration-ish and DB-backed; keep them deterministic and self-cleaning.
- Follow existing patterns:
  - Use `db_case` fixture from `tests/conftest.py`.
  - Use helper inserters (`insert_league`, `insert_team`, `insert_fixture`, etc.).
  - Use `run_script([...])` to exercise CLI scripts end-to-end.

### JavaScript (scrapers)

- CommonJS only (package.json: `"type": "commonjs"`). Use `require(...)` not ESM imports.
- Keep scripts idempotent:
  - Skip work when output files already exist (see `scrapers/premium_enricher_v4.js`).
- Be explicit about CLI args parsing and defaults (see `scrapers/run_seed_all.js`).
- Avoid unnecessary abstraction; prioritize robustness (timeouts, retries, clear error messages).

## Repo Guardrails

- Don’t commit secrets (.env, DB URLs, auth cookies). Use environment variables.
- Prefer writing new outputs under `data/v1/` and keep paths consistent with README.
- Keep long-running steps observable:
  - Print step headers and commands (see `daily_pipeline.py`).
  - Persist pipeline run metadata to DB when possible (see `pipeline_logging.py`).

## Cursor / Copilot Rules

- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` were found in this checkout.
