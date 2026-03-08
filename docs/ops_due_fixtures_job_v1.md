# Ops Runbook: due fixtures tick job v1

> Canonical for the current live/runtime prediction path as of 2026-03-06.
> For the repo-wide current-state summary, see `docs/current_state.md`.

`src/jobs/tick_due_fixtures_v1.py` runs the v1 fixtures-first operational tick in three phases:

1. settle: refresh due fixtures, enrich premium payloads, and ingest v1 fixture rows
2. predict: build team snapshots, generate fixtures-first predictions, and export outputs
3. score: score recent finished fixtures for each target league

## v1-only scope and paths

This job is intentionally tied to v1-only scripts and v1-only data paths.

- Tick entrypoint: `src/jobs/tick_due_fixtures_v1.py`
- Premium JSON output root: `data/v1/premium`
- Tick ID batches root: `data/v1/ids/tick`
- v1 ingestion/prediction scripts invoked by the job:
  - `src/ingest/ingest_premium_fixtures_v1.py`
  - `src/features/build_team_premium_snapshots_v1.py`
  - `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py`
  - `src/modeling/export/export_market_outcomes_fixtures_first.py`
  - `src/modeling/evaluation/score_market_outcomes_fixtures_first.py`

## Environment variables

- Required DB connection: `DATABASE_URL` or `DEV_DATABASE_URL`
- Use the same DB env in local runs, Task Scheduler, cron, and GitHub Actions

## Install and runtime prerequisites

1. Install Python dependencies for the project environment.
2. Install JavaScript dependencies with Bun:

```bash
bun install
```

3. Ensure Python and Node are available on `PATH` (the job calls both).

## CLI flags

From `src/jobs/tick_due_fixtures_v1.py`:

- `--leagues` (comma-separated league codes, optional; defaults to enabled leagues in registry)
- `--predict-days` (default `3`)
- `--settlement-delay-minutes` (default `180`)
- `--max-settle` (default `25`)
- `--max-predict` (default `50`)
- `--max-score` (default `500`)
- `--score-since-days` (default `30`)
- `--dry-run` (print commands and skip mutating actions)

## Example commands

Required dry-run smoke test:

```bash
python src/jobs/tick_due_fixtures_v1.py --leagues E0 --dry-run
```

Required live single-league run:

```bash
python src/jobs/tick_due_fixtures_v1.py --leagues E0
```

Optional tuned run:

```bash
python src/jobs/tick_due_fixtures_v1.py --leagues E0,E1 --predict-days 3 --settlement-delay-minutes 180 --max-settle 25 --max-predict 50 --max-score 500 --score-since-days 30
```

## Windows Task Scheduler example

Run every 15 minutes.

- Program/script: `C:\Path\To\Python\python.exe`
- Add arguments:

```text
src\jobs\tick_due_fixtures_v1.py --leagues E0,E1 --predict-days 3 --settlement-delay-minutes 180 --max-settle 25 --max-predict 50 --max-score 500 --score-since-days 30
```

- Start in: `C:\Developer\soccer\footy-model`
- Set `DATABASE_URL` (or `DEV_DATABASE_URL`) in the task user environment.

PowerShell registration example:

```powershell
$action = New-ScheduledTaskAction -Execute "C:\Path\To\Python\python.exe" -Argument "src\jobs\tick_due_fixtures_v1.py --leagues E0,E1 --predict-days 3 --settlement-delay-minutes 180 --max-settle 25 --max-predict 50 --max-score 500 --score-since-days 30"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName "FootyTickDueFixturesV1" -Action $action -Trigger $trigger -Description "Run v1 due-fixtures settle/predict/score"
```

## Linux / Oracle cron example

Run every 15 minutes:

```cron
*/15 * * * * cd /opt/footy-model && DATABASE_URL='postgres://user:pass@host:5432/db' /usr/bin/python3 src/jobs/tick_due_fixtures_v1.py --predict-days 3 --settlement-delay-minutes 180 --max-settle 25 --max-predict 50 --max-score 500 --score-since-days 30 >> /var/log/footy/tick_due_fixtures_v1.log 2>&1
```

Single-league dry run:

```cron
*/10 * * * * cd /opt/footy-model && DATABASE_URL='postgres://user:pass@host:5432/db' /usr/bin/python3 src/jobs/tick_due_fixtures_v1.py --leagues E0 --dry-run >> /var/log/footy/tick_due_fixtures_v1_dry.log 2>&1
```

## GitHub Actions note

Use `on.schedule` in workflow YAML, then run this job command in a step. Configure DB credentials in repository/environment secrets (for example `DATABASE_URL`). Use Bun setup (`oven-sh/setup-bun`) and run `bun install` before invoking the Python tick command.
