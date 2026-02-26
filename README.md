# footy-model

Fixtures-first, DB-first football modeling pipeline keyed by `fixtures.flashscore_id`.

## Quickstart

1. Install JS dependencies (Bun only):

```bash
bun install
```

2. Set `DATABASE_URL` in your environment.
3. Bootstrap the v1 schema:

```bash
python src/ingest/bootstrap_fixtures_schema_v1.py
```

4. Run the daily fixtures-first pipeline:

```bash
python src/pipelines/daily_pipeline.py --mode incremental --days 3
```

## Main entrypoints

Daily pipeline (discover -> ingest -> enrich -> ingest premium -> readiness -> snapshots -> predict -> export -> score):

```bash
python src/pipelines/daily_pipeline.py
```

Frequent tick job (settle -> predict -> score for due fixtures):

```bash
python src/jobs/tick_due_fixtures_v1.py --leagues E0
```

Web app (prediction browser):

```bash
python -m uvicorn src.api.app:app --reload
```

Model scripts:

```bash
python src/modeling/layer2_markets/market_outcome_calibrator.py
python src/modeling/evaluation/predict_market_outcomes_fixtures_first.py --league E0 --days 3
python src/modeling/export/export_market_outcomes_fixtures_first.py --league E0 --days 3
python src/modeling/evaluation/score_market_outcomes_fixtures_first.py --league E0
```

## Output paths (`data/v1/`)

- Discovery fixtures: `data/v1/discovery/discovery_fixtures_<league>.json`
- Fixture ID lists: `data/v1/ids/match_ids_<league>.json`, `data/v1/ids/upcoming_ids_<league>.json`
- Tick ID batches: `data/v1/ids/tick/<timestamp>/`
- Discovery reports: `data/v1/reports/seed_summary_<mode>.json`
- Premium match payloads: `data/v1/premium/<league>/<flashscore_id>.json`
- Daily prediction export CSV (default): `storage/reports/market_predictions.csv`

## Odds as-of semantics and leakage guards

- Train and predict both use odds snapshots with as-of semantics: latest snapshot where `snapshot_time_utc <= match_datetime_utc`.
- `src/modeling/layer2_markets/market_outcome_calibrator.py` has a hard guard: if any row has post-kickoff odds (`odds_snapshot_time_utc > match_datetime_utc`), training aborts.
- `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py` has the same hard guard and aborts if post-kickoff odds are detected.

## Testing

```bash
pytest -q
```
