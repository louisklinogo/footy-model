# footy-model

Canonical workspace for Flashscore-first migration and clean pre-match modeling.

## Current status

- Active clean model stack (leakage-free) is in `models/v2_clean`.
- Daily prediction output is `data/daily/predictions_v2_clean.csv`.
- Legacy scripts still exist but are not the primary production path.

## Single entrypoints (active)

Use `daily_pipeline.py` for the full cycle:

```bash
python daily_pipeline.py
```

Or use `models/run_v2_clean.py` for specific steps:

```bash
python models/run_v2_clean.py train
python models/run_v2_clean.py predict
python models/run_v2_clean.py eval-day --date 2026-02-17
```

Under the hood:

- Train/evaluate holdout: `models/train_v2_prematch_clean.py`
- Predict upcoming: `models/predict_v2_prematch_clean.py`
- Evaluate one prediction day: `models/evaluate_prediction_day.py`

## Directory intent

- `models/`: training, prediction, and evaluation code.
- `models/v2_clean/`: model artifacts and metrics for the clean stack.
- `models/legacy/`: archived scripts and models from previous iterations.
- `scrapers/`: Flashscore enrichment and ingestion tooling.
- `scrapers/legacy/`: archived scrapers and experimental scripts.
- `data/`: outputs (daily predictions, slips, mappings, training exports).
- `daily/`: operational daily notes/work products.

## Legacy vs active scripts

Active now:

- `models/train_v2_prematch_clean.py`
- `models/predict_v2_prematch_clean.py`
- `models/evaluate_prediction_day.py`
- `models/run_v2_clean.py`

Legacy/research (keep, but do not use as default production path):

- `models/train_multioutput.py`
- `models/predict_upcoming.py`
- `models/extract_features.py`
- `models/extract_features_v3.py`
- `models/train_v3_hierarchy.py`

## Next cleanup steps

1. Add one data contract doc for each table (`matches`, `matches_premium`, `upcoming_fixtures`).
2. Add one daily pipeline command (scrape -> ingest -> predict -> score).
3. Monitor daily pipeline performance.
