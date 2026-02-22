# Workspace Cleanup and Reorganization Plan

The goal is to consolidate core logic into the `src/` directory and remove unused or placeholder directories to make the workspace cleaner and more professional.

## Proposed Changes

### [Cleanup] Redundant/Empty/Legacy Directories
- **Remove** the following directories that are either empty or contain only empty placeholders:
    - `pipelines/` (Logic moving to `src/pipelines/`)
    - `outputs/` (Standardized on `data/` or create on-demand)
    - `configs/`
    - `src/backtest/` (Empty)
- **Archive** legacy scripts into their respective `legacy/` folders:
    - `models/predict_v2_prematch_clean.py` -> `models/legacy/`
    - `models/train_v2_prematch_clean.py` -> `models/legacy/`
    - `models/run_v2_clean.py` -> `models/legacy/`
    - `models/v2_clean/` -> `models/legacy/v2_clean/`

### [Reorganization] Consolidating Core Logic into `src/`

#### [Component] Database & Common Utilities
- [MODIFY] [db_utils.py](file:///c:/Developer/soccer/footy-model/db_utils.py) -> `src/db/db_utils.py`
- [MODIFY] [audit_db_script.py](file:///c:/Developer/soccer/footy-model/audit_db_script.py) -> `src/db/audit_db_script.py`
- [MODIFY] [pipeline_logging.py](file:///c:/Developer/soccer/footy-model/pipeline_logging.py) -> `src/common/pipeline_logging.py`
- [MODIFY] [models/team_resolver.py](file:///c:/Developer/soccer/footy-model/models/team_resolver.py) -> `src/common/team_resolver.py`
- [MODIFY] [models/elo_utils.py](file:///c:/Developer/soccer/footy-model/models/elo_utils.py) -> `src/common/elo_utils.py`

#### [Component] Ingest & Scraping
- [MODIFY] [scrapers/ingest_*.py](file:///c:/Developer/soccer/footy-model/scrapers/) -> `src/ingest/`
    - Includes `ingest_premium_v4.py`, `ingest_premium_fixtures_v1.py`, etc.
- [MODIFY] [scrapers/bootstrap_fixtures_schema_v1.py](file:///c:/Developer/soccer/footy-model/scrapers/bootstrap_fixtures_schema_v1.py) -> `src/ingest/bootstrap_fixtures_schema_v1.py`
- [MODIFY] [scrapers/seed_leagues_v1.py](file:///c:/Developer/soccer/footy-model/scrapers/seed_leagues_v1.py) -> `src/ingest/seed_leagues_v1.py`
- [MODIFY] [scrapers/build_team_premium_snapshots_v1.py](file:///c:/Developer/soccer/footy-model/scrapers/build_team_premium_snapshots_v1.py) -> `src/features/build_team_premium_snapshots_v1.py`

#### [Component] Modeling & Pipeline
- [MODIFY] [daily_pipeline.py](file:///c:/Developer/soccer/footy-model/daily_pipeline.py) -> `src/pipelines/daily_pipeline.py`
- [MODIFY] [models/predict_v3_fixtures_first.py](file:///c:/Developer/soccer/footy-model/models/predict_v3_fixtures_first.py) -> `src/modeling/predict_v3_fixtures_first.py`
- [MODIFY] [models/train_v3_fixtures_first.py](file:///c:/Developer/soccer/footy-model/models/train_v3_fixtures_first.py) -> `src/modeling/train_v3_fixtures_first.py`
- [MODIFY] [models/score_predictions_v3_fixtures_first.py](file:///c:/Developer/soccer/footy-model/models/score_predictions_v3_fixtures_first.py) -> `src/modeling/score_predictions_v3_fixtures_first.py`
- [MODIFY] [models/calculate_stake.py](file:///c:/Developer/soccer/footy-model/models/calculate_stake.py) -> `src/betting/calculate_stake.py`

### [Configuration] Update Entry Points
- [MODIFY] [package.json](file:///c:/Developer/soccer/footy-model/package.json)
    - Remove defunct scripts: `auth`, `scrape`, `rank`, `standardize`, `load-data`.
    - Update remaining script paths to their new locations in `src/`.
- [MODIFY] [src/pipelines/daily_pipeline.py](file:///c:/Developer/soccer/footy-model/src/pipelines/daily_pipeline.py)
    - Adjust `ROOT` path and all subprocess call paths to match the new structure.

> [!IMPORTANT]
> This plan moves NON-SCRAPER logic out of `scrapers/` and into `src/ingest/`, and consolidates active V3 models into `src/modeling/`. V2 and other older versions are moved to `legacy/`.

---

## Verification Plan

### Automated Tests
I will run the existing test suite to ensure that the reorganization hasn't broken core database or pricing logic.
```bash
pytest
```
*Specifically focusing on:*
- `tests/test_smoke_minipipeline.py`
- `tests/test_fixtures_upsert.py`

### Manual Verification
- Run `python src/pipelines/daily_pipeline.py --dry-run` to verify that it still finds the scrapers and models correctly.
- Verify that `db_utils` functions can still be imported by other scripts.
