import sys
from pathlib import Path

import psycopg2

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import get_database_url


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS leagues (
    league_id BIGSERIAL PRIMARY KEY,
    league_code TEXT NOT NULL UNIQUE,
    league_name TEXT NOT NULL,
    country TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS teams (
    team_id BIGSERIAL PRIMARY KEY,
    team_name TEXT NOT NULL,
    league_code TEXT REFERENCES leagues(league_code) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE leagues ADD COLUMN IF NOT EXISTS league_id BIGSERIAL;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS league_code TEXT;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS league_name TEXT;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS country TEXT;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

ALTER TABLE teams ADD COLUMN IF NOT EXISTS team_id BIGSERIAL;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS team_name TEXT;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS league_code TEXT;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE teams ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE UNIQUE INDEX IF NOT EXISTS idx_leagues_league_code_unique
    ON leagues(league_code);

CREATE UNIQUE INDEX IF NOT EXISTS idx_leagues_league_id_unique
    ON leagues(league_id);

ALTER TABLE teams DROP CONSTRAINT IF EXISTS teams_team_name_key;

DROP INDEX IF EXISTS idx_teams_team_name_unique;

CREATE UNIQUE INDEX IF NOT EXISTS idx_teams_league_code_team_name_unique
    ON teams(league_code, team_name);

CREATE UNIQUE INDEX IF NOT EXISTS idx_teams_team_id_unique
    ON teams(team_id);

CREATE TABLE IF NOT EXISTS team_aliases (
    alias_id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    alias_name TEXT NOT NULL,
    team_id BIGINT NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, alias_name)
);

CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id BIGSERIAL PRIMARY KEY,
    flashscore_id TEXT NOT NULL UNIQUE,
    league_code TEXT REFERENCES leagues(league_code) ON DELETE SET NULL,
    home_team_id BIGINT REFERENCES teams(team_id) ON DELETE RESTRICT,
    away_team_id BIGINT REFERENCES teams(team_id) ON DELETE RESTRICT,
    match_datetime_utc TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('scheduled', 'live', 'ft', 'postponed', 'cancelled', 'abandoned')),
    flashscore_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        home_team_id IS NULL
        OR away_team_id IS NULL
        OR home_team_id <> away_team_id
    )
);

CREATE TABLE IF NOT EXISTS fixture_results (
    result_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    home_goals SMALLINT,
    away_goals SMALLINT,
    result_status TEXT NOT NULL DEFAULT 'ft',
    result_source TEXT NOT NULL DEFAULT 'flashscore',
    settled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (fixture_id)
);

CREATE TABLE IF NOT EXISTS fixture_odds_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    snapshot_time_utc TIMESTAMPTZ NOT NULL,
    snapshot_type TEXT NOT NULL CHECK (snapshot_type IN ('opening', 'latest_pre_match', 'live', 'closing')),
    ou_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ah_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    one_x_two_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (fixture_id, snapshot_time_utc, snapshot_type)
);

CREATE TABLE IF NOT EXISTS fixture_stats_premium (
    fixture_id BIGINT PRIMARY KEY REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    h_xg DOUBLE PRECISION,
    a_xg DOUBLE PRECISION,
    h_xgot DOUBLE PRECISION,
    a_xgot DOUBLE PRECISION,
    h_xa DOUBLE PRECISION,
    a_xa DOUBLE PRECISION,
    h_big_chances INTEGER,
    a_big_chances INTEGER,
    h_possession DOUBLE PRECISION,
    a_possession DOUBLE PRECISION,
    h_box_touches INTEGER,
    a_box_touches INTEGER,
    h_crosses INTEGER,
    a_crosses INTEGER,
    h_blocked_shots INTEGER,
    a_blocked_shots INTEGER,
    h_through_passes INTEGER,
    a_through_passes INTEGER,
    h_sot INTEGER,
    a_sot INTEGER,
    h_shots_inside_box INTEGER,
    a_shots_inside_box INTEGER,
    h_corners INTEGER,
    a_corners INTEGER,
    h_goals_prevented DOUBLE PRECISION,
    a_goals_prevented DOUBLE PRECISION,
    h_tackles_pct DOUBLE PRECISION,
    a_tackles_pct DOUBLE PRECISION,
    h_interceptions INTEGER,
    a_interceptions INTEGER,
    h_errors_lead_to_shot INTEGER,
    a_errors_lead_to_shot INTEGER,
    fidelity_score DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (fidelity_score >= 0.0 AND fidelity_score <= 1.0),
    raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS team_premium_snapshots (
    fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    team_id BIGINT NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
    is_home BOOLEAN NOT NULL,
    sample_size INTEGER NOT NULL DEFAULT 0,
    rolling_xg DOUBLE PRECISION,
    rolling_xg_against DOUBLE PRECISION,
    rolling_xgot DOUBLE PRECISION,
    rolling_xgot_against DOUBLE PRECISION,
    rolling_xa DOUBLE PRECISION,
    rolling_xa_against DOUBLE PRECISION,
    rolling_box_touches DOUBLE PRECISION,
    rolling_box_touches_against DOUBLE PRECISION,
    rolling_big_chances DOUBLE PRECISION,
    rolling_big_chances_against DOUBLE PRECISION,
    rolling_crosses DOUBLE PRECISION,
    rolling_crosses_against DOUBLE PRECISION,
    rolling_sot DOUBLE PRECISION,
    rolling_sot_against DOUBLE PRECISION,
    rolling_corners DOUBLE PRECISION,
    rolling_corners_against DOUBLE PRECISION,
    rolling_goals_prevented DOUBLE PRECISION,
    rolling_goals_prevented_against DOUBLE PRECISION,
    fidelity_score DOUBLE PRECISION,
    built_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (fixture_id, team_id, is_home)
);

ALTER TABLE team_premium_snapshots ADD COLUMN IF NOT EXISTS rolling_xg_against DOUBLE PRECISION;
ALTER TABLE team_premium_snapshots ADD COLUMN IF NOT EXISTS rolling_xgot_against DOUBLE PRECISION;
ALTER TABLE team_premium_snapshots ADD COLUMN IF NOT EXISTS rolling_xa_against DOUBLE PRECISION;
ALTER TABLE team_premium_snapshots ADD COLUMN IF NOT EXISTS rolling_box_touches_against DOUBLE PRECISION;
ALTER TABLE team_premium_snapshots ADD COLUMN IF NOT EXISTS rolling_big_chances_against DOUBLE PRECISION;
ALTER TABLE team_premium_snapshots ADD COLUMN IF NOT EXISTS rolling_crosses_against DOUBLE PRECISION;
ALTER TABLE team_premium_snapshots ADD COLUMN IF NOT EXISTS rolling_sot_against DOUBLE PRECISION;
ALTER TABLE team_premium_snapshots ADD COLUMN IF NOT EXISTS rolling_corners_against DOUBLE PRECISION;
ALTER TABLE team_premium_snapshots ADD COLUMN IF NOT EXISTS rolling_goals_prevented_against DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    market_code TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    p_model DOUBLE PRECISION NOT NULL CHECK (p_model >= 0.0 AND p_model <= 1.0),
    p_final DOUBLE PRECISION CHECK (p_final >= 0.0 AND p_final <= 1.0),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (fixture_id, market_code, model_name, model_version)
);

CREATE TABLE IF NOT EXISTS prediction_scores (
    score_id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    actual DOUBLE PRECISION,
    brier DOUBLE PRECISION,
    log_loss DOUBLE PRECISION,
    hit BOOLEAN,
    odds_used DOUBLE PRECISION,
    edge DOUBLE PRECISION,
    roi_unit DOUBLE PRECISION,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (prediction_id)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id BIGSERIAL PRIMARY KEY,
    job_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    message TEXT,
    details_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS data_quality_runs (
    dq_id BIGSERIAL PRIMARY KEY,
    run_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    coverage_pct DOUBLE PRECISION,
    fidelity_avg DOUBLE PRECISION,
    null_rate DOUBLE PRECISION,
    passed BOOLEAN NOT NULL,
    details_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_fixtures_status_match_datetime_utc
    ON fixtures(status, match_datetime_utc);

CREATE INDEX IF NOT EXISTS idx_fixture_odds_snapshots_fixture_snapshot_time
    ON fixture_odds_snapshots(fixture_id, snapshot_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_fixture_id
    ON predictions(fixture_id);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_job_name_started_at
    ON pipeline_runs(job_name, started_at DESC);
"""


def bootstrap_schema() -> None:
    database_url = get_database_url()
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()


if __name__ == "__main__":
    try:
        bootstrap_schema()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    print("V1 fixtures-first schema bootstrap complete.")
