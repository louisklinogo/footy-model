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
    sofascore_league_id BIGINT,
    sofascore_season_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS teams (
    team_id BIGSERIAL PRIMARY KEY,
    team_name TEXT NOT NULL,
    league_code TEXT REFERENCES leagues(league_code) ON DELETE SET NULL,
    sofascore_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE leagues ADD COLUMN IF NOT EXISTS league_id BIGSERIAL;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS league_code TEXT;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS league_name TEXT;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS country TEXT;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS sofascore_league_id BIGINT;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS sofascore_season_id BIGINT;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

ALTER TABLE teams ADD COLUMN IF NOT EXISTS team_id BIGSERIAL;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS team_name TEXT;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS league_code TEXT;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS sofascore_id BIGINT;
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
    season VARCHAR(16),
    settlement_status TEXT,
    sofascore_id VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        home_team_id IS NULL
        OR away_team_id IS NULL
        OR home_team_id <> away_team_id
    )
);

ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS season VARCHAR(16);
ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS settlement_status TEXT;
ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS sofascore_id VARCHAR(64);

CREATE TABLE IF NOT EXISTS players (
    player_id SERIAL PRIMARY KEY,
    sofascore_id TEXT UNIQUE,
    name TEXT NOT NULL,
    slug TEXT,
    short_name TEXT,
    position TEXT,
    user_count INTEGER,
    market_value_euro BIGINT,
    nationality_code TEXT,
    country_name TEXT,
    date_of_birth DATE,
    height INTEGER,
    preferred_foot TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE players ADD COLUMN IF NOT EXISTS sofascore_id TEXT;
ALTER TABLE players ADD COLUMN IF NOT EXISTS name TEXT;
ALTER TABLE players ADD COLUMN IF NOT EXISTS slug TEXT;
ALTER TABLE players ADD COLUMN IF NOT EXISTS short_name TEXT;
ALTER TABLE players ADD COLUMN IF NOT EXISTS position TEXT;
ALTER TABLE players ADD COLUMN IF NOT EXISTS user_count INTEGER;
ALTER TABLE players ADD COLUMN IF NOT EXISTS market_value_euro BIGINT;
ALTER TABLE players ADD COLUMN IF NOT EXISTS nationality_code TEXT;
ALTER TABLE players ADD COLUMN IF NOT EXISTS country_name TEXT;
ALTER TABLE players ADD COLUMN IF NOT EXISTS date_of_birth DATE;
ALTER TABLE players ADD COLUMN IF NOT EXISTS height INTEGER;
ALTER TABLE players ADD COLUMN IF NOT EXISTS preferred_foot TEXT;
ALTER TABLE players ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE players ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

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
    h_yellow_cards INTEGER,
    a_yellow_cards INTEGER,
    h_red_cards INTEGER,
    a_red_cards INTEGER,
    fidelity_score DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (fidelity_score >= 0.0 AND fidelity_score <= 1.0),
    raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE fixture_stats_premium ADD COLUMN IF NOT EXISTS h_yellow_cards INTEGER;
ALTER TABLE fixture_stats_premium ADD COLUMN IF NOT EXISTS a_yellow_cards INTEGER;
ALTER TABLE fixture_stats_premium ADD COLUMN IF NOT EXISTS h_red_cards INTEGER;
ALTER TABLE fixture_stats_premium ADD COLUMN IF NOT EXISTS a_red_cards INTEGER;

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

CREATE TABLE IF NOT EXISTS team_external_context (
    context_id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    context_type TEXT NOT NULL,
    team_id BIGINT REFERENCES teams(team_id) ON DELETE SET NULL,
    provider_team_id TEXT NOT NULL,
    league_code TEXT REFERENCES leagues(league_code) ON DELETE SET NULL,
    provider_league_id BIGINT,
    season_label TEXT,
    provider_season_id BIGINT,
    snapshot_time_utc TIMESTAMPTZ NOT NULL,
    overall_rank INTEGER,
    overall_points INTEGER,
    overall_played INTEGER,
    overall_wins INTEGER,
    overall_draws INTEGER,
    overall_losses INTEGER,
    overall_goals_for INTEGER,
    overall_goals_against INTEGER,
    overall_goal_diff INTEGER,
    home_rank INTEGER,
    home_points INTEGER,
    home_played INTEGER,
    home_wins INTEGER,
    home_draws INTEGER,
    home_losses INTEGER,
    home_goals_for INTEGER,
    home_goals_against INTEGER,
    home_goal_diff INTEGER,
    away_rank INTEGER,
    away_points INTEGER,
    away_played INTEGER,
    away_wins INTEGER,
    away_draws INTEGER,
    away_losses INTEGER,
    away_goals_for INTEGER,
    away_goals_against INTEGER,
    away_goal_diff INTEGER,
    form_sequence TEXT,
    form_points_last5 INTEGER,
    form_wins_last5 INTEGER,
    form_draws_last5 INTEGER,
    form_losses_last5 INTEGER,
    pregame_avg_rating DOUBLE PRECISION,
    pregame_position INTEGER,
    pregame_value DOUBLE PRECISION,
    performance_graph_points_avg DOUBLE PRECISION,
    performance_graph_goal_diff_avg DOUBLE PRECISION,
    performance_graph_samples INTEGER,
    league_stats_matches INTEGER,
    league_stats_goals_scored INTEGER,
    league_stats_goals_conceded INTEGER,
    league_stats_big_chances INTEGER,
    league_stats_shots_on_target INTEGER,
    league_stats_corners INTEGER,
    league_stats_average_ball_possession DOUBLE PRECISION,
    league_stats_accurate_passes_percentage DOUBLE PRECISION,
    league_stats_accurate_long_balls_percentage DOUBLE PRECISION,
    league_stats_accurate_crosses_percentage DOUBLE PRECISION,
    league_stats_clean_sheets INTEGER,
    league_stats_tackles INTEGER,
    league_stats_interceptions INTEGER,
    league_stats_saves INTEGER,
    league_stats_errors_leading_to_shot INTEGER,
    league_stats_total_duels INTEGER,
    league_stats_duels_won_percentage DOUBLE PRECISION,
    league_stats_total_aerial_duels INTEGER,
    league_stats_aerial_duels_won_percentage DOUBLE PRECISION,
    league_stats_possession_lost INTEGER,
    league_stats_offsides INTEGER,
    league_stats_fouls INTEGER,
    league_stats_yellow_cards INTEGER,
    league_stats_red_cards INTEGER,
    raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, context_type, provider_team_id, league_code, season_label, snapshot_time_utc)
);

CREATE TABLE IF NOT EXISTS match_external_context (
    context_id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    context_type TEXT NOT NULL,
    fixture_id BIGINT REFERENCES fixtures(fixture_id) ON DELETE SET NULL,
    provider_fixture_id TEXT NOT NULL,
    snapshot_time_utc TIMESTAMPTZ NOT NULL,
    home_form_sequence TEXT,
    away_form_sequence TEXT,
    home_form_points_last5 INTEGER,
    away_form_points_last5 INTEGER,
    home_avg_rating DOUBLE PRECISION,
    away_avg_rating DOUBLE PRECISION,
    home_position INTEGER,
    away_position INTEGER,
    home_value DOUBLE PRECISION,
    away_value DOUBLE PRECISION,
    home_streak_win INTEGER,
    away_streak_win INTEGER,
    home_streak_unbeaten INTEGER,
    away_streak_unbeaten INTEGER,
    h2h_home_wins_last_n INTEGER,
    h2h_draws_last_n INTEGER,
    h2h_away_wins_last_n INTEGER,
    h2h_matches_count INTEGER,
    win_probability_home DOUBLE PRECISION,
    win_probability_draw DOUBLE PRECISION,
    win_probability_away DOUBLE PRECISION,
    provider_match_code TEXT,
    raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, context_type, provider_fixture_id, snapshot_time_utc)
);

CREATE TABLE IF NOT EXISTS player_external_context (
    context_id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    context_type TEXT NOT NULL,
    player_id INTEGER REFERENCES players(player_id) ON DELETE SET NULL,
    provider_player_id TEXT NOT NULL,
    league_code TEXT REFERENCES leagues(league_code) ON DELETE SET NULL,
    provider_league_id BIGINT,
    season_label TEXT,
    provider_season_id BIGINT,
    snapshot_time_utc TIMESTAMPTZ NOT NULL,
    position_group TEXT,
    attribute_attacking INTEGER,
    attribute_technical INTEGER,
    attribute_tactical INTEGER,
    attribute_defending INTEGER,
    attribute_creativity INTEGER,
    rating_avg DOUBLE PRECISION,
    minutes_played INTEGER,
    goals INTEGER,
    assists INTEGER,
    expected_goals DOUBLE PRECISION,
    expected_assists DOUBLE PRECISION,
    market_value_euro_snapshot BIGINT,
    raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, context_type, provider_player_id, league_code, season_label, snapshot_time_utc)
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

CREATE TABLE IF NOT EXISTS prediction_risk_assessments (
    risk_assessment_id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    market_code TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    p_model DOUBLE PRECISION NOT NULL CHECK (p_model >= 0.0 AND p_model <= 1.0),
    p_conservative DOUBLE PRECISION NOT NULL CHECK (p_conservative >= 0.0 AND p_conservative <= 1.0),
    implied_probability DOUBLE PRECISION CHECK (implied_probability >= 0.0 AND implied_probability <= 1.0),
    odds_used DOUBLE PRECISION CHECK (odds_used IS NULL OR odds_used > 0.0),
    edge_raw DOUBLE PRECISION,
    edge_adjusted DOUBLE PRECISION,
    risk_score DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 100.0),
    action TEXT NOT NULL CHECK (action IN ('pass', 'watch', 'bet_small', 'bet')),
    kelly_fraction DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (kelly_fraction >= 0.0 AND kelly_fraction <= 1.0),
    stake_fraction DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (stake_fraction >= 0.0 AND stake_fraction <= 1.0),
    risk_flags_json JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(risk_flags_json) = 'array'),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata_json) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (prediction_id),
    UNIQUE (fixture_id, market_code, model_name, model_version)
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

CREATE INDEX IF NOT EXISTS idx_teams_sofascore_id
    ON teams(sofascore_id)
    WHERE sofascore_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_players_sofascore_id
    ON players(sofascore_id)
    WHERE sofascore_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_fixtures_sofascore_id
    ON fixtures(sofascore_id)
    WHERE sofascore_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_team_external_context_team_snapshot
    ON team_external_context(team_id, snapshot_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_team_external_context_provider_snapshot
    ON team_external_context(provider, context_type, league_code, snapshot_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_match_external_context_fixture_snapshot
    ON match_external_context(fixture_id, snapshot_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_match_external_context_provider_snapshot
    ON match_external_context(provider, context_type, snapshot_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_player_external_context_player_snapshot
    ON player_external_context(player_id, snapshot_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_player_external_context_provider_snapshot
    ON player_external_context(provider, context_type, league_code, snapshot_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_fixture_odds_snapshots_fixture_snapshot_time
    ON fixture_odds_snapshots(fixture_id, snapshot_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_fixture_id
    ON predictions(fixture_id);

CREATE INDEX IF NOT EXISTS idx_prediction_risk_assessments_fixture_action
    ON prediction_risk_assessments(fixture_id, action);

CREATE INDEX IF NOT EXISTS idx_prediction_risk_assessments_model_fixture
    ON prediction_risk_assessments(model_name, model_version, fixture_id);

CREATE INDEX IF NOT EXISTS idx_prediction_risk_assessments_updated_at
    ON prediction_risk_assessments(updated_at DESC);

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
