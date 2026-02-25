-- Migration: create_player_tables.sql
-- Description: Creates the players and fixture_player_stats tables.

CREATE TABLE IF NOT EXISTS public.players (
    player_id BIGSERIAL PRIMARY KEY,
    sofascore_id INTEGER UNIQUE NOT NULL,
    name TEXT NOT NULL,
    slug TEXT,
    short_name TEXT,
    position TEXT,
    market_value_eur BIGINT,
    country_alpha2 CHAR(2),
    date_of_birth DATE,
    gender CHAR(1) DEFAULT 'M',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.fixture_player_stats (
    fixture_id BIGINT REFERENCES public.fixtures(fixture_id) ON DELETE CASCADE,
    player_id BIGINT REFERENCES public.players(player_id) ON DELETE CASCADE,
    team_id INTEGER, -- Simplified for now, or link to teams table if needed
    minutes_played INTEGER,
    rating DOUBLE PRECISION,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    expected_goals DOUBLE PRECISION,
    expected_assists DOUBLE PRECISION,
    expected_goals_on_target DOUBLE PRECISION,
    key_passes INTEGER DEFAULT 0,
    total_shots INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    total_tackles INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    recoveries INTEGER DEFAULT 0,
    duels_won INTEGER DEFAULT 0,
    duels_lost INTEGER DEFAULT 0,
    ground_duels_won INTEGER DEFAULT 0,
    aerial_duels_won INTEGER DEFAULT 0,
    dribbles_success INTEGER DEFAULT 0,
    was_fouled INTEGER DEFAULT 0,
    fouls INTEGER DEFAULT 0,
    catches INTEGER DEFAULT 0, -- GK specific
    saves INTEGER DEFAULT 0,   -- GK specific
    punches INTEGER DEFAULT 0, -- GK specific
    goals_prevented DOUBLE PRECISION, -- GK specific
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (fixture_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_fixture_player_stats_player ON public.fixture_player_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_fixture_player_stats_team ON public.fixture_player_stats(team_id);
