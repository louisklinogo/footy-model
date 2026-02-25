-- Migration: Create Player Database Schema
-- Created: 2026-02-24
-- Updated: Added card columns, xg_on_target, penalty cols, captain, position_played.
--          Removed redundant market_value_raw.
--          Renamed was_fouled -> fouled_count for clarity.

CREATE TABLE IF NOT EXISTS players (
    player_id         SERIAL PRIMARY KEY,
    sofascore_id      TEXT UNIQUE NOT NULL,
    name              TEXT NOT NULL,
    slug              TEXT,
    short_name        TEXT,
    position          TEXT,           -- Natural / registered position
    user_count        INTEGER,        -- SofaScore follower count (proxy for star power)
    market_value_euro BIGINT,         -- Proposed market value in EUR
    nationality_code  TEXT,           -- ISO alpha-2
    country_name      TEXT,
    date_of_birth     DATE,
    height            INTEGER,        -- cm
    preferred_foot    TEXT,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fixture_player_stats (
    fixture_id           BIGINT   REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    player_id            BIGINT   REFERENCES players(player_id) ON DELETE CASCADE,
    team_id              BIGINT   REFERENCES teams(team_id) ON DELETE CASCADE,
    is_home              BOOLEAN,
    position_played      TEXT,           -- Position in this specific match
    captain              BOOLEAN  DEFAULT FALSE,

    -- Playtime
    minutes_played       INTEGER,
    substituted_in       BOOLEAN  DEFAULT FALSE,
    substituted_out      BOOLEAN  DEFAULT FALSE,
    rating               NUMERIC(4, 2), -- SofaScore rating (e.g. 7.4)

    -- Attacking
    goals                INTEGER  DEFAULT 0,
    assists              INTEGER  DEFAULT 0,
    expected_goals       NUMERIC(5, 3),
    expected_goals_ot    NUMERIC(5, 3), -- xG on target only
    expected_assists     NUMERIC(5, 3),
    shots_on_target      INTEGER  DEFAULT 0,
    shots_off_target     INTEGER  DEFAULT 0,
    shots_blocked        INTEGER  DEFAULT 0,
    key_passes           INTEGER  DEFAULT 0,
    penalty_scored       INTEGER  DEFAULT 0,
    penalty_missed       INTEGER  DEFAULT 0,

    -- Passing & Possession
    touches              INTEGER  DEFAULT 0,
    passes_total         INTEGER  DEFAULT 0,
    passes_accurate      INTEGER  DEFAULT 0,
    crosses_total        INTEGER  DEFAULT 0,
    crosses_accurate     INTEGER  DEFAULT 0,
    possessions_lost     INTEGER  DEFAULT 0,

    -- Defensive
    tackles_total        INTEGER  DEFAULT 0,
    interceptions        INTEGER  DEFAULT 0,
    clearances           INTEGER  DEFAULT 0,
    blocked_shots        INTEGER  DEFAULT 0,
    ground_duels_total   INTEGER  DEFAULT 0,
    ground_duels_won     INTEGER  DEFAULT 0,
    aerial_duels_total   INTEGER  DEFAULT 0,
    aerial_duels_won     INTEGER  DEFAULT 0,
    fouls                INTEGER  DEFAULT 0,
    fouled_count         INTEGER  DEFAULT 0, -- Times the player was fouled

    -- Discipline
    yellow_card          INTEGER  DEFAULT 0,
    red_card             INTEGER  DEFAULT 0,

    -- GK only
    saves                INTEGER  DEFAULT 0,
    punches              INTEGER  DEFAULT 0,
    runs_out             INTEGER  DEFAULT 0,
    high_claims          INTEGER  DEFAULT 0,

    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (fixture_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_fps_player  ON fixture_player_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_fps_team    ON fixture_player_stats(team_id);
CREATE INDEX IF NOT EXISTS idx_fps_fixture ON fixture_player_stats(fixture_id);
