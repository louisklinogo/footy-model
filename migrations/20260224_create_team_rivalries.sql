-- Migration: Create team_rivalries lookup table
-- Created: 2026-02-24
-- Purpose: Seeds major football derbies/rivalries used for the `is_derby` situational feature.
-- Note: team_id values populated by seed_rivalries.py which matches on team name.

CREATE TABLE IF NOT EXISTS team_rivalries (
    rivalry_id   SERIAL PRIMARY KEY,
    team_id_a    BIGINT REFERENCES teams(team_id),
    team_id_b    BIGINT REFERENCES teams(team_id),
    rivalry_name TEXT NOT NULL,
    league_code  TEXT,
    UNIQUE(team_id_a, team_id_b)
);

-- Seed data inserted via seed_rivalries.py by fuzzy-matching team names in our teams table.
-- See: src/ingest/seed_rivalries.py
