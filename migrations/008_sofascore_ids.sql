-- Migration 008: Add SofaScore IDs to teams and leagues
-- This adds the internal SofaScore IDs needed for API calls

-- Add sofascore_id to teams table
ALTER TABLE teams ADD COLUMN IF NOT EXISTS sofascore_id BIGINT;

-- Add sofascore league/season IDs to leagues table
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS sofascore_league_id BIGINT;
ALTER TABLE leagues ADD COLUMN IF NOT EXISTS sofascore_season_id BIGINT;

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_teams_sofascore_id ON teams(sofascore_id);
CREATE INDEX IF NOT EXISTS idx_leagues_sofascore_league_id ON leagues(sofascore_league_id);

-- Add comments
COMMENT ON COLUMN teams.sofascore_id IS 'SofaScore internal team ID for API calls';
COMMENT ON COLUMN leagues.sofascore_league_id IS 'SofaScore internal league ID';
COMMENT ON COLUMN leagues.sofascore_season_id IS 'SofaScore internal season ID';
