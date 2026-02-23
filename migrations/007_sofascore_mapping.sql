-- Migration: Add SofaScore mapping support
-- Created: 2026-02-22
-- Purpose: Enable SofaScore ID tracking for fixtures and team aliases

-- =============================================================================
-- 1. Add sofascore_id to fixtures table
-- =============================================================================

ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS sofascore_id VARCHAR(50);

-- Create index for fast SofaScore lookups
CREATE INDEX IF NOT EXISTS idx_fixtures_sofascore_id ON fixtures(sofascore_id) WHERE sofascore_id IS NOT NULL;

-- Add unique constraint (nullable, but unique when present)
CREATE UNIQUE INDEX IF NOT EXISTS idx_fixtures_sofascore_id_unique ON fixtures(sofascore_id) WHERE sofascore_id IS NOT NULL;

-- =============================================================================
-- 2. Insert SofaScore provider into team_aliases if not exists
--    (team_aliases table already exists in bootstrap schema)
-- =============================================================================

-- Note: team_aliases table structure:
--   alias_id BIGSERIAL PRIMARY KEY
--   provider TEXT NOT NULL          -- 'flashscore', 'sofascore', etc.
--   alias_name TEXT NOT NULL        -- Provider-specific team name
--   team_id BIGINT NOT NULL         -- FK to teams table
--   created_at TIMESTAMPTZ
--   UNIQUE (provider, alias_name)

-- =============================================================================
-- 3. Helper function to find SofaScore match ID for a fixture
-- =============================================================================

CREATE OR REPLACE FUNCTION find_sofascore_match_id(
    p_match_date DATE,
    p_home_team_name TEXT,
    p_away_team_name TEXT
) RETURNS VARCHAR(50) AS $$
DECLARE
    v_home_sofa_id INTEGER;
    v_away_sofa_id INTEGER;
    v_sofascore_id VARCHAR(50);
BEGIN
    -- Get SofaScore team IDs from team_aliases
    SELECT ta.team_id INTO v_home_sofa_id
    FROM team_aliases ta
    WHERE ta.provider = 'sofascore' 
    AND LOWER(ta.alias_name) = LOWER(p_home_team_name);
    
    SELECT ta.team_id INTO v_away_sofa_id
    FROM team_aliases ta
    WHERE ta.provider = 'sofascore' 
    AND LOWER(ta.alias_name) = LOWER(p_away_team_name);
    
    -- If both teams found, look for matching fixture
    -- Note: This is a placeholder - actual matching is done in Python
    -- by querying SofaScore API with date + team IDs
    
    RETURN NULL; -- Placeholder - Python will do the actual lookup
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 4. View for fixtures with both Flashscore and SofaScore IDs
-- =============================================================================

CREATE OR REPLACE VIEW fixtures_with_ids AS
SELECT 
    f.fixture_id,
    f.flashscore_id,
    f.sofascore_id,
    f.league_code,
    ht.team_name AS home_team,
    at.team_name AS away_team,
    f.match_datetime_utc,
    f.status
FROM fixtures f
JOIN teams ht ON f.home_team_id = ht.team_id
JOIN teams at ON f.away_team_id = at.team_id;

-- =============================================================================
-- 5. Comments for documentation
-- =============================================================================

COMMENT ON COLUMN fixtures.sofascore_id IS 'SofaScore match/event ID for API lookups';
COMMENT ON TABLE team_aliases IS 'Maps provider-specific team names to canonical team_id. Providers: flashscore, sofascore';
