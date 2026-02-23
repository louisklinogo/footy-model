-- Migration 003: Convenience views for AI research and predictions

-- ============================================================================
-- VIEW: v_fixtures_with_context
-- Full fixture context for AI research (teams, odds, availability)
-- ============================================================================

CREATE OR REPLACE VIEW v_fixtures_with_context AS
SELECT 
    f.fixture_id,
    f.flashscore_id,
    f.league_code,
    ht.team_name AS home_team,
    at.team_name AS away_team,
    f.match_datetime_utc,
    f.status,
    -- Availability
    fa.home_missing,
    fa.away_missing,
    fa.home_questionable,
    fa.away_questionable,
    fa.home_lineup,
    fa.away_lineup,
    fa.scraped_at AS availability_scraped_at,
    -- Prematch odds (latest)
    fos.ou_json AS prematch_ou_odds,
    fos.ah_json AS prematch_ah_odds,
    fos.one_x_two_json AS prematch_1x2_odds,
    fos.snapshot_time_utc AS odds_snapshot_time
FROM fixtures f
JOIN teams ht ON f.home_team_id = ht.team_id
JOIN teams at ON f.away_team_id = at.team_id
LEFT JOIN fixture_availability fa ON f.fixture_id = fa.fixture_id
LEFT JOIN LATERAL (
    SELECT ou_json, ah_json, one_x_two_json, snapshot_time_utc
    FROM fixture_odds_snapshots
    WHERE fixture_id = f.fixture_id
    AND snapshot_type = 'prematch'
    ORDER BY snapshot_time_utc DESC
    LIMIT 1
) fos ON true;

COMMENT ON VIEW v_fixtures_with_context IS 'Full fixture context including teams, odds, and player availability for AI research';

-- ============================================================================
-- VIEW: v_edges_for_research
-- High-conviction prediction edges with context for AI research
-- ============================================================================

CREATE OR REPLACE VIEW v_edges_for_research AS
SELECT 
    vfc.fixture_id,
    vfc.flashscore_id,
    vfc.league_code,
    vfc.home_team,
    vfc.away_team,
    vfc.match_datetime_utc,
    vfc.status,
    vfc.home_missing,
    vfc.away_missing,
    vfc.home_questionable,
    vfc.away_questionable,
    vfc.prematch_ou_odds,
    vfc.prematch_1x2_odds,
    p.prediction_id,
    p.market_code,
    p.model_name,
    p.model_version,
    p.p_model AS model_probability,
    p.p_final AS final_probability,
    p.metadata_json->>'odds_used' AS odds_used,
    p.metadata_json->>'edge' AS edge
FROM v_fixtures_with_context vfc
JOIN predictions p ON vfc.fixture_id = p.fixture_id
WHERE vfc.status = 'scheduled'
ORDER BY p.p_model DESC;

COMMENT ON VIEW v_edges_for_research IS 'High-conviction prediction edges with full context for AI research pass';

-- ============================================================================
-- VIEW: v_upcoming_fixtures
-- Simple view of upcoming fixtures for scraping
-- ============================================================================

CREATE OR REPLACE VIEW v_upcoming_fixtures AS
SELECT 
    f.fixture_id,
    f.flashscore_id,
    f.league_code,
    ht.team_name AS home_team,
    at.team_name AS away_team,
    f.match_datetime_utc,
    f.status,
    CASE 
        WHEN fa.fixture_id IS NULL THEN 'never_scraped'
        WHEN fa.scraped_at < NOW() - INTERVAL '12 hours' THEN 'stale'
        ELSE 'fresh'
    END AS availability_status
FROM fixtures f
JOIN teams ht ON f.home_team_id = ht.team_id
JOIN teams at ON f.away_team_id = at.team_id
LEFT JOIN fixture_availability fa ON f.fixture_id = fa.fixture_id
WHERE f.status = 'scheduled'
AND f.match_datetime_utc > NOW()
ORDER BY f.match_datetime_utc;

COMMENT ON VIEW v_upcoming_fixtures IS 'Upcoming scheduled fixtures with availability status';

-- ============================================================================
-- VIEW: v_injury_summary
-- Quick summary of injuries for a match
-- ============================================================================

CREATE OR REPLACE VIEW v_injury_summary AS
SELECT 
    f.fixture_id,
    f.flashscore_id,
    ht.team_name AS home_team,
    at.team_name AS away_team,
    f.match_datetime_utc,
    jsonb_array_length(fa.home_missing) AS home_missing_count,
    jsonb_array_length(fa.away_missing) AS away_missing_count,
    jsonb_array_length(fa.home_questionable) AS home_questionable_count,
    jsonb_array_length(fa.away_questionable) AS away_questionable_count,
    fa.home_missing,
    fa.away_missing
FROM fixtures f
JOIN teams ht ON f.home_team_id = ht.team_id
JOIN teams at ON f.away_team_id = at.team_id
LEFT JOIN fixture_availability fa ON f.fixture_id = fa.fixture_id
WHERE f.status = 'scheduled'
AND (
    jsonb_array_length(fa.home_missing) > 0 
    OR jsonb_array_length(fa.away_missing) > 0
);

COMMENT ON VIEW v_injury_summary IS 'Upcoming fixtures with injuries, with counts and details';
