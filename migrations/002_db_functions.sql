-- Migration 002: Database helper functions
-- Encapsulates common queries and business logic

-- ============================================================================
-- TRIGGER: Auto-update updated_at timestamp
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_fixtures_updated_at ON fixtures;
CREATE TRIGGER trigger_fixtures_updated_at
    BEFORE UPDATE ON fixtures
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- FUNCTION: get_fixtures_to_scrape
-- Returns fixtures that need prematch data scraping
-- ============================================================================

CREATE OR REPLACE FUNCTION get_fixtures_to_scrape(days_ahead INT DEFAULT 3)
RETURNS TABLE (
    fixture_id BIGINT,
    flashscore_id TEXT,
    league_code TEXT,
    home_team TEXT,
    away_team TEXT,
    match_datetime_utc TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.fixture_id,
        f.flashscore_id,
        f.league_code,
        ht.team_name,
        at.team_name,
        f.match_datetime_utc
    FROM fixtures f
    JOIN teams ht ON f.home_team_id = ht.team_id
    JOIN teams at ON f.away_team_id = at.team_id
    LEFT JOIN fixture_availability fa ON f.fixture_id = fa.fixture_id
    WHERE f.status = 'scheduled'
    AND f.match_datetime_utc BETWEEN NOW() AND NOW() + days_ahead * INTERVAL '1 day'
    AND (
        fa.fixture_id IS NULL 
        OR fa.scraped_at < NOW() - INTERVAL '12 hours'
    )
    ORDER BY f.match_datetime_utc;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_fixtures_to_scrape(INT) IS 'Returns upcoming fixtures that need prematch data scraping (no data or stale >12h)';

-- ============================================================================
-- FUNCTION: settle_fixture
-- Atomically settles a fixture with final score
-- ============================================================================

CREATE OR REPLACE FUNCTION settle_fixture(
    p_flashscore_id TEXT,
    p_home_goals INT,
    p_away_goals INT
) RETURNS VOID AS $$
DECLARE
    v_fixture_id BIGINT;
BEGIN
    -- Get fixture_id
    SELECT fixture_id INTO v_fixture_id 
    FROM fixtures WHERE flashscore_id = p_flashscore_id;
    
    IF v_fixture_id IS NULL THEN
        RAISE EXCEPTION 'Fixture not found: %', p_flashscore_id;
    END IF;
    
    -- Update fixture status
    UPDATE fixtures SET status = 'ft' WHERE fixture_id = v_fixture_id;
    
    -- Insert or update result
    INSERT INTO fixture_results (fixture_id, home_goals, away_goals, result_status, settled_at)
    VALUES (v_fixture_id, p_home_goals, p_away_goals, 'ft', NOW())
    ON CONFLICT (fixture_id) DO UPDATE SET
        home_goals = EXCLUDED.home_goals,
        away_goals = EXCLUDED.away_goals,
        settled_at = NOW();
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION settle_fixture(TEXT, INT, INT) IS 'Atomically settle a fixture with final score';

-- ============================================================================
-- FUNCTION: get_stale_fixtures
-- Returns fixtures that are scheduled but past match time
-- ============================================================================

CREATE OR REPLACE FUNCTION get_stale_fixtures(hours_passed INT DEFAULT 3)
RETURNS TABLE (
    fixture_id BIGINT,
    flashscore_id TEXT,
    league_code TEXT,
    match_datetime_utc TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.fixture_id,
        f.flashscore_id,
        f.league_code,
        f.match_datetime_utc
    FROM fixtures f
    WHERE f.status = 'scheduled'
    AND f.match_datetime_utc < NOW() - hours_passed * INTERVAL '1 hour'
    ORDER BY f.match_datetime_utc;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_stale_fixtures(INT) IS 'Returns fixtures that are still scheduled but past match time by N hours';

-- ============================================================================
-- FUNCTION: get_ft_without_results
-- Returns fixtures marked as FT but missing result records
-- ============================================================================

CREATE OR REPLACE FUNCTION get_ft_without_results()
RETURNS TABLE (
    fixture_id BIGINT,
    flashscore_id TEXT,
    league_code TEXT,
    match_datetime_utc TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.fixture_id,
        f.flashscore_id,
        f.league_code,
        f.match_datetime_utc
    FROM fixtures f
    LEFT JOIN fixture_results fr ON f.fixture_id = fr.fixture_id
    WHERE f.status = 'ft'
    AND fr.result_id IS NULL
    ORDER BY f.match_datetime_utc;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_ft_without_results() IS 'Returns fixtures with FT status but no result record';

-- ============================================================================
-- FUNCTION: upsert_prematch_odds
-- Upserts prematch odds snapshot
-- ============================================================================

CREATE OR REPLACE FUNCTION upsert_prematch_odds(
    p_fixture_id BIGINT,
    p_ou_json JSONB,
    p_ah_json JSONB,
    p_1x2_json JSONB,
    p_scraped_at TIMESTAMPTZ DEFAULT NOW()
) RETURNS VOID AS $$
BEGIN
    INSERT INTO fixture_odds_snapshots (
        fixture_id, 
        snapshot_time_utc, 
        snapshot_type, 
        ou_json, 
        ah_json, 
        one_x_two_json
    )
    VALUES (
        p_fixture_id,
        COALESCE(p_scraped_at, NOW()),
        'prematch',
        p_ou_json,
        p_ah_json,
        p_1x2_json
    )
    ON CONFLICT (fixture_id, snapshot_time_utc, snapshot_type) DO UPDATE SET
        ou_json = EXCLUDED.ou_json,
        ah_json = EXCLUDED.ah_json,
        one_x_two_json = EXCLUDED.one_x_two_json;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION upsert_prematch_odds(BIGINT, JSONB, JSONB, JSONB, TIMESTAMPTZ) IS 'Upsert prematch odds snapshot for a fixture';
