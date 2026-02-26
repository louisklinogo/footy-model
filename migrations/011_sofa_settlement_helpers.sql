-- Migration 011: Sofa-aware settlement helpers and compatibility wrappers

-- ============================================================================
-- FUNCTION: settle_fixture_by_fixture_id
-- Atomically settles a fixture by fixture_id (provider-agnostic)
-- ============================================================================

CREATE OR REPLACE FUNCTION settle_fixture_by_fixture_id(
    p_fixture_id BIGINT,
    p_home_goals INT,
    p_away_goals INT
) RETURNS VOID AS $$
BEGIN
    IF p_fixture_id IS NULL THEN
        RAISE EXCEPTION 'Fixture id is required';
    END IF;

    -- Update fixture status and settlement marker.
    UPDATE fixtures
    SET status = 'ft',
        settlement_status = 'settled',
        updated_at = NOW()
    WHERE fixture_id = p_fixture_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Fixture not found: %', p_fixture_id;
    END IF;

    -- Insert or update result.
    INSERT INTO fixture_results (fixture_id, home_goals, away_goals, result_status, settled_at)
    VALUES (p_fixture_id, p_home_goals, p_away_goals, 'ft', NOW())
    ON CONFLICT (fixture_id) DO UPDATE SET
        home_goals = EXCLUDED.home_goals,
        away_goals = EXCLUDED.away_goals,
        result_status = EXCLUDED.result_status,
        settled_at = NOW();
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION settle_fixture_by_fixture_id(BIGINT, INT, INT)
IS 'Atomically settle a fixture with final score by fixture_id';

-- ============================================================================
-- FUNCTION: settle_fixture_by_sofascore_id
-- Atomically settles a fixture by sofascore_id
-- ============================================================================

CREATE OR REPLACE FUNCTION settle_fixture_by_sofascore_id(
    p_sofascore_id TEXT,
    p_home_goals INT,
    p_away_goals INT
) RETURNS VOID AS $$
DECLARE
    v_fixture_id BIGINT;
BEGIN
    SELECT fixture_id INTO v_fixture_id
    FROM fixtures
    WHERE sofascore_id = p_sofascore_id;

    IF v_fixture_id IS NULL THEN
        RAISE EXCEPTION 'Fixture not found for sofascore_id: %', p_sofascore_id;
    END IF;

    PERFORM settle_fixture_by_fixture_id(v_fixture_id, p_home_goals, p_away_goals);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION settle_fixture_by_sofascore_id(TEXT, INT, INT)
IS 'Atomically settle a fixture with final score by sofascore_id';

-- ============================================================================
-- FUNCTION: settle_fixture (legacy compatibility)
-- ============================================================================

CREATE OR REPLACE FUNCTION settle_fixture(
    p_flashscore_id TEXT,
    p_home_goals INT,
    p_away_goals INT
) RETURNS VOID AS $$
DECLARE
    v_fixture_id BIGINT;
BEGIN
    SELECT fixture_id INTO v_fixture_id
    FROM fixtures
    WHERE flashscore_id = p_flashscore_id;

    IF v_fixture_id IS NULL THEN
        RAISE EXCEPTION 'Fixture not found: %', p_flashscore_id;
    END IF;

    PERFORM settle_fixture_by_fixture_id(v_fixture_id, p_home_goals, p_away_goals);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION settle_fixture(TEXT, INT, INT)
IS 'Legacy settlement helper by flashscore_id; delegates to settle_fixture_by_fixture_id';

-- ============================================================================
-- FUNCTION: get_stale_fixtures_v2
-- Returns stale fixtures with both provider identifiers.
-- ============================================================================

CREATE OR REPLACE FUNCTION get_stale_fixtures_v2(hours_passed INT DEFAULT 3)
RETURNS TABLE (
    fixture_id BIGINT,
    sofascore_id TEXT,
    flashscore_id TEXT,
    league_code TEXT,
    match_datetime_utc TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.fixture_id,
        f.sofascore_id,
        f.flashscore_id,
        f.league_code,
        f.match_datetime_utc
    FROM fixtures f
    WHERE f.match_datetime_utc IS NOT NULL
      AND f.status = 'scheduled'
      AND f.match_datetime_utc < NOW() - hours_passed * INTERVAL '1 hour'
    ORDER BY (f.sofascore_id IS NOT NULL) DESC, f.match_datetime_utc ASC, f.fixture_id ASC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_stale_fixtures_v2(INT)
IS 'Returns scheduled fixtures past kickoff with both sofascore_id and flashscore_id (Sofa-prioritized order)';

-- ============================================================================
-- FUNCTION: get_ft_without_results_v2
-- Returns FT fixtures missing results with both provider identifiers.
-- ============================================================================

CREATE OR REPLACE FUNCTION get_ft_without_results_v2()
RETURNS TABLE (
    fixture_id BIGINT,
    sofascore_id TEXT,
    flashscore_id TEXT,
    league_code TEXT,
    match_datetime_utc TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.fixture_id,
        f.sofascore_id,
        f.flashscore_id,
        f.league_code,
        f.match_datetime_utc
    FROM fixtures f
    LEFT JOIN fixture_results fr ON f.fixture_id = fr.fixture_id
    WHERE f.status = 'ft'
      AND fr.result_id IS NULL
    ORDER BY (f.sofascore_id IS NOT NULL) DESC, f.match_datetime_utc ASC, f.fixture_id ASC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_ft_without_results_v2()
IS 'Returns FT fixtures without result records and both provider identifiers (Sofa-prioritized order)';

-- ============================================================================
-- Legacy wrappers keep existing callsites stable while inheriting v2 ordering.
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
        s.fixture_id,
        s.flashscore_id,
        s.league_code,
        s.match_datetime_utc
    FROM get_stale_fixtures_v2(hours_passed) s;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_stale_fixtures(INT)
IS 'Legacy stale-fixture helper returning flashscore_id; ordered via Sofa-aware v2 query';

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
        s.fixture_id,
        s.flashscore_id,
        s.league_code,
        s.match_datetime_utc
    FROM get_ft_without_results_v2() s;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_ft_without_results()
IS 'Legacy FT-without-results helper returning flashscore_id; ordered via Sofa-aware v2 query';
