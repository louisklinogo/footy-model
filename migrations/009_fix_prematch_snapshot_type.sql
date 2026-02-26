-- Migration 009: Canonicalize prematch odds snapshot type to latest_pre_match

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
        'latest_pre_match',
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
      AND snapshot_type = 'latest_pre_match'
    ORDER BY snapshot_time_utc DESC
    LIMIT 1
) fos ON true;

UPDATE fixture_odds_snapshots
SET snapshot_type = 'latest_pre_match'
WHERE snapshot_type = 'prematch';
