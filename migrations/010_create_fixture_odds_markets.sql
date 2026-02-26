-- Migration 010: Normalized odds markets table (Sofascore-ready, legacy Flashscore preserved)

CREATE TABLE IF NOT EXISTS fixture_odds_markets (
    odds_id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    provider_id INT,
    market_code TEXT NOT NULL,
    line_num NUMERIC(6,3),
    line_text TEXT,
    odds_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    snapshot_time_utc TIMESTAMPTZ NOT NULL,
    snapshot_type TEXT NOT NULL CHECK (snapshot_type IN ('opening', 'latest_pre_match', 'live', 'closing')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_fixture_odds_markets UNIQUE (
        fixture_id,
        provider,
        provider_id,
        market_code,
        line_num,
        line_text,
        snapshot_time_utc,
        snapshot_type
    )
);

CREATE INDEX IF NOT EXISTS idx_fixture_odds_markets_fixture_snapshot_time
    ON fixture_odds_markets(fixture_id, snapshot_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_fixture_odds_markets_provider_market_line_time
    ON fixture_odds_markets(fixture_id, provider, market_code, line_num, snapshot_time_utc DESC);

CREATE INDEX IF NOT EXISTS idx_fixture_odds_markets_latest_pre_match
    ON fixture_odds_markets(fixture_id, snapshot_time_utc DESC)
    WHERE snapshot_type = 'latest_pre_match';
