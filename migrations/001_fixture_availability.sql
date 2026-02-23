-- Migration 001: fixture_availability table
-- Stores pre-match injury/lineup data scraped from Flashscore

CREATE TABLE IF NOT EXISTS fixture_availability (
    fixture_id BIGINT PRIMARY KEY REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    
    -- Home team availability
    home_missing JSONB NOT NULL DEFAULT '[]'::jsonb,
    home_questionable JSONB NOT NULL DEFAULT '[]'::jsonb,
    home_lineup JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Away team availability
    away_missing JSONB NOT NULL DEFAULT '[]'::jsonb,
    away_questionable JSONB NOT NULL DEFAULT '[]'::jsonb,
    away_lineup JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Metadata
    scraped_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for finding fixtures by scrape time (useful for re-scraping old data)
CREATE INDEX IF NOT EXISTS idx_fixture_availability_scraped_at 
    ON fixture_availability(scraped_at DESC);

-- Comment on table for documentation
COMMENT ON TABLE fixture_availability IS 'Pre-match player availability data (injuries, suspensions, lineups) scraped from Flashscore';
COMMENT ON COLUMN fixture_availability.home_missing IS 'JSON array of home team players who will not play: [{"name": "Player Name", "reason": "Knee Injury"}]';
COMMENT ON COLUMN fixture_availability.home_questionable IS 'JSON array of home team players who are questionable';
COMMENT ON COLUMN fixture_availability.home_lineup IS 'JSON array of predicted/confirmed home team lineup';
