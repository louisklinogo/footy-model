-- Migration 006: fixture_research table
-- Stores AI research results per fixture (injury impact, market sentiment)

CREATE TABLE IF NOT EXISTS fixture_research (
    fixture_id BIGINT PRIMARY KEY REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    
    -- Input context (what we already knew)
    home_missing_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    away_missing_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    home_questionable_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    away_questionable_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Factual findings (from web search)
    injury_news TEXT,
    odds_movement TEXT,
    market_volume TEXT,
    
    -- Categorical impact assessments
    overall_impact TEXT CHECK (overall_impact IN ('none', 'low', 'medium', 'high')),
    home_defensive_impact TEXT CHECK (home_defensive_impact IN ('none', 'low', 'medium', 'high')),
    home_attacking_impact TEXT CHECK (home_attacking_impact IN ('none', 'low', 'medium', 'high')),
    away_defensive_impact TEXT CHECK (away_defensive_impact IN ('none', 'low', 'medium', 'high')),
    away_attacking_impact TEXT CHECK (away_attacking_impact IN ('none', 'low', 'medium', 'high')),
    
    -- Market sentiment
    market_sentiment TEXT CHECK (market_sentiment IN ('with_home', 'with_away', 'neutral', 'against_home', 'against_away')),
    sentiment_confidence TEXT CHECK (sentiment_confidence IN ('low', 'medium', 'high')),
    
    -- Final recommendation
    recommendation TEXT CHECK (recommendation IN ('CONFIRM', 'DOWNGRADE', 'REJECT', 'NO_CHANGE')),
    reasoning TEXT,
    
    -- Tracking
    tavily_request_id TEXT,
    research_confidence TEXT CHECK (research_confidence IN ('low', 'medium', 'high')),
    researched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for finding unresearched fixtures
CREATE INDEX IF NOT EXISTS idx_fixture_research_researched_at
ON fixture_research(researched_at DESC);

COMMENT ON TABLE fixture_research IS 'AI research results: injury impact analysis, market sentiment, recommendations';
COMMENT ON COLUMN fixture_research.overall_impact IS 'Overall impact of missing players: none/low/medium/high';
COMMENT ON COLUMN fixture_research.market_sentiment IS 'Where the market/betting crowd is leaning';
COMMENT ON COLUMN fixture_research.recommendation IS 'AI recommendation: CONFIRM/DOWNGRADE/REJECT/NO_CHANGE';
