-- Migration 004: Add settlement_status column to fixtures
-- Tracks data pipeline state separately from match status

ALTER TABLE fixtures
ADD COLUMN IF NOT EXISTS settlement_status TEXT
DEFAULT 'pending'
CHECK (settlement_status IN ('pending', 'due', 'settled', 'failed'));

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_fixtures_settlement_status
ON fixtures(settlement_status);

-- Update existing fixtures based on current state
-- FT fixtures with results -> settled
UPDATE fixtures f
SET settlement_status = 'settled'
WHERE f.status = 'ft'
AND EXISTS (SELECT 1 FROM fixture_results fr WHERE fr.fixture_id = f.fixture_id);

-- FT fixtures without results -> failed (or due, depending on attempts)
UPDATE fixtures f
SET settlement_status = 'failed'
WHERE f.status = 'ft'
AND NOT EXISTS (SELECT 1 FROM fixture_results fr WHERE fr.fixture_id = f.fixture_id);

-- Scheduled fixtures past match time -> due
UPDATE fixtures f
SET settlement_status = 'due'
WHERE f.status = 'scheduled'
AND f.match_datetime_utc < NOW();

-- Add comment
COMMENT ON COLUMN fixtures.settlement_status IS 'Data pipeline state: pending (waiting), due (needs result), settled (has result), failed (exhausted retries)';
