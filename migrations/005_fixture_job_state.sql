-- Migration 005: Add fixture_job_state table
-- Tracks settlement attempts and job state per fixture

CREATE TABLE IF NOT EXISTS fixture_job_state (
    fixture_id BIGINT PRIMARY KEY,
    last_status_check_at TIMESTAMPTZ,
    settle_attempts INTEGER NOT NULL DEFAULT 0,
    last_settle_attempt_at TIMESTAMPTZ,
    last_predict_at TIMESTAMPTZ,
    last_score_at TIMESTAMPTZ,
    CONSTRAINT fixture_job_state_fixture_fk
        FOREIGN KEY (fixture_id)
        REFERENCES fixtures (fixture_id)
        ON DELETE CASCADE
);

-- Index for querying fixtures needing settlement
CREATE INDEX IF NOT EXISTS idx_fixture_job_state_settle_attempts
ON fixture_job_state(settle_attempts)
WHERE settle_attempts > 0;

COMMENT ON TABLE fixture_job_state IS 'Tracks pipeline job state per fixture: settlement attempts, predictions, scoring';
