-- Migration 014: Pre-match risk assessments for market predictions

CREATE TABLE IF NOT EXISTS prediction_risk_assessments (
    risk_assessment_id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    market_code TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    p_model DOUBLE PRECISION NOT NULL CHECK (p_model >= 0.0 AND p_model <= 1.0),
    p_conservative DOUBLE PRECISION NOT NULL CHECK (p_conservative >= 0.0 AND p_conservative <= 1.0),
    implied_probability DOUBLE PRECISION CHECK (implied_probability >= 0.0 AND implied_probability <= 1.0),
    odds_used DOUBLE PRECISION CHECK (odds_used IS NULL OR odds_used > 0.0),
    edge_raw DOUBLE PRECISION,
    edge_adjusted DOUBLE PRECISION,
    risk_score DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 100.0),
    action TEXT NOT NULL CHECK (action IN ('pass', 'watch', 'bet_small', 'bet')),
    kelly_fraction DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (kelly_fraction >= 0.0 AND kelly_fraction <= 1.0),
    stake_fraction DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (stake_fraction >= 0.0 AND stake_fraction <= 1.0),
    risk_flags_json JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(risk_flags_json) = 'array'),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata_json) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (prediction_id),
    UNIQUE (fixture_id, market_code, model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_prediction_risk_assessments_fixture_action
    ON prediction_risk_assessments (fixture_id, action);

CREATE INDEX IF NOT EXISTS idx_prediction_risk_assessments_model_fixture
    ON prediction_risk_assessments (model_name, model_version, fixture_id);

CREATE INDEX IF NOT EXISTS idx_prediction_risk_assessments_updated_at
    ON prediction_risk_assessments (updated_at DESC);

COMMENT ON TABLE prediction_risk_assessments IS 'Pre-match risk-adjusted decision layer for each prediction row';
COMMENT ON COLUMN prediction_risk_assessments.p_conservative IS 'Probability after risk haircut';
COMMENT ON COLUMN prediction_risk_assessments.risk_score IS '0..100 aggregate risk score (higher means riskier)';
COMMENT ON COLUMN prediction_risk_assessments.action IS 'Decision label: pass/watch/bet_small/bet';
