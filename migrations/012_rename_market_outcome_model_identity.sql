-- Rename legacy market model identity labels to descriptive canonical labels.
-- This is safe for existing prediction_scores because scores are keyed by prediction_id.

UPDATE predictions
SET
    model_name = 'market_outcome_gbm',
    model_version = 'fixtures_first_prematch_v1'
WHERE
    model_name = 'premium_gbm'
    AND model_version IN ('v3', 'market_outcome');
