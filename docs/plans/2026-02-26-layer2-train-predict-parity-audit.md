# Layer 2 Train/Predict Parity Audit (Step 5 Deliverable)

Date: 2026-02-26  
Scope:
- `src/modeling/layer2_situational/train_situational_residual.py`
- `src/modeling/layer2_situational/predict_situational_residual.py`
- shared contract: `src/modeling/layer2_situational/situational_utils.py`

## Objective
Confirm that Layer 2 train and predict paths use compatible feature semantics, timing gates, and model contract behavior.

## Findings
1. Feature-schema parity is enforced at inference.
   - Predict path loads `model_artifacts/situational_model/situational_model.pkl` and uses `model_blob["features"]`.
   - This prevents feature-column mismatch between trained artifact and runtime inference input.

2. Lambda baseline fetch path is shared.
   - Both train and predict call `situational_utils.latest_lambda_pairs_sql()`.
   - This keeps Layer 1 lambda selection logic consistent across train and predict.

3. Odds timing gate parity is present in both paths.
   - Both train and predict enforce `snapshot_time_utc <= kickoff` in SQL and fail fast if post-kickoff rows are selected.

4. Main parity risk is implementation duplication, not contract mismatch.
   - Significant feature engineering logic is duplicated across train and predict scripts (injury impact, last-5 goals, derived interactions).
   - Current outputs are compatible, but duplication increases future drift risk when one side changes without the other.

5. Data-quality blocker remains external to parity wiring.
   - `player_availability` historical timing contamination is still critical for injury-derived feature trustworthiness.
   - This is a source-timing issue, not a train/predict feature-schema mismatch.

## Assessment
- Train/predict contract parity: **acceptable for current freeze-stage experimentation**.
- Residual governance blocker: **still active** due to upstream availability timing quality, not due to train/predict wiring mismatch.

## Recommended Hardening
1. Move duplicated feature engineering into one shared builder module used by both train and predict.
2. Add a parity test that generates both train-like and predict-like frames for overlapping fixtures and compares shared feature columns.
3. Keep leakage/source audits in validation loop until availability timing metrics stabilize on newly accrued data.
