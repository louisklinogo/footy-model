# Layer 1 Residual Anatomy Audit (Step 2)

Date: 2026-02-26  
Scope: all leagues, full history, `lambda_xgb`  
Decision-grade threshold: `n >= 120`  
Informative threshold: `n >= 50`

## Executive Finding
Residual anatomy timing for Layer 1 lambda is now materially repaired and usable for decisioning, with one explicit caveat:
- `feature_asof_utc` was backfilled for legacy `lambda_xgb` rows as `kickoff - 60 minutes` (aligned with current feature-build window).
- This is a defensible reconstruction for current pipeline semantics, but should still be treated as a modeled backfill, not raw source event truth.

## What Was Run
- Script: `src/modeling/layer2_situational/audit_layer1_residual_anatomy.py`
- Outputs:
  - `artifacts/reports/layer2_reconciliation/layer1_residual_anatomy.json`
  - `artifacts/reports/layer2_reconciliation/layer1_residual_anatomy.md`

## Key Numbers
- FT fixtures inspected: `7824`
- Fixtures with latest lambda pair: `7137`
- Fixtures with pre-kickoff lambda pair: `7137`
- Fixtures missing pre-kickoff lambda pair: `687`
- Latest chosen lambda pair after kickoff: `0 / 7137 = 0.00%`
- Latest pair lineage source: `feature_asof_utc` for all latest pairs

## Root Cause (Code-Level)
Original lineage corruption source:
- `src/modeling/layer1_poisson/predict_lambda.py`
- previous upsert behavior overwrote `created_at` on conflict.

Repair applied:
- upsert no longer overwrites `created_at`
- schema migration `013` added immutable/mutable lineage columns
- one-time backfill populated `feature_asof_utc` for existing `lambda_xgb` rows.

## Why This Matters
Layer 2 training and audit currently select latest lambda pairs. If timing history is overwritten:
- point-in-time integrity cannot be proven
- residual anatomy segments are not decision-grade
- train/predict parity assumptions become unreliable

## Immediate Decision
Layer 1 lambda timing lineage is no longer the primary blocker for reconciliation.

## Next Action Required
1. Continue writing `feature_asof_utc` on every new lambda prediction row (already implemented).
2. Keep residual anatomy anchored on lineage-safe fields (`feature_asof_utc` / `first_created_at`).
3. Shift primary blocker focus to `player_availability` timing contamination and sparse rivalry prevalence.

## Update: Post-Stabilization Rerun (Same Date)
After promoting the Layer 1 regularization defaults and rerunning full lambda refresh, the residual profile improved materially:

- Full pre-kickoff (`n=7137`):
  - home bias: `+0.0149`
  - away bias: `-0.0080`
  - home/away RMSE: `1.1959 / 1.0756`
- Recent pre-kickoff, 365d (`n=6447`):
  - home bias: `+0.0108`
  - away bias: `+0.0101`
  - home/away RMSE: `1.1750 / 1.0653`

Interpretation:
- Layer 1 remains chronologically valid and now also looks better calibrated on residual mean error.
- The active program blocker is no longer Layer 1 tuning; it is Layer 2 source coverage quality (odds + availability) and controlled reruns.
