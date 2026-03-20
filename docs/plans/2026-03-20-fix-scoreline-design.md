# Design: Fix Scoreline Model & Deprecate GBM

**Date:** 2026-03-20
**Status:** Approved
**Author:** Memo (Letta Code)

---

## Problem Statement

The market_outcome_gbm model produces broken 1X2 probabilities:
- Trains each outcome (home, draw, away) as independent binary classifiers
- Probabilities don't sum to 100% (typical sum = 1.04)
- High-edge predictions were all wrong

The market_outcome_v2 (Scoreline) model has correct architecture but:
- Truncation at max_goals=10 loses 2-3% probability mass
- Only 562 fixtures have predictions (vs 8,015 for gbm)

---

## Solution

Deprecate market_outcome_gbm, fix and expand market_outcome_v2 (Scoreline).

---

## Design

### Fix 1: Increase max_goals from 10 to 15

**What:**
- Change max_goals parameter in inference code
- Truncation at 10 loses probability mass; at 15 captures 99.99%

**Files to modify:**
- src/modeling/v2/families/scoreline/derive_markets.py
- src/modeling/v2/families/scoreline/predict_scoreline.py

**Change:**
`python
# Before
max_goals = 10

# After
max_goals = 15
`

**Impact:**
- 1X2 probability sums: 0.97 -> 1.00
- No change to training (lambda models don't use the matrix)

---

### Fix 2: Expand Scoreline Coverage to 8,044 Fixtures

**What:**
- Run Scoreline prediction flow on all fixtures with lambda predictions
- Target: 8,044 fixtures (up from 562)

**Process:**
1. Query fixtures with lambda_xgb predictions but no market_outcome_v2
2. Build features (point-in-time safe)
3. Run predict_scoreline.py
4. Upsert to predictions table with model_name = 'market_outcome_v2'

**Approach:**
- Create a script: scripts/backfill_scoreline_predictions.py
- Batch processing (500 fixtures at a time)
- Progress logging

**Dependencies:**
- Requires lambda_xgb predictions to exist (already have 8,044)
- Requires team_premium_snapshots (already have 11,102 fixtures)

---

### Fix 3: Update Edge Calculator to Use Scoreline

**What:**
- Update scripts/calculate_edge.py to query market_outcome_v2
- Ignore market_outcome_gbm predictions

**Change:**
`python
# In calculate_edge.py
model_name = 'market_outcome_v2'  # Changed from 'market_outcome_gbm'
`

**Note:**
- gbm predictions remain in database for historical comparison
- No deletion, no deprecation flag needed

---

## Implementation Plan

### Phase 1: Fix Truncation
1. Locate max_goals parameter in derive_markets.py and predict_scoreline.py
2. Change from 10 to 15
3. Verify with unit test or manual calculation

### Phase 2: Backfill Predictions
1. Create backfill script
2. Query fixtures needing predictions
3. Run prediction flow in batches
4. Verify output

### Phase 3: Update Edge Calculator
1. Update model_name parameter
2. Run edge calculation
3. Compare results with previous

---

## Verification

### After Fix 1:
- Run Scoreline on sample fixture
- Check 1X2 sums to 1.00 (within 0.001)

### After Fix 2:
- Query predictions table for market_outcome_v2 count
- Should be ~8,044 fixtures

### After Fix 3:
- Run edge calculator
- Check edge values are reasonable (no more 40% edges)
- Run calibration check on historical predictions

---

## Risks

1. **Feature availability:** Some fixtures may be missing features
   - Mitigation: Log and skip fixtures with missing features

2. **Runtime:** 8,044 fixtures may take time
   - Mitigation: Batch processing, progress logging

3. **Data storage:** 8,044 fixtures x ~30 markets = ~240k predictions
   - Mitigation: Upsert will handle duplicates

---

## Success Criteria

1. Scoreline predictions for 8,000+ fixtures
2. 1X2 probabilities sum to 1.00 (+/- 0.001)
3. Edge calculator uses Scoreline only
4. Calibration check shows improvement (60-70% predictions win 60-70% of time)
