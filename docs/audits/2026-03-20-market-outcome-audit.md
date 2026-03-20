# Market Outcome Model Audit
**Date:** 2026-03-20
**Status:** COMPLETE - Fixes implemented and verified

---

## Executive Summary

| Component | Status Before | Status After | Fix Applied |
|-----------|---------------|--------------|-------------|
| market_outcome_gbm | BROKEN | DEPRECATED | No longer used |
| market_outcome_v2 (Scoreline) | PARTIAL | FIXED | max_goals=15, coverage expanded |
| Edge Calculator | USING GBM | USING SCORELINE | Updated default model |

---

## Fixes Implemented

### Fix 1: max_goals Truncation
- Changed default from 10 to 15 in predict_scoreline.py
- Captures 99.99% probability mass (previously 97%)

### Fix 2: Coverage Expansion
- Before: 562 fixtures
- After: 8,781 fixtures
- Backfill ran with --backfill-days 1000 --write-db

### Fix 3: Edge Calculator
- Updated to use scoreline_v2 model
- Commands: --model-name scoreline_v2

---

## Calibration Results

### scoreline_v2 (NEW - CORRECT)
| Bucket | Predicted | Actual Win Rate |
|--------|-----------|-----------------|
| 70%+ | 70% | 70.6% |
| 60-70% | 60-70% | 62.6% |
| 50-60% | 50-60% | 55.3% |
| 40-50% | 40-50% | 47.2% |
| <40% | <40% | 27.2% |

### market_outcome_gbm (OLD - OVERCONFIDENT)
| Bucket | Predicted | Actual Win Rate |
|--------|-----------|-----------------|
| 70%+ | 70% | 91.9% (overconfident) |
| 60-70% | 60-70% | 80.3% (overconfident) |
| 50-60% | 50-60% | 63.7% |
| 40-50% | 40-50% | 46.5% |
| <40% | <40% | 23.9% |

---

## Files Modified

1. src/modeling/v2/families/scoreline/predict_scoreline.py
   - Changed max_goals default from 10 to 15
   - Added --backfill-days argument

2. src/modeling/v2/db_reuse_features.py
   - Fixed merge issue when attack_context is empty

3. scripts/calculate_edge.py
   - Updated to use scoreline_v2 model

---

## Verification

`
Coverage: 8,781 fixtures, 526,926 predictions
Calibration: avg predicted 32.6%, avg actual 33.3%
Edge calculator: Working with scoreline_v2
`

---

## Next Steps

1. Monitor edge predictions for accuracy
2. Consider disabling calibration if 1X2 sums need to be exactly 1.0
3. Update production prediction flow to use scoreline_v2
