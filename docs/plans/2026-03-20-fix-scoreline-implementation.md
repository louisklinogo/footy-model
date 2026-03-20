# Implementation Plan: Fix Scoreline Model

**Date:** 2026-03-20
**Design Doc:** docs/plans/2026-03-20-fix-scoreline-design.md

---

## Task List

### Task 1: Fix max_goals Truncation
**File:** `src/modeling/v2/families/scoreline/derive_markets.py`
**Change:** Update `max_goals` from 10 to 15
**Verification:** Run unit test or manual calculation

### Task 2: Fix max_goals in Prediction Script
**File:** `src/modeling/v2/families/scoreline/predict_scoreline.py`
**Change:** Update `max_goals` from 10 to 15
**Verification:** Run on sample fixture, check 1X2 sums to 1.0

### Task 3: Create Backfill Script
**File:** `scripts/backfill_scoreline_predictions.py` (new)
**Purpose:** Run Scoreline predictions on all fixtures with lambdas
**Logic:**
- Query fixtures with lambda_xgb but no market_outcome_v2
- Build features for each fixture
- Run predict_scoreline
- Upsert to predictions table

### Task 4: Run Backfill
**Command:** `python scripts/backfill_scoreline_predictions.py`
**Expected:** ~8,000 fixtures processed
**Time:** TBD based on batch size

### Task 5: Update Edge Calculator
**File:** `scripts/calculate_edge.py`
**Change:** Change default model_name to 'market_outcome_v2'

### Task 6: Verification
- Check predictions table count for market_outcome_v2
- Run calibration check on historical predictions
- Compare edge calculation results

---

## Execution Order

1. Task 1: Fix derive_markets.py (max_goals)
2. Task 2: Fix predict_scoreline.py (max_goals)
3. Task 3: Create backfill script
4. Task 4: Run backfill
5. Task 5: Update edge calculator
6. Task 6: Verification

---

## Rollback Plan

If issues arise:
1. Revert max_goals changes (git checkout)
2. Delete backfill script
3. Revert edge calculator change
4. No data deletion needed (upserts are idempotent)
