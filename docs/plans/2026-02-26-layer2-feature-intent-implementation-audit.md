# Layer 2 Feature Intent vs Implementation Audit

Date: 2026-02-26  
Scope: all 36 Layer 2 features (train + predict + DB timing evidence)

## Inputs Used
- Intent contract:
  - `docs/plans/2026-02-26-layer2-feature-intent-contract.md`
- Reconciliation matrix:
  - `artifacts/reports/layer2_reconciliation/layer2_feature_reconciliation_matrix.json`
- Code parity:
  - `artifacts/reports/layer2_reconciliation/layer2_feature_code_parity.json`
- Source timing checks:
  - `artifacts/reports/layer2_reconciliation/layer2_source_timing_checks.json`
- Full per-feature audit output:
  - `artifacts/reports/layer2_reconciliation/layer2_feature_intent_implementation_audit.json`
  - `artifacts/reports/layer2_reconciliation/layer2_feature_intent_implementation_audit.md`

## Headline Result
- Total features audited: `36`
- Implemented well: `1`
- Implemented with issues: `35`

## Why Most Features Were Marked "With Issues"
The dominant issue classes were:
1. `mixed_incremental_value` (23 features):
   - features are present and timed correctly, but controlled evidence is not stable enough to claim reliable incremental lift.
2. `drop_candidate` (9 features):
   - current controlled evidence says these degrade fit in their present form.
3. `source/readiness risk`:
   - derby features remain sparse.
   - availability-derived features still have upcoming-fixture coverage gaps.

## Critical Findings
1. Historical `player_availability` timing contamination was repaired:
   - `rows_after_kickoff = 0` (post-repair)
   - historical FT timing integrity is now clean.
   - This removes the prior hard timing blocker for injury-derived features.

2. Derby features remain structurally sparse:
   - `is_derby` prevalence too low for robust global learning.
   - `team_rivalries` remains critical risk.

3. Odds timing is clean, but upcoming coverage is still partial:
   - No post-kickoff chosen odds snapshots in training join.
   - Upcoming (14d) odds-gap feature population remains partial.

4. `home_lame_duck` and `away_lame_duck` are currently zero-variance in train frame.

5. Post-repair controlled comparison is a tradeoff:
   - full model: better home RMSE and broader enabled-league set.
   - keyabs-only model: better away RMSE but narrower enabled-league set.

## Decision Implication
Current Layer 2 can run controlled experiments, but feature-governance is still not ready for broad confidence claims.
The main blockers are:
1. Final keep/drop/defer stabilization for mixed-value feature set.
2. Explicit decision on full vs keyabs-only promotion objective (home+breadth vs away-only gain).
3. Upcoming availability population as a serving-readiness gate (not a historical training gate).

## Immediate Required Actions
1. Lock post-repair timing evidence as baseline for further Layer 2 decisions.
2. Run one governance pass to choose promotion objective:
   - prioritize home+breadth (`full`) or away-only RMSE gain (`keyabs-only`).
3. Keep derby features frozen as conditional/deferred until prevalence support improves.
4. Keep upcoming availability as a prediction-time population gate and monitor it outside historical training sign-off.
