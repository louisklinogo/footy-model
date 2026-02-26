# Layer 2 Feature Reconciliation Review (Step 4)

Date: 2026-02-26  
Inputs:
- `artifacts/reports/layer2_reconciliation/layer2_feature_ledger.json`
- `artifacts/reports/layer2_reconciliation/layer2_feature_reconciliation_matrix.json`

## Update (Post-Repair)
This Step 4 review reflects an earlier snapshot before post-repair reruns. Current governance should use:
- `artifacts/reports/layer2_reconciliation/layer2_feature_intent_implementation_audit.json`
- `artifacts/reports/layer2_reconciliation/layer2_feature_intent_implementation_audit.md`
- `artifacts/reports/layer2_reconciliation/layer2_postrepair_full_vs_keyabs_comparison.md`

Those artifacts include repaired availability timing and the latest full-vs-keyabs tradeoff.

## Summary
Using full-feature vs baseline holdout diagnostics:
- Baseline RMSE (home/away): `1.1172 / 0.9980`
- Full-feature RMSE (home/away): `1.1446 / 1.0413`
- Net: full feature set still underperforms baseline.

Feature decision split:
- `keep_candidate`: 10
- `conditional_candidate`: 17
- `drop_candidate`: 2
- `fix_lineage_before_judgment`: 7

## Current Keep Candidates (provisional)
- `home_rolling_xg_against`
- `away_rolling_xg`
- `away_rolling_xg_against`
- `rest_delta`
- `points_gap`
- `home_form_streak`
- `home_xg_over_conceded`
- `away_xg_over_scored`
- `away_xg_over_conceded`
- `away_rolling_corners`

## Current Drop Candidates (provisional)
- `odds_model_gap_draw`
- `odds_opening_gap_draw`

## Deferred Until Lineage Repair
All features depending on:
- `player_availability` timestamp semantics
- `team_rivalries` sparse prevalence context

Deferred features include:
- `is_derby`, `derby_position_gap`
- `home_xg_lost`, `away_xg_lost`, `home_key_absent`, `away_key_absent`, `injury_impact`

## Decision
Do not finalize injury/rivalry-driven feature governance until sufficient new pre-kickoff availability data is collected under repaired lineage semantics.
