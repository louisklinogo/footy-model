# Layer 2 Data Source Audit Review (Step 3)

Date: 2026-02-26  
Scope: all Layer 2 training and inference dependencies  
Primary artifact: `artifacts/reports/layer2_reconciliation/layer2_data_sources_audit.json`

## Update (Post-Repair)
This Step 3 note captured the pre-repair state. After the timing repair run (`player_availability_event_timing_repair_20260226T171652Z`), historical FT availability timing is now clean (`rows_after_kickoff = 0`).

Current timing reference artifacts:
- `artifacts/reports/layer2_reconciliation/layer2_source_timing_checks.json`
- `artifacts/reports/layer2_reconciliation/layer2_data_sources_audit.json`

Current remaining availability risk is upcoming-fixture population coverage, not historical timing leakage.

## Objective
Audit every source Layer 2 depends on for:
- coverage
- timing integrity relative to kickoff
- practical fit for decision-grade modeling

## Summary Verdict
Layer 2 source quality is mixed:
- strong: `fixtures+fixture_results`, `team_premium_snapshots`, `fixture_odds_markets`
- primary critical blocker: `team_rivalries` utility sparsity
- improved from prior run: `predictions(lambda_xgb)` lineage now repaired to decision-grade timing integrity
- improved from prior run: `player_availability` historical timing repaired to pre-kickoff-safe semantics
- contextual/low utility: `team_rivalries` (very low prevalence)
- caution: `fixture_player_stats` has high `expected_goals` missingness

## Source-by-Source Findings

### 1) `fixtures + fixture_results`
- Coverage: `100.00%` (`7824/7824` FT baseline)
- Timing integrity: `100.00%`
- Risk: `low`
- Assessment: suitable as target and chronology anchor.

### 2) `predictions (lambda_xgb)`  **Repaired**
- Fixture coverage: `88.27%` (`6906/7824` latest pair availability)
- Timing integrity (latest pair pre-kickoff): `100.00%` (`6906/6906`)
- Post-kickoff latest pairs: `0/6906` (`0.00%`)
- Risk: `medium`
- Lineage anchor used: `feature_asof_utc`
- Notes:
  - schema + writer changes are active.
  - legacy backfill set `feature_asof_utc = kickoff - 60m` for existing `lambda_xgb` rows.

### 3) `team_premium_snapshots`
- Fixture coverage: `100.00%` home/away pair coverage on FT set
- Timing integrity: table has `built_at`, but no immutable source decision timestamp per feature row.
- Risk: `low`
- Assessment: operationally strong; timestamp lineage can still be improved for strict PIT audits.

### 4) `fixture_odds_markets (sofascore 1x2)`
- Fixture coverage (pre-kickoff eligible rows): `84.86%` (`6638/7822`)
- Timing integrity for selected pre-match types: `100.00%` (no post-kickoff rows in `latest_pre_match/closing` join scope)
- Risk: `medium`
- Assessment: usable, but coverage gaps still matter for stability.

### 5) `player_availability`  **Repaired timing / serving gap remains**
- Historical fixture coverage (FT fixtures with availability rows): `99.86%` (`7813/7824`)
- Event-timing integrity vs kickoff (historical FT rows): `100.00%` (`0` late rows)
- Upcoming 14d coverage: `0.00%` (`0/630`)
- Risk:
  - historical timing leakage: `low` (repaired)
  - upcoming serving population: `high`
- Impact:
  - injury-feature historical training is no longer blocked by timing contamination.
  - prediction-time injury features remain weak until upcoming coverage is populated.

### 6) `team_rivalries`
- Feature prevalence on FT fixtures: `0.47%`
- Risk: `critical` for utility (not for leakage)
- Assessment: signal is too sparse for stable global learning; keep as optional/experimental feature only.

### 7) `fixture_player_stats`
- Fixture coverage: `99.85%` (`7810/7822`)
- `expected_goals` null rate: `71.67%`
- Risk: `high`
- Assessment: table is broad, but the specific metric feeding injury impact priors is sparse; this weakens reliability of player-impact estimates.

## Decisions From Step 3
1. Remove `predictions(lambda_xgb)` as a critical lineage blocker (now repaired).
2. Keep `team_rivalries` as a critical utility blocker for global governance decisions.
3. Treat `player_availability` as a serving-readiness population gate (upcoming fixtures), not a historical timing blocker.

## Required Follow-Up (before final feature reconciliation)
1. Populate upcoming pre-kickoff `player_availability` rows under repaired writer semantics.
2. Re-run source audit and feature ledger after each availability backfill batch.
3. Keep availability timing gates explicit in every validation cycle (expect `0` late rows).
