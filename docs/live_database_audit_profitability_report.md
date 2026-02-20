# Live Database Audit and Profitability Assessment

Date: 2026-02-20

NOTE: This file is a historical audit log. Some earlier metrics (e.g., snapshot counts and league coverage) were superseded by later rebuilds and policy decisions made the same day.

For the current source of truth on the new system design, see:
- docs/new_model_blueprint.md

## Addendum - Why `team_premium_snapshots` only has E0

After deeper due diligence against `pipeline_runs` and script code:

- This is not caused by `ingest_premium_fixtures_v1.py`. Ingest did run for multiple leagues (E1, E2, E3, EC, SC0, SC1, SC2, D1, D2, I1).
- `team_premium_snapshots` is populated only by `scrapers/build_team_premium_snapshots_v1.py`, not by ingest.
- Logged snapshot-builder runs are E0-only (`details_json.league = 'E0'` for all recorded runs).
- Logged daily snapshot steps are also E0-only (`daily.build_snapshots.E0`).
- A multi-league daily run was in progress through ingest, but had not reached snapshot stage for later leagues (the run stalled during `daily.premium_enrich.E3`).

Conclusion: only E0 snapshots exist because snapshot build execution scope/history is E0-only so far, not because the snapshot script is hardcoded to E0.

## SECTION 1 - Database Audit Findings

- Core schema is present and coherent for a fixtures-first system (fixtures, results, premium stats, odds snapshots, team snapshots, predictions, scores).
- Scale/depth:
  - `fixtures`: 9,800 rows (`2025-07-11` -> `2026-12-03`)
  - `fixture_stats_premium`: 3,740
  - `fixture_odds_snapshots`: 4,049
  - `team_premium_snapshots`: 760
  - `predictions`: 18, `prediction_scores`: 3
- Critical coverage reality:
  - `team_premium_snapshots` only exists for E0 (380 fixtures, 760 team rows).
  - `fixture_results` only 2,560 rows while `fixtures.status='ft'` is 6,244 (3,684 FT fixtures have no result row).
- Timestamp integrity:
  - Odds timing check passed: `post_kickoff=0`, `pre_kickoff=4049`.
- Major operational blocker:
  - In `latest_pre_match`, odds JSON is mostly empty: only `167 / 1220` non-empty (~13.7%).
  - `one_x_two_json` is empty in all snapshots (0 non-empty).

## SECTION 2 - Feature Quality Assessment

- High predictive value (when present):
  - Rolling team snapshots (`rolling_xg`, `rolling_xg_against`, etc.) - currently E0-only.
  - Pre-match O/U and AH odds snapshots - currently sparse in `latest_pre_match`.
- Medium:
  - Premium event stats (`xg/xgot/possession/sot/corners`) with fidelity weighting.
- Low:
  - Current model outputs are heavily saturated (`p_model >= 0.99` in 15/18 predictions), suggesting low practical discrimination at this stage.
- Dangerous / leakage-prone:
  - Placeholder premium rows: 1,764 rows have `h_xg/a_xg = NULL`; 31 rows have `h_xg=a_xg=0` ingested long before kickoff.
  - `built_at` for team snapshots is after kickoff for 522/760 rows (not automatic leakage, but point-in-time reproducibility must be proven).
  - Scoring table cannot evaluate EV yet: `odds_used`, `edge`, `roi_unit` are null for all 3 scored rows.

## SECTION 3 - Recommended Modeling Approach

- Why xG/intensity foundation:
  - Expected goals (`lambda_home`, `lambda_away`) gives one coherent latent state from which all markets are derived; this prevents contradictory market probabilities.
- Best architecture:
  - Hybrid: gradient boosting (feature mapping) -> bivariate goal distribution (Poisson + Dixon-Coles low-score correction) -> market probabilities.
- Practical status with current DB:
  - Not data-ready for profitable multi-market deployment.
  - Immediate realistic scope is a narrow E0 pilot after data repairs.

## SECTION 4 - Feature Engineering Plan

- Team strength:
  - `att_i = EWMA_N(xg_for_i)`, `def_i = EWMA_N(xg_against_i)` with N in {5,10,20}.
- Matchup:
  - `mu_home = exp(b0 + H + att_home - def_away + rest_adj + lineup_adj)`
  - `mu_away = exp(b0 + att_away - def_home + rest_adj + lineup_adj)`
- Form/trend:
  - Short-long deltas: `EWMA_5 - EWMA_20` for xG/xGA/SOT/pace proxies.
- Schedule/fatigue:
  - `days_rest`, `matches_7d`, `matches_21d`.
- Market-derived:
  - De-vig implied probabilities, line movement, cross-book dispersion (only when populated at decision time).
- Hard rule:
  - Zero/null placeholders must be excluded or explicitly flagged; never let them pass as true signal.

## SECTION 5 - Market Targeting Strategy

- With current data quality, no market is truly production-ready for +EV sizing.
- Post-fix priority:
  1. Over/Under 2.5
  2. Asian Handicap
  3. BTTS
  4. Draw No Bet
  5. 1X2
- Why this order now:
  - O/U and AH structures exist in data.
  - 1X2 input is currently absent (`one_x_two_json` empty), so 1X2 should not be prioritized yet.

## SECTION 6 - Validation and Backtesting Framework

- Mandatory split protocol:
  - Rolling walk-forward by decision timestamp.
  - Purge by `fixture_id` to prevent same-fixture contamination.
- Mandatory metrics:
  - Primary: CLV, realized EV after vig/slippage/limits.
  - Secondary: log loss, Brier, calibration slope/intercept.
- Current status:
  - Not sufficient for profitability inference (`prediction_scores=3`, all EV fields null).
  - CLV/ROI cannot yet be validated with integrity.

## SECTION 7 - Profitability Optimization Strategy

- Bet selection:
  - Bet only when `EV_net > threshold`, where EV includes vig, expected slippage, and rejection risk.
- Sizing:
  - Start fractional Kelly 0.10-0.25 with strict caps (per match/per league/per day).
- Risk controls:
  - Correlation cap across same match and same league window.
  - Drawdown-triggered stake haircut and auto-disable under calibration drift.
- Promotion gates (hard):
  - Sustained positive CLV, calibration near 1.0 slope, and stable out-of-sample EV by league/market.

## SECTION 8 - Deployment Architecture

- Required production chain:
  - DB (point-in-time) -> feature store (`as_of_ts`) -> intensity model -> score distribution -> market probabilities -> de-vig EV -> bet policy -> execution logs -> CLV/settlement feedback.
- Current gap:
  - Execution/odds-tracking layer exists structurally but is not populated enough to compute real edge (`odds_used/edge/roi_unit` null).

## SECTION 9 - Implementation Roadmap (Step-by-step)

1. Data hygiene pass:
   - Remove/flag placeholder premium rows (`NULL` and `0,0` pseudo-stats) from training features.
2. Coverage expansion:
   - Build `team_premium_snapshots` beyond E0 (current single-league bottleneck).
3. Odds completeness:
   - Fix `latest_pre_match` JSON extraction/fill rate and populate 1X2 odds.
4. Results completeness:
   - Backfill `fixture_results` for all FT fixtures.
5. Point-in-time contract:
   - Persist `as_of_ts` and prove `source_ts <= as_of_ts` for every feature.
6. Modeling refactor:
   - Move from per-market classifiers toward intensity + distribution coherence.
7. Evaluation upgrade:
   - Walk-forward + CLV tracking + realistic execution simulation.
8. Pilot:
   - Single-league, single-market controlled deployment; scale only after CLV-positive stability.

## Executive Bottom Line

You have a strong skeleton, but the current bottleneck is data readiness and execution-grade validation, not model sophistication.

## 2026-02-20 Update - Post Snapshot Rebuild and New-Model Direction

### What changed

- `team_premium_snapshots` was rebuilt for: `E1`, `E2`, `E3`, `EC`, `SC0`, `SC1`, `SC2`, `D1`, `D2`, `I1` (in addition to existing `E0`).
- Root cause of E0-only snapshots: execution scope/history (snapshot builder had only been run for `E0`), not ingest failure.

### Revised readiness view

- Strong fixture readiness for a new model build now exists in: `E0`, `E1`, `E2`, `E3`, `D1`, `D2`, `I1`, `SC0`.
- `EC`, `SC1`, `SC2` remain low-quality for xG-driven modeling (`xG missingness ~100%`, fidelity near zero) despite snapshot presence.
- `one_x_two_json` remains empty in odds snapshots, so direct 1X2 market modeling should remain deferred.
- Browser validation confirms pre-match lineups and absence lists are available on many matches (e.g., `t8wgAYtf`, `C8gZTMv1`, `fXEfnyLt`) via summary and lineups pages, but coverage is inconsistent by league and timing.

### Lineup/injury/suspension reliability stance

- Treat lineup and absence data as optional alpha, not mandatory core input.
- Dependable when present (players, formation, and reasons like `Injury`, `Yellow Cards`, `Inactive` can be extracted), but not dependable for complete coverage at all decision times.
- Production policy:
  1. Keep core model independent of lineup/absence availability.
  2. Add a conditional adjustment layer only when lineup/absence payload passes quality checks.
  3. Track `lineup_data_available` and `absence_data_available` per prediction.
  4. Use coverage thresholds (for example, 70%+ in bet window) before promoting lineup-based adjustments to live sizing logic.

### Advice if building a new model from scratch

1. Ignore the legacy model artifacts and retrain from a clean pipeline.
2. Start with an xG-intensity-first architecture (`lambda_home`, `lambda_away`) and derive market probabilities from score distributions.
3. First deployment scope: `E0`, `E1`, `E2`, `E3`, `D1`, `D2`, `I1`, `SC0`.
4. First live markets: O/U 2.5, then AH. Add BTTS/DNB later; defer 1X2 until 1X2 odds data is reliably populated.
5. Use strict rolling time validation and CLV-first promotion gates.

### Immediate next actions

1. Retrain new baseline models on refreshed multi-league snapshots.
2. Add point-in-time feature contract checks (`source_ts <= as_of_ts`) into training and prediction data pulls.
3. Implement EV/CLV logging in scoring before any serious deployment decisions.
4. Add lineup/absence scraping and ingestion as a separate optional feature stream, with null-safe handling and monitoring dashboards for coverage by league and hours-to-kickoff.

## 2026-02-20 Addendum - Concern Resolutions (Decision Policy)

### 1) Upcoming odds appear late

- Treat this as a design constraint, not an exception.
- Run windowed decision models (example: `T-60`, `T-15`, `T-5`) with strict as-of feature snapshots.
- If required odds are missing at a window cutoff, mark fixture as `not tradable` for that window.
- Do not forward-fill from later snapshots into earlier windows.

### 2) H2H and segmentation/clustering

- H2H should be optional and low weight, only with recency decay and minimum sample thresholds.
- Prioritize stable segmentation (league/season/liquidity/strength buckets) before advanced clustering.
- Keep segmentation features only if they improve out-of-sample CLV/ROI across multiple seasons.

### 3) Psychological and match context

- Include only objective, timestamped proxies: rest, congestion, travel, lineup/absence availability, and rule-based stakes pressure.
- Exclude narrative/manual motivation tags from production features unless they can be measured consistently.
- Enforce coverage and stability gates before promotion to live sizing.

### 4) 1X2 1UP and odds requirements

- You can bootstrap 1UP fair pricing from O/U + AH + score-distribution modeling.
- 1X2 odds are still recommended as a medium-priority data upgrade to better identify draw vs win decomposition and improve calibration.
- 1UP settlement must be modeled per bookmaker rule-set (trigger conditions and exceptions), not as a generic market.

### Production go/no-go policy

- Keep features only if all are true:
  1. Coverage passes threshold in target decision window.
  2. No leakage against declared as-of timestamp.
  3. Out-of-sample CLV/ROI lift is stable across leagues/seasons.
- If any condition fails, demote feature to research-only.
