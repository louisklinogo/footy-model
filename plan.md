# Revised Plan: Small-Bankroll Strategy with Full Corners Market Expansion

## Summary
Update the plan so corners are modeled and evaluated the way real books structure them: not just one `8.5` line, but match totals + home team + away team ladders, with settlement-consistent logic and bankroll-protective gating.

This keeps the core objective intact:
- high prediction quality,
- minimized stake loss,
- aggressive-but-controlled profit seeking.

## Scope
1. Keep current linchpin market families.
2. Expand corners from single `c85` to full phase-1 ladder:
- Match total corners O/U anchor lines: `7.5`, `8.5`, `9.5`, `10.5`
- Home team corners O/U anchor lines: `2.5`, `3.5`, `4.5`, `5.5`
- Away team corners O/U anchor lines: `2.5`, `3.5`, `4.5`, `5.5`
3. Phase-1 sides:
- Model/store Over side explicitly
- Derive Under side as `1 - Over` in evaluation/export layers
4. Phase-2 corners (deferred):
- corner handicap / most corners
- quarter/whole-line Asian corner handling

## Market Interface Changes
Add canonical over-market codes:
- Match totals: `c75`, `c85`, `c95`, `c105`
- Home team corners: `hc25`, `hc35`, `hc45`, `hc55`
- Away team corners: `ac25`, `ac35`, `ac45`, `ac55`

Derived under aliases are reporting-only in phase 1.

## Data and Settlement Rules
Use these rules consistently in training, scoring, and backtests:
1. Settlement window: 90 minutes + stoppage only.
2. Metric counted: corners taken.
3. Void handling: if fixture is abandoned/postponed/cancelled before settlement, mark non-scorable.
4. Inputs:
- Match corners from `fixture_stats_premium` (`h_corners`, `a_corners`)
- Team-corner targets:
  - home team = `h_corners`
  - away team = `a_corners`
- Match total corners = `h_corners + a_corners`

## Implementation Plan
1. Expand training targets in `src/modeling/layer2_markets/market_outcome_calibrator.py`.
- Add target generation for new corner lines.
- Add the new corners markets to training market list and metrics output.

2. Expand prediction in `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py`.
- Include new corners market codes in `MARKETS`.
- Add fallback probability logic for each corners line.
- Persist new market predictions to `predictions`.

3. Expand scoring in `src/modeling/evaluation/score_market_outcomes_fixtures_first.py`.
- Settle all new corners markets from premium stats.
- Keep missing-input behavior as `None` and skip scoring for those rows.

4. Expand risk mapping in `src/modeling/evaluation/assess_prediction_risk.py`.
- Add odds resolver mapping for new corners market codes.
- Keep robust fallback when no matching odds market/line exists.

5. Expand export in `src/modeling/export/export_market_outcomes_fixtures_first.py`.
- Add columns and SQL pivots for all new corners over markets.

6. Add small-bankroll policy constraints in risk assessment.
- Minimum rolling precision gate
- 2-loss streak pause guard
- Stake cap control through policy defaults

## Acceptance Criteria
1. Prediction coverage:
- every eligible fixture emits the expanded corners markets.
2. Scoring correctness:
- settlement thresholds match each line exactly.
3. Policy gating:
- risk layer can force pass for low-precision or active loss-streak markets.
4. Export completeness:
- CSV includes all new corners markets.

## Tests
1. Unit tests:
- corners threshold settlement (`c75/c85/c95/c105`, `hc*`, `ac*`).
- fallback generation stays bounded [0.001, 0.999].
2. Integration tests:
- predict -> score -> risk -> export with new corners codes.
3. Regression:
- existing non-corners markets continue to behave unchanged.

## Assumptions and Defaults
- Small bankroll constraints remain primary.
- Over-side modeled explicitly; under derived where needed.
- Initial ladders are fixed as above for sample stability.
- Asian quarter/whole-line corner handling is phase 2.
