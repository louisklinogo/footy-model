# Layer 2 Data Source Audit

Generated: 2026-02-26T17:17:12.481661+00:00

- FT fixtures baseline: 7824
- Time span: 2023-09-19 16:45:00+00:00 -> 2026-12-03 00:00:00+00:00

| Source | Role | Rows | Fixture Coverage % | Timing Integrity % | Risk | Key Risks |
| --- | --- | ---: | ---: | ---: | --- | --- |
| fixtures+fixture_results | ground-truth target and chronology anchor | 7824 | 100.00 | 100.00 | low |  |
| predictions(lambda_xgb) | Layer 1 baseline lambdas for residual target | 20594 | 91.22 | 100.00 | low |  |
| team_premium_snapshots | rolling team-form features and rest/schedule context | 22204 | 100.00 | 100.00 | low |  |
| fixture_odds_markets(sofascore_1x2) | market-implied probabilities for odds-model gap features | 8947 | 97.83 | 100.00 | low |  |
| player_availability | injury/absence impact features | 338699 | 99.86 | 100.00 | low |  |
| team_rivalries | derby context features | 35 | 0.47 | 100.00 | critical | very_low_derby_prevalence |
| fixture_player_stats | historical player impact prior strength estimation | 319999 | 99.82 | 100.00 | high | expected_goals_missingness_high |

## Critical Sources
- team_rivalries

## High-Risk Sources
- fixture_player_stats

## Notes
- Timing integrity is measured against kickoff when timestamp lineage is available.
- Some sources do not encode immutable as-of timestamps; those are flagged in risk narrative when relevant.