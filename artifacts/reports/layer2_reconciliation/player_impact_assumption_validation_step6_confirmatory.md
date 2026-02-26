# Player-Impact Assumption Validation

Generated: 2026-02-26T20:31:09.374117+00:00

## Gate Status
- correctness: `pass`
- football_logic: `warn`
- predictive_value: `warn`
- operational_readiness: `fail`
- overall_status: `fail`

## Correctness
- rows compared (train vs predict formula): 7824
- timing late rows: 0 (column: event_recorded_at)
- parity home_xg_lost: {'mismatch_count_abs_gt_1e-9': 0, 'max_abs_diff': 0.0}
- parity away_xg_lost: {'mismatch_count_abs_gt_1e-9': 0, 'max_abs_diff': 0.0}
- parity home_key_absent: {'mismatch_count': 0, 'max_abs_diff': 0}
- parity away_key_absent: {'mismatch_count': 0, 'max_abs_diff': 0}

## Football Logic
- corr(xg_lost, residual): 0.0217
- residual delta (key_absent=1 minus key_absent=0): 0.0080
- triggered share: 28.52%

## Predictive Value
- best by triggered lift: key_absent_only (triggered_combined_lift=0.07%, global_combined_lift=0.01%)
- best by global lift: key_absent_only (triggered_combined_lift=0.07%, global_combined_lift=0.01%)

## Operational Readiness
- upcoming coverage (14d): 46/614 (7.49%)
- historical FT coverage: 7813/7824 (99.86%)
- fixture_player_stats expected_goals null pct: 71.67%

## Recommendation
- Keep player-impact features as conditional for serving until upcoming availability coverage improves.