# AUC Lift Feature-Block Plan

Date: 2026-03-05
Window: last 180 days
Source: `diagnostic_report.json`

## Weak Segment Readout (sample-filtered)

- Market-level weak AUC: `c85`, `dc_12`, `1x2_d`, `ao15`, `a_1up`, `ho15`, `h_1up`
- League-level weak AUC: `BR1`, `JP1`, `IE1`, `SC3`, `SC2`
- Market+league tails: `dc_12|F2`, `1x2_d|F2`, `c85|E2`, `c85|E3`
- Edge-band tails are concentrated in `o15` low-edge buckets (`0-2%`, `2-4%`, `4-6%`)

## Candidate Feature Blocks (Mapped to Existing Dataset Fields)

1. Corners attack-intensity block (for `c85/c95/c105`, home/away corners ladders)
- Fields:
  - `home_rolling_xg`, `away_rolling_xg`
  - `home_rolling_sot`, `away_rolling_sot`
  - `home_rolling_big_chances`, `away_rolling_big_chances`
  - `xg_net_diff`, `goal_diff_proxy`
- Status: executed in first experiment.

2. Scoreline draw/variance block (for `1x2_d`, `dc_12`)
- Fields:
  - `home_rolling_xg_against`, `away_rolling_xg_against`
  - `home_rolling_possession`, `away_rolling_possession`
  - `style_delta`
  - `home_rolling_xgot_against`, `away_rolling_xgot_against`
- Status: pending.

3. Anytime momentum-path block (for `h_1up/a_1up`)
- Fields:
  - `home_rolling_xg_p1`, `away_rolling_xg_p1`
  - `home_rolling_sot_p1`, `away_rolling_sot_p1`
  - `home_rolling_sot_h2_delta`, `away_rolling_sot_h2_delta`
  - `recent_vs_baseline_gap`, `regressed_xg_diff`
- Status: pending.

## Experiment 1 Result (Corners Attack-Intensity Block)

- Contract: `model_v2/feature_contracts/experiments/corners_attack_block.yaml`
- Compare report: `artifacts/reports/v2_feature_block_experiments/20260305T025316Z/compare_corners_attack_block.md`
- Summary:
  - `markets_compared=12`
  - `auc_delta_mean=+0.000385`
  - `brier_delta_mean=-0.000202`
  - `auc_up_markets=6`
  - `brier_down_markets=9`
- Decision:
  - Keep as candidate, but lift is small and mixed across tail markets (`c85` lift is near zero).
  - Prioritize Experiment 2 on draw/variance block for stronger upside in low-AUC scoreline segments.
