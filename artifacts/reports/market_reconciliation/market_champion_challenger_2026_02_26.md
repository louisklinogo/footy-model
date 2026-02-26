# Market Champion-Challenger Decision (2026-02-26)

- champion commit: `deecc4b`
- challenger commit: `33d2341`
- markets compared: 22
- feature count: 54 -> 60

## New Features in Challenger
- adj_lambda_away_final
- adj_lambda_home_final
- lambda_away_l1
- lambda_home_l1
- rule_fired_away
- rule_fired_home

## Aggregate Deltas (challenger - champion)
- avg AUC delta: 0.007738
- avg Brier delta: -0.002129 (lower is better)
- avg Accuracy delta: 0.002558
- away-market AUC delta: -0.001901
- away-market Brier delta: -0.002801
- away-market Accuracy delta: 0.003322
- min test sample across markets: 1499

## Gate Results
- global_non_negative_quality: PASS
- away_degradation_within_tolerance: PASS
- sample_gate_met: PASS

## Decision: **PROMOTE**

## Market-Level Deltas
| Market | AUC delta | Brier delta | Accuracy delta | test_n |
| --- | ---: | ---: | ---: | ---: |
| 1x2_a | 0.001850 | -0.004119 | 0.018943 | 1567 |
| 1x2_d | 0.040145 | -0.001257 | 0.000896 | 1567 |
| 1x2_h | 0.003489 | -0.000380 | -0.028092 | 1567 |
| ao15 | -0.026937 | 0.002701 | 0.008272 | 1567 |
| away_and_o25 | -0.006314 | -0.004781 | 0.005192 | 1567 |
| away_or_o15 | 0.019317 | -0.009880 | 0.014561 | 1567 |
| away_or_o25 | -0.002779 | -0.000391 | 0.000206 | 1567 |
| btts | 0.036695 | -0.003906 | 0.017789 | 1567 |
| c85 | -0.014430 | 0.005245 | -0.017292 | 1499 |
| dc_12 | 0.037729 | -0.001070 | 0.000896 | 1567 |
| dc_1x | 0.004373 | -0.004182 | 0.021920 | 1567 |
| dc_x2 | 0.003455 | -0.000333 | -0.027241 | 1567 |
| ho15 | 0.017279 | 0.000352 | -0.000892 | 1567 |
| home_and_o25 | 0.000906 | 0.007318 | -0.018036 | 1567 |
| home_or_o15 | 0.003043 | -0.008037 | 0.011185 | 1567 |
| home_or_o25 | -0.008051 | -0.001931 | 0.006612 | 1567 |
| o15 | 0.002451 | -0.011058 | 0.022176 | 1567 |
| o25 | 0.009353 | -0.003191 | 0.005026 | 1567 |
| o35 | 0.008453 | 0.001387 | -0.004871 | 1567 |
| o45 | 0.028038 | 0.004866 | -0.008175 | 1567 |
| u15 | 0.003294 | -0.011018 | 0.022176 | 1567 |
| u25 | 0.008877 | -0.003172 | 0.005026 | 1567 |
