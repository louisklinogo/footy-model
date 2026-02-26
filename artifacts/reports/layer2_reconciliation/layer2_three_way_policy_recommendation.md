# Layer 2 Three-Way Policy Recommendation

Generated: 2026-02-26T21:00:16.233281+00:00

## Decision
- Global model: keep `baseline_full_features` (do not promote pruned_drop9).
- Rule layer: promote `keyabs_home_only_minus0.120` deterministic overrides on top of full Layer 2.

## Global Three-Way (Holdout RMSE)
| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | ---: | ---: | ---: | ---: |
| layer1_only | 1.1963 | 1.0854 | - | - |
| layer1_plus_layer2 | 1.1949 | 1.0848 | 0.12% | 0.06% |
| layer1_plus_layer2_plus_rules(keyabs_home_minus0.12) | 1.1939 | 1.0848 | 0.20% | 0.06% |

## Triggered Segment
| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | ---: | ---: | ---: | ---: |
| layer1_only | 1.2163 | 1.1369 | - | - |
| layer1_plus_layer2 | 1.2148 | 1.1317 | 0.13% | 0.46% |
| layer1_plus_layer2_plus_rules(keyabs_home_minus0.12) | 1.1866 | 1.1317 | 2.45% | 0.46% |

## Why this rule config
- Key-absence-only sweep selected `-12%` as best passing variant by triggered-home and global-home deltas.
- Sweep artifact: `artifacts/reports/layer2_reconciliation/layer2_rule_layer_key_absent_sweep_step1_6.md`.
- Confirmatory backtest artifact: `artifacts/reports/layer2_reconciliation/layer2_rule_layer_segmented_backtest_full_keyabs_home_minus012_v1.md`.

## Why full over pruned_drop9
- Pruned candidate reduced away RMSE but worsened home RMSE and dropped enabled league `E1`.
- Comparison artifact: `artifacts/reports/layer2_reconciliation/layer2_pruned_drop9_vs_full_comparison.md`.

## Operational Notes
- Active rule config: `model_artifacts/situational_model/rule_layer_config.json`.
- Freeze remains active until calibration sample and upcoming-availability serving gates are cleared.
