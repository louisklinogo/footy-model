# Layer 2 Rule-Layer Segmented Backtest

Generated: 2026-02-26T19:01:34.980947+00:00

## Coverage
- test fixtures: 1423
- triggered fixtures: 14
- home trigger events: 11
- away trigger events: 10

## Global RMSE
| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | ---: | ---: | ---: | ---: |
| layer1_only | 1.1963 | 1.0854 | - | - |
| layer1_plus_layer2 | 1.1954 | 1.0850 | 0.08% | 0.04% |
| layer1_plus_layer2_plus_rules | 1.1956 | 1.0850 | 0.06% | 0.04% |

## Triggered Segment
| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | ---: | ---: | ---: | ---: |
| layer1_only | 1.2536 | 0.9176 | - | - |
| layer1_plus_layer2 | 1.2441 | 0.9310 | 0.76% | -1.46% |
| layer1_plus_layer2_plus_rules | 1.2626 | 0.9303 | -0.72% | -1.38% |

## Directional Hit Rate (Triggered Side Events)
- home events: n=11, layer2=0.45454545454545453, layer2+rules=0.36363636363636365
- away events: n=10, layer2=0.4, layer2+rules=0.4

## Rule Counts (Home)
- congestion: 11

## Rule Counts (Away)
- congestion: 10