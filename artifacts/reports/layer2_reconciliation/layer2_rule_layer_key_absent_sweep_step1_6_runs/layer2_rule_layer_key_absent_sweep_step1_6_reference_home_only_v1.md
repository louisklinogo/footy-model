# Layer 2 Rule-Layer Segmented Backtest

Generated: 2026-02-26T20:48:36.492873+00:00

## Coverage
- test fixtures: 1423
- triggered fixtures: 65
- home trigger events: 65
- away trigger events: 0

## Global RMSE
| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | ---: | ---: | ---: | ---: |
| layer1_only | 1.1963 | 1.0854 | - | - |
| layer1_plus_layer2 | 1.1949 | 1.0848 | 0.12% | 0.06% |
| layer1_plus_layer2_plus_rules | 1.1941 | 1.0848 | 0.18% | 0.06% |

## Triggered Segment
| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | ---: | ---: | ---: | ---: |
| layer1_only | 1.2410 | 1.2048 | - | - |
| layer1_plus_layer2 | 1.2366 | 1.2057 | 0.35% | -0.08% |
| layer1_plus_layer2_plus_rules | 1.2211 | 1.2057 | 1.60% | -0.08% |

## Directional Hit Rate (Triggered Side Events)
- home events: n=65, layer2=0.47692307692307695, layer2+rules=0.6153846153846154
- away events: n=0, layer2=None, layer2+rules=None

## Rule Counts (Home)
- congestion: 22
- key_absent: 50

## Rule Counts (Away)