# Layer 2 Rule-Layer Segmented Backtest

Generated: 2026-02-26T20:50:09.972947+00:00

## Coverage
- test fixtures: 1423
- triggered fixtures: 50
- home trigger events: 50
- away trigger events: 0

## Global RMSE
| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | ---: | ---: | ---: | ---: |
| layer1_only | 1.1963 | 1.0854 | - | - |
| layer1_plus_layer2 | 1.1949 | 1.0848 | 0.12% | 0.06% |
| layer1_plus_layer2_plus_rules | 1.1945 | 1.0848 | 0.15% | 0.06% |

## Triggered Segment
| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | ---: | ---: | ---: | ---: |
| layer1_only | 1.2163 | 1.1369 | - | - |
| layer1_plus_layer2 | 1.2148 | 1.1317 | 0.13% | 0.46% |
| layer1_plus_layer2_plus_rules | 1.2052 | 1.1317 | 0.92% | 0.46% |

## Directional Hit Rate (Triggered Side Events)
- home events: n=50, layer2=0.44, layer2+rules=0.52
- away events: n=0, layer2=None, layer2+rules=None

## Rule Counts (Home)
- congestion: 7
- key_absent: 50

## Rule Counts (Away)