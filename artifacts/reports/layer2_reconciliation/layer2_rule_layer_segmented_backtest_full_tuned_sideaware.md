# Layer 2 Rule-Layer Segmented Backtest

Generated: 2026-02-26T19:03:56.053258+00:00

## Coverage
- test fixtures: 1423
- triggered fixtures: 77
- home trigger events: 65
- away trigger events: 66

## Global RMSE
| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | ---: | ---: | ---: | ---: |
| layer1_only | 1.1963 | 1.0854 | - | - |
| layer1_plus_layer2 | 1.1949 | 1.0848 | 0.12% | 0.06% |
| layer1_plus_layer2_plus_rules | 1.1942 | 1.0855 | 0.18% | -0.01% |

## Triggered Segment
| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | ---: | ---: | ---: | ---: |
| layer1_only | 1.1984 | 1.1523 | - | - |
| layer1_plus_layer2 | 1.1892 | 1.1507 | 0.77% | 0.13% |
| layer1_plus_layer2_plus_rules | 1.1760 | 1.1640 | 1.87% | -1.01% |

## Directional Hit Rate (Triggered Side Events)
- home events: n=65, layer2=0.47692307692307695, layer2+rules=0.6307692307692307
- away events: n=66, layer2=0.5303030303030303, layer2+rules=0.5151515151515151

## Rule Counts (Home)
- congestion: 22
- key_absent: 50

## Rule Counts (Away)
- congestion: 31
- key_absent: 48