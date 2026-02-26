# Layer 1 Residual Anatomy Audit

Generated: 2026-02-26T15:48:39.143519+00:00

## Defaults Used
- scope: all leagues
- recent window: 365 days
- decision-grade segment threshold: n >= 120
- informative segment threshold: n >= 50

## Coverage and Timing Integrity
- fixtures total: 7824
- fixtures with latest lambda pair: 7137
- fixtures with pre-kickoff lambda pair: 7137
- fixtures missing pre-kickoff lambda pair: 687
- latest chosen lambda pair occurs after kickoff: 0 (0.00%)
- pre-kickoff lambda lag minutes (p50/p90/p99): 60.0 / 60.0 / 60.0

## Overall Residual Metrics
| View | n | Home Bias | Away Bias | Home RMSE | Away RMSE | Home MAE | Away MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| latest_anytime_pair | 7137 | 0.0149 | -0.0080 | 1.1959 | 1.0756 | 0.9492 | 0.8497 |
| latest_pre_kickoff_pair | 7137 | 0.0149 | -0.0080 | 1.1959 | 1.0756 | 0.9492 | 0.8497 |
| recent_pre_kickoff_pair | 6447 | 0.0108 | 0.0101 | 1.1750 | 1.0653 | 0.9377 | 0.8386 |
| historical_pre_kickoff_pair | 690 | 0.0533 | -0.1777 | 1.3764 | 1.1671 | 1.0569 | 0.9530 |

## Top Problem Segments (Decision-Grade n)
| Segment | Value | n | Home Bias | Away Bias | Home RMSE | Away RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| league_code | MX1 | 222 | 0.1467 | 0.1057 | 1.2590 | 1.1280 |
| league_code | ECL | 320 | 0.0199 | -0.2294 | 1.3607 | 1.0744 |
| league_code | CH1 | 149 | 0.1073 | 0.1271 | 1.3828 | 1.2037 |
| league_code | EL | 397 | 0.0361 | -0.1983 | 1.2315 | 1.1126 |
| league_code | F2 | 206 | -0.1210 | 0.0776 | 1.1079 | 1.1195 |
| league_code | B1 | 192 | -0.1658 | 0.0253 | 1.0332 | 1.1549 |
| league_code | P1 | 197 | 0.1167 | 0.0736 | 1.1604 | 1.0309 |
| league_code | F1 | 198 | 0.1293 | -0.0539 | 1.2569 | 1.2490 |
| league_code | DK1 | 120 | -0.0331 | 0.1255 | 1.3204 | 1.1793 |
| league_code | CL | 400 | 0.1352 | -0.0191 | 1.4275 | 1.3032 |
| league_code | E2 | 379 | 0.1031 | -0.0253 | 1.1921 | 0.9902 |
| league_code | E3 | 382 | -0.0399 | 0.0864 | 1.1643 | 0.9903 |
| league_code | PL1 | 186 | 0.0415 | -0.0756 | 1.2698 | 1.0239 |
| league_code | D2 | 198 | 0.0047 | 0.1110 | 1.1522 | 1.0743 |
| league_code | T1 | 196 | -0.0357 | 0.0799 | 1.0085 | 1.0535 |

## Artifacts
- JSON: `artifacts\reports\layer2_reconciliation\layer1_residual_anatomy.json`
- Markdown: `artifacts\reports\layer2_reconciliation\layer1_residual_anatomy.md`