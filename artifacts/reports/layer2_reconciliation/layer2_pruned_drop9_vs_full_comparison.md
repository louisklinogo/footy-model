# Layer 2 Global Variant Comparison

Generated: 2026-02-26T18:59:14.255951+00:00

| Variant | Features | Home RMSE | Away RMSE | Home Lift | Away Lift | Enabled Leagues |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_full_features | 36 | 1.1910 | 1.0783 | 0.44% | 0.66% | E1, RO1 |
| candidate_pruned_drop9 | 27 | 1.1960 | 1.0758 | 0.02% | 0.89% | RO1 |

## Delta (candidate - baseline)
- home_test_rmse: +0.005038
- away_test_rmse: -0.002539
- home_lift: -0.42%
- away_lift: +0.23%
- enabled leagues removed: E1
- enabled leagues added: -