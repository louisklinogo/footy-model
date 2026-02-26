# Layer 2 Player Block Controlled Comparison

Generated: 2026-02-26T16:29:45.373033Z

| Variant | Features | Home RMSE | Away RMSE | Home Lift | Away Lift | Enabled Leagues |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_full_features | 36 | 1.1928 | 1.0776 | 0.29% | 0.72% | E1, E3, RO1 |
| candidate_no_player_block | 31 | 1.1933 | 1.0759 | 0.25% | 0.88% | E3, RO1 |

## Delta (candidate - baseline)

- home_test_rmse: +0.000464 (worse)
- away_test_rmse: -0.001708 (better)
- home_lift: -0.04%
- away_lift: +0.16%
- enabled leagues change: -1
- enabled leagues removed: E1
- enabled leagues added: -