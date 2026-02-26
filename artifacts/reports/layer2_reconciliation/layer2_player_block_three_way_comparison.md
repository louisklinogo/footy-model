# Layer 2 Player Block Three-Way Comparison

Generated: 2026-02-26T16:33:18.587421+00:00

| Variant | Features | Home RMSE | Away RMSE | Home Lift | Away Lift | Enabled Leagues |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_full_features | 36 | 1.1928 | 1.0776 | 0.29% | 0.72% | E1, E3, RO1 |
| candidate_no_player_block | 31 | 1.1933 | 1.0759 | 0.25% | 0.88% | E3, RO1 |
| candidate_keyabs_only | 33 | 1.1917 | 1.0758 | 0.39% | 0.89% | E3, RO1 |

## Delta Vs Baseline Full Features

### candidate_no_player_block
- home_test_rmse: +0.000464
- away_test_rmse: -0.001708
- home_lift: -0.04%
- away_lift: +0.16%
- enabled_league_count: -1
- enabled_leagues_removed: E1
- enabled_leagues_added: -

### candidate_keyabs_only
- home_test_rmse: -0.001159
- away_test_rmse: -0.001772
- home_lift: +0.10%
- away_lift: +0.16%
- enabled_league_count: -1
- enabled_leagues_removed: E1
- enabled_leagues_added: -

Recommendation winner by home+away RMSE sum: `candidate_keyabs_only`