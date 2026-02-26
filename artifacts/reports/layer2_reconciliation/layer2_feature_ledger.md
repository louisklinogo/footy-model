# Layer 2 Feature Ledger

Generated: 2026-02-26T17:17:24.600098+00:00

- train_n: 4394
- test_n: 1423
- split_time: 2026-01-18 14:30:00+00:00
- baseline_rmse home/away: 1.1963 / 1.0854
- full_model_rmse home/away: 1.1938 / 1.0781

| Feature | Source | Train Cov % | Test Cov % | Single Lift H | Single Lift A | Drop dRMSE H | Drop dRMSE A | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| home_rolling_xg | team_premium_snapshots | 100.0 | 100.0 | -0.0015 | -0.0044 | -0.0030 | 0.0020 | conditional_candidate |
| home_rolling_xg_against | team_premium_snapshots | 100.0 | 100.0 | -0.0032 | -0.0014 | -0.0033 | -0.0007 | drop_candidate |
| away_rolling_xg | team_premium_snapshots | 100.0 | 100.0 | -0.0041 | -0.0025 | -0.0013 | -0.0002 | drop_candidate |
| away_rolling_xg_against | team_premium_snapshots | 100.0 | 100.0 | -0.0045 | -0.0038 | -0.0007 | -0.0014 | drop_candidate |
| xg_diff | team_premium_snapshots | 100.0 | 100.0 | 0.0019 | -0.0001 | 0.0003 | -0.0004 | conditional_candidate |
| home_rolling_corners | team_premium_snapshots | 100.0 | 100.0 | -0.0075 | -0.0093 | -0.0006 | -0.0014 | drop_candidate |
| rest_delta | team_premium_snapshots | 100.0 | 100.0 | -0.0069 | -0.0023 | -0.0016 | 0.0007 | conditional_candidate |
| congestion_flag | team_premium_snapshots | 100.0 | 100.0 | 0.0003 | -0.0001 | -0.0012 | -0.0019 | conditional_candidate |
| home_upcoming_tier | team_premium_snapshots | 100.0 | 100.0 | -0.0008 | 0.0004 | 0.0001 | 0.0000 | conditional_candidate |
| away_upcoming_tier | team_premium_snapshots | 100.0 | 100.0 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | conditional_candidate |
| position_gap | fixtures+results+standings | 100.0 | 100.0 | 0.0049 | -0.0014 | -0.0000 | -0.0005 | conditional_candidate |
| points_gap | fixtures+results+standings | 100.0 | 100.0 | 0.0031 | 0.0015 | -0.0063 | 0.0005 | conditional_candidate |
| home_lame_duck | fixtures+results+standings | 100.0 | 100.0 | 0.0001 | 0.0003 | 0.0000 | 0.0000 | conditional_candidate |
| away_lame_duck | fixtures+results+standings | 100.0 | 100.0 | 0.0001 | 0.0003 | 0.0000 | 0.0000 | conditional_candidate |
| is_derby | team_rivalries | 100.0 | 100.0 | 0.0001 | 0.0003 | 0.0000 | 0.0000 | fix_lineage_before_judgment |
| home_form_streak | team_premium_snapshots | 100.0 | 100.0 | 0.0013 | -0.0005 | -0.0028 | -0.0008 | conditional_candidate |
| away_form_streak | team_premium_snapshots | 100.0 | 100.0 | -0.0005 | 0.0023 | -0.0009 | -0.0005 | conditional_candidate |
| home_xg_lost | player_availability+fixture_player_stats | 100.0 | 100.0 | -0.0054 | -0.0058 | -0.0008 | -0.0008 | drop_candidate |
| away_xg_lost | player_availability+fixture_player_stats | 100.0 | 100.0 | -0.0023 | -0.0047 | 0.0005 | -0.0016 | conditional_candidate |
| home_key_absent | player_availability+fixture_player_stats | 100.0 | 100.0 | -0.0001 | 0.0004 | 0.0000 | -0.0001 | conditional_candidate |
| away_key_absent | player_availability+fixture_player_stats | 100.0 | 100.0 | 0.0001 | 0.0003 | 0.0000 | 0.0000 | conditional_candidate |
| injury_impact | player_availability+fixture_player_stats | 100.0 | 100.0 | -0.0044 | -0.0048 | -0.0041 | -0.0017 | drop_candidate |
| home_xg_over_scored | team_premium_snapshots | 100.0 | 100.0 | -0.0084 | -0.0106 | -0.0032 | -0.0013 | drop_candidate |
| home_xg_over_conceded | team_premium_snapshots | 100.0 | 100.0 | -0.0044 | -0.0047 | -0.0019 | -0.0010 | drop_candidate |
| away_xg_over_scored | team_premium_snapshots | 100.0 | 100.0 | -0.0035 | -0.0038 | 0.0003 | 0.0001 | keep_candidate |
| away_xg_over_conceded | team_premium_snapshots | 100.0 | 100.0 | -0.0034 | -0.0030 | -0.0004 | 0.0001 | conditional_candidate |
| home_playing_top4 | fixtures+results+standings | 100.0 | 100.0 | 0.0009 | 0.0013 | 0.0000 | 0.0000 | conditional_candidate |
| away_playing_top4 | fixtures+results+standings | 100.0 | 100.0 | 0.0014 | 0.0015 | 0.0000 | 0.0000 | conditional_candidate |
| derby_position_gap | team_rivalries | 100.0 | 100.0 | 0.0001 | 0.0003 | 0.0000 | 0.0000 | fix_lineage_before_judgment |
| away_rolling_corners | team_premium_snapshots | 100.0 | 100.0 | 0.0002 | -0.0055 | -0.0011 | -0.0015 | conditional_candidate |
| odds_model_gap_home | fixture_odds_markets+predictions(lambda_xgb) | 97.3 | 100.0 | 0.0104 | 0.0093 | 0.0004 | -0.0010 | conditional_candidate |
| odds_model_gap_draw | fixture_odds_markets+predictions(lambda_xgb) | 97.3 | 100.0 | 0.0011 | -0.0093 | 0.0026 | 0.0001 | keep_candidate |
| odds_model_gap_away | fixture_odds_markets+predictions(lambda_xgb) | 97.3 | 100.0 | 0.0033 | 0.0042 | 0.0014 | -0.0031 | conditional_candidate |
| odds_opening_gap_home | fixture_odds_markets+predictions(lambda_xgb) | 97.3 | 100.0 | 0.0062 | 0.0028 | -0.0022 | 0.0016 | conditional_candidate |
| odds_opening_gap_draw | fixture_odds_markets+predictions(lambda_xgb) | 97.3 | 100.0 | -0.0077 | -0.0042 | -0.0008 | -0.0017 | drop_candidate |
| odds_opening_gap_away | fixture_odds_markets+predictions(lambda_xgb) | 97.3 | 100.0 | 0.0065 | 0.0077 | 0.0011 | -0.0001 | conditional_candidate |

## Decision Key
- `fix_lineage_before_judgment`: source timing semantics are critical-risk; decision deferred.
- `keep_candidate`: drop-column worsens both sides.
- `drop_candidate`: hurts both sides and no standalone lift.
- `conditional_candidate`: mixed behavior.
- `defer_sparse`: insufficient feature coverage.