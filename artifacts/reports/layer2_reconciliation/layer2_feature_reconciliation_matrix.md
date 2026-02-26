# Layer 2 Feature Reconciliation Matrix

Generated: 2026-02-26T17:25:31.489949+00:00

## Summary
- baseline RMSE home/away: 1.1963 / 1.0854
- full-model RMSE home/away: 1.1938 / 1.0781
- decision counts: {"conditional_candidate": 23, "drop_candidate": 9, "fix_lineage_before_judgment": 2, "keep_candidate": 2}

| Feature | Source | Risk | Train Cov % | Test Cov % | Decision | Action |
| --- | --- | --- | ---: | ---: | --- | --- |
| home_rolling_xg | team_premium_snapshots | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| home_rolling_xg_against | team_premium_snapshots | low | 100.0 | 100.0 | drop_candidate | drop_in_next_controlled_experiment |
| away_rolling_xg | team_premium_snapshots | low | 100.0 | 100.0 | drop_candidate | drop_in_next_controlled_experiment |
| away_rolling_xg_against | team_premium_snapshots | low | 100.0 | 100.0 | drop_candidate | drop_in_next_controlled_experiment |
| xg_diff | team_premium_snapshots | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| home_rolling_corners | team_premium_snapshots | low | 100.0 | 100.0 | drop_candidate | drop_in_next_controlled_experiment |
| rest_delta | team_premium_snapshots | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| congestion_flag | team_premium_snapshots | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| home_upcoming_tier | team_premium_snapshots | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| away_upcoming_tier | team_premium_snapshots | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| position_gap | fixtures+results+standings | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| points_gap | fixtures+results+standings | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| home_lame_duck | fixtures+results+standings | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| away_lame_duck | fixtures+results+standings | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| is_derby | team_rivalries | critical | 100.0 | 100.0 | fix_lineage_before_judgment | repair_upstream_lineage_then_retest |
| home_form_streak | team_premium_snapshots | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| away_form_streak | team_premium_snapshots | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| home_xg_lost | player_availability+fixture_player_stats | high | 100.0 | 100.0 | drop_candidate | drop_in_next_controlled_experiment |
| away_xg_lost | player_availability+fixture_player_stats | high | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| home_key_absent | player_availability+fixture_player_stats | high | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| away_key_absent | player_availability+fixture_player_stats | high | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| injury_impact | player_availability+fixture_player_stats | high | 100.0 | 100.0 | drop_candidate | drop_in_next_controlled_experiment |
| home_xg_over_scored | team_premium_snapshots | low | 100.0 | 100.0 | drop_candidate | drop_in_next_controlled_experiment |
| home_xg_over_conceded | team_premium_snapshots | low | 100.0 | 100.0 | drop_candidate | drop_in_next_controlled_experiment |
| away_xg_over_scored | team_premium_snapshots | low | 100.0 | 100.0 | keep_candidate | retain_in_candidate_feature_set |
| away_xg_over_conceded | team_premium_snapshots | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| home_playing_top4 | fixtures+results+standings | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| away_playing_top4 | fixtures+results+standings | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| derby_position_gap | team_rivalries | critical | 100.0 | 100.0 | fix_lineage_before_judgment | repair_upstream_lineage_then_retest |
| away_rolling_corners | team_premium_snapshots | low | 100.0 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| odds_model_gap_home | fixture_odds_markets+predictions(lambda_xgb) | low | 97.3 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| odds_model_gap_draw | fixture_odds_markets+predictions(lambda_xgb) | low | 97.3 | 100.0 | keep_candidate | retain_in_candidate_feature_set |
| odds_model_gap_away | fixture_odds_markets+predictions(lambda_xgb) | low | 97.3 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| odds_opening_gap_home | fixture_odds_markets+predictions(lambda_xgb) | low | 97.3 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |
| odds_opening_gap_draw | fixture_odds_markets+predictions(lambda_xgb) | low | 97.3 | 100.0 | drop_candidate | drop_in_next_controlled_experiment |
| odds_opening_gap_away | fixture_odds_markets+predictions(lambda_xgb) | low | 97.3 | 100.0 | conditional_candidate | segment_and_retest_before_final_decision |

## Notes
- `implemented_in_train` and `implemented_in_predict` are true because the deployed model payload carries the feature contract.
- Final keep/drop calls for lineage-blocked sources remain deferred until timestamp repair is in place.