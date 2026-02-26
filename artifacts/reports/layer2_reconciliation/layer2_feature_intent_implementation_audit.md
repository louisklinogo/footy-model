# Layer 2 Feature Intent vs Implementation Audit

Generated: 2026-02-26T17:26:09.155939+00:00

## Summary
- total features: 36
- implemented well: 1
- implemented with issues: 35
- reconciliation decisions: {"conditional_candidate": 23, "drop_candidate": 9, "fix_lineage_before_judgment": 2, "keep_candidate": 2}

## Source Timing Snapshot
- lambda late latest pairs: 0 / 7137
- odds late chosen snapshots: 0
- availability late rows (FT): 0.0%
- availability upcoming coverage (14d): 0.0%

| Feature | Source Risk | Train% | Predict% | Decision | Verdict | Issues |
| --- | --- | ---: | ---: | --- | --- | --- |
| home_rolling_xg | low | 91.8% | 95.7% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| home_rolling_xg_against | low | 91.8% | 95.7% | drop_candidate | implemented_with_issues | reconciliation_decision:drop_candidate |
| away_rolling_xg | low | 91.9% | 95.7% | drop_candidate | implemented_with_issues | reconciliation_decision:drop_candidate |
| away_rolling_xg_against | low | 91.9% | 95.7% | drop_candidate | implemented_with_issues | reconciliation_decision:drop_candidate |
| xg_diff | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| home_rolling_corners | low | 91.9% | 97.3% | drop_candidate | implemented_with_issues | reconciliation_decision:drop_candidate |
| rest_delta | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| congestion_flag | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| home_upcoming_tier | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| away_upcoming_tier | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| position_gap | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| points_gap | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| home_lame_duck | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | near_constant_train_distribution, mixed_incremental_value, no_active_signal_in_current_window |
| away_lame_duck | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | near_constant_train_distribution, mixed_incremental_value, no_active_signal_in_current_window |
| is_derby | critical | 100.0% | 100.0% | fix_lineage_before_judgment | implemented_with_issues | high_source_risk:critical, reconciliation_decision:fix_lineage_before_judgment, derby_prevalence_very_low |
| home_form_streak | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| away_form_streak | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| home_xg_lost | high | 100.0% | 100.0% | drop_candidate | implemented_with_issues | high_source_risk:high, reconciliation_decision:drop_candidate, upcoming_availability_unpopulated |
| away_xg_lost | high | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | high_source_risk:high, mixed_incremental_value, upcoming_availability_unpopulated |
| home_key_absent | high | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | high_source_risk:high, mixed_incremental_value, upcoming_availability_unpopulated |
| away_key_absent | high | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | high_source_risk:high, mixed_incremental_value, upcoming_availability_unpopulated |
| injury_impact | high | 100.0% | 100.0% | drop_candidate | implemented_with_issues | high_source_risk:high, reconciliation_decision:drop_candidate, upcoming_availability_unpopulated |
| home_xg_over_scored | low | 100.0% | 100.0% | drop_candidate | implemented_with_issues | reconciliation_decision:drop_candidate |
| home_xg_over_conceded | low | 100.0% | 100.0% | drop_candidate | implemented_with_issues | reconciliation_decision:drop_candidate |
| away_xg_over_scored | low | 100.0% | 100.0% | keep_candidate | implemented_well | - |
| away_xg_over_conceded | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| home_playing_top4 | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| away_playing_top4 | low | 100.0% | 100.0% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| derby_position_gap | critical | 100.0% | 100.0% | fix_lineage_before_judgment | implemented_with_issues | high_source_risk:critical, reconciliation_decision:fix_lineage_before_judgment, derby_prevalence_very_low |
| away_rolling_corners | low | 92.1% | 97.3% | conditional_candidate | implemented_with_issues | mixed_incremental_value |
| odds_model_gap_home | low | 89.2% | 39.2% | conditional_candidate | implemented_with_issues | mixed_incremental_value, upcoming_odds_sparse |
| odds_model_gap_draw | low | 89.2% | 39.2% | keep_candidate | implemented_with_issues | upcoming_odds_sparse |
| odds_model_gap_away | low | 89.2% | 39.2% | conditional_candidate | implemented_with_issues | mixed_incremental_value, upcoming_odds_sparse |
| odds_opening_gap_home | low | 89.2% | 39.2% | conditional_candidate | implemented_with_issues | mixed_incremental_value, upcoming_odds_sparse |
| odds_opening_gap_draw | low | 89.2% | 39.2% | drop_candidate | implemented_with_issues | reconciliation_decision:drop_candidate, upcoming_odds_sparse |
| odds_opening_gap_away | low | 89.2% | 39.2% | conditional_candidate | implemented_with_issues | mixed_incremental_value, upcoming_odds_sparse |