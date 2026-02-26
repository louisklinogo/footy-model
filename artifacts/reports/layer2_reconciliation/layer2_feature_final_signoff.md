# Layer 2 Feature Final Sign-Off

Generated: 2026-02-26T18:36:30.7483219Z

## Summary
- keep: 2
- fix: 23
- drop: 9
- defer: 2
- total: 36

## Mapping Rule
- keep_candidate -> keep
- conditional_candidate -> fix
- drop_candidate -> drop
- fix_lineage_before_judgment -> defer

## Keep
- away_xg_over_scored
- odds_model_gap_draw

## Fix
- home_rolling_xg
- xg_diff
- rest_delta
- congestion_flag
- home_upcoming_tier
- away_upcoming_tier
- position_gap
- points_gap
- home_lame_duck
- away_lame_duck
- home_form_streak
- away_form_streak
- away_xg_lost
- home_key_absent
- away_key_absent
- away_xg_over_conceded
- home_playing_top4
- away_playing_top4
- away_rolling_corners
- odds_model_gap_home
- odds_model_gap_away
- odds_opening_gap_home
- odds_opening_gap_away

## Drop
- home_rolling_xg_against
- away_rolling_xg
- away_rolling_xg_against
- home_rolling_corners
- home_xg_lost
- injury_impact
- home_xg_over_scored
- home_xg_over_conceded
- odds_opening_gap_draw

## Defer
- is_derby
- derby_position_gap

## Gates
- Historical availability timing leakage is repaired (FT late rows = 0).
- Upcoming availability population remains a serving-readiness gate.
- Calibration remains sample-limited and blocks freeze exit.
