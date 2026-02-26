# Layer 2 Feature Intent Contract (Plain English)

Date: 2026-02-26  
Status: active source-of-truth for Layer 2 feature behavior  
Scope: `src/modeling/layer2_situational/train_situational_residual.py` features

## Purpose
This contract defines what every Layer 2 feature is intended to do.  
Layer 2 should only make small, context-aware corrections to Layer 1 when this intent is supported by point-in-time data and out-of-sample evidence.

## Core Rule
If a feature is present in code but does not behave according to this intent in DB evidence (coverage, timing, variance, incremental lift), it must be marked as **implemented with issues**.

## Feature-by-Feature Intent

1. `home_rolling_xg`  
Expected behavior: raise home correction when home has been creating better chances recently.

2. `home_rolling_xg_against`  
Expected behavior: reduce home correction (or support away correction) when home has been allowing strong chances.

3. `away_rolling_xg`  
Expected behavior: raise away correction when away has been creating better chances recently.

4. `away_rolling_xg_against`  
Expected behavior: raise home correction when away has been allowing strong chances.

5. `xg_diff`  
Expected behavior: encode net chance-quality edge; larger home edge should push correction toward home.

6. `home_rolling_corners`  
Expected behavior: capture sustained attacking pressure for home; small positive home correction.

7. `rest_delta`  
Expected behavior: fresher side gets a small positive correction.

8. `congestion_flag`  
Expected behavior: detect fatigue regime; reduce correction for teams under compressed schedule stress.

9. `home_upcoming_tier`  
Expected behavior: represent potential look-ahead/rotation risk for home due to upcoming fixture difficulty.

10. `away_upcoming_tier`  
Expected behavior: same look-ahead/rotation context for away.

11. `position_gap`  
Expected behavior: table-strength separation; larger positive home edge should support home correction.

12. `points_gap`  
Expected behavior: points-based strength gap; stronger side should receive modest favorable correction.

13. `home_lame_duck`  
Expected behavior: reduce home correction when late-season motivation is structurally low.

14. `away_lame_duck`  
Expected behavior: reduce away correction when late-season motivation is structurally low.

15. `is_derby`  
Expected behavior: capture derby regime shift where normal strength priors may weaken.

16. `home_form_streak`  
Expected behavior: short-run momentum/confidence proxy for home.

17. `away_form_streak`  
Expected behavior: short-run momentum/confidence proxy for away.

18. `home_xg_lost`  
Expected behavior: estimate missing attacking output from unavailable home players; reduce home correction.

19. `away_xg_lost`  
Expected behavior: estimate missing attacking output from unavailable away players; reduce away correction.

20. `home_key_absent`  
Expected behavior: binary downgrade when home has a key absentee.

21. `away_key_absent`  
Expected behavior: binary downgrade when away has a key absentee.

22. `injury_impact`  
Expected behavior: relative injury burden (`home_xg_lost - away_xg_lost`); shift correction toward healthier side.

23. `home_xg_over_scored`  
Expected behavior: finishing overperformance signal; encourage mild mean reversion if persistent.

24. `home_xg_over_conceded`  
Expected behavior: defensive concession overperformance signal; encourage mean reversion.

25. `away_xg_over_scored`  
Expected behavior: away finishing overperformance signal; encourage mean reversion.

26. `away_xg_over_conceded`  
Expected behavior: away defensive concession overperformance signal; encourage mean reversion.

27. `home_playing_top4`  
Expected behavior: penalty context when home faces elite opposition.

28. `away_playing_top4`  
Expected behavior: penalty context when away faces elite opposition.

29. `derby_position_gap`  
Expected behavior: interaction term to temper rank-gap influence inside derby regime.

30. `away_rolling_corners`  
Expected behavior: sustained attacking pressure for away; small positive away correction.

31. `odds_model_gap_home`  
Expected behavior: if market is more bullish on home than model baseline, nudge home correction up.

32. `odds_model_gap_draw`  
Expected behavior: if market is more draw-leaning than baseline, shift correction toward draw-like outcome balance.

33. `odds_model_gap_away`  
Expected behavior: if market is more bullish on away than model baseline, nudge away correction up.

34. `odds_opening_gap_home`  
Expected behavior: opening-price disagreement for home; early market signal for correction.

35. `odds_opening_gap_draw`  
Expected behavior: opening-price disagreement for draw; early market signal for correction.

36. `odds_opening_gap_away`  
Expected behavior: opening-price disagreement for away; early market signal for correction.

## Acceptance Standard
A feature is considered **implemented well** only if all are true:
1. Present and aligned in train + predict code paths.
2. Point-in-time timing semantics are valid for its source data.
3. Coverage/variance are adequate for learning (not effectively constant/sparse noise).
4. Controlled experiments show non-negative incremental value in relevant blocks.

