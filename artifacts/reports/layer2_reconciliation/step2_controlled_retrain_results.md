# Step 2 Controlled Retrain Results

Generated: 2026-02-26T07:17:14.725871Z

- Train rows: 4237
- Test rows: 1392
- Coverage threshold: >10%

| Model | Features | Home RMSE | Away RMSE | Home Lift vs Baseline | Away Lift vs Baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (zero residual) | 0 | 1.1172 | 0.9980 | 0.00% | 0.00% |
| Full current feature set | 36 | 1.1469 | 1.0440 | -2.66% | -4.61% |
| Controlled step-2 set | 27 | 1.1373 | 1.0302 | -1.80% | -3.23% |

## Delta (Controlled - Full)
- Home RMSE delta: -0.0095
- Away RMSE delta: -0.0138

## Excluded in Step 2
- Blocked injury/rivalry features: away_key_absent, away_xg_lost, derby_position_gap, home_key_absent, home_xg_lost, injury_impact, is_derby
- Draw-gap features dropped: odds_model_gap_draw, odds_opening_gap_draw