# Player-Impact Variant Sweep

Generated: 2026-02-26T20:28:35.722638+00:00

## Decision
- decision: `keep_player_block_conditional`
- runs: `36`
- passing runs (logic+predictive): `0`

## Best Run
- variant: `season_xg90`
- thresholds: xg_share>=0.12, minutes_share>=0.07
- football: corr=0.0179, key_delta=0.1273, status=warn
- predictive: triggered_lift=0.09%, global_lift=0.02%, status=warn

## Top 10
- 1. season_xg90 | xg_thr=0.12, min_thr=0.07 | logic=warn | pred=warn | trig=0.09% | global=0.02%
- 2. recency_xg90 | xg_thr=0.12, min_thr=0.07 | logic=warn | pred=warn | trig=0.09% | global=0.02%
- 3. baseline_avg_xg10 | xg_thr=0.12, min_thr=0.07 | logic=warn | pred=warn | trig=0.09% | global=0.02%
- 4. starts_minutes_weighted | xg_thr=0.12, min_thr=0.07 | logic=warn | pred=warn | trig=0.09% | global=0.02%
- 5. season_xg90 | xg_thr=0.08, min_thr=0.07 | logic=warn | pred=warn | trig=0.04% | global=0.00%
- 6. recency_xg90 | xg_thr=0.08, min_thr=0.07 | logic=warn | pred=warn | trig=0.04% | global=0.00%
- 7. baseline_avg_xg10 | xg_thr=0.08, min_thr=0.07 | logic=warn | pred=warn | trig=0.04% | global=0.00%
- 8. starts_minutes_weighted | xg_thr=0.08, min_thr=0.07 | logic=warn | pred=warn | trig=0.04% | global=0.00%
- 9. season_xg90 | xg_thr=0.10, min_thr=0.07 | logic=warn | pred=warn | trig=0.01% | global=-0.01%
- 10. recency_xg90 | xg_thr=0.10, min_thr=0.07 | logic=warn | pred=warn | trig=0.01% | global=-0.01%