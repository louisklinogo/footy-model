# Footy-Model Improvement Plan (Evidence-Backed)

Date: March 3, 2026  
Scope: practical changes that can improve prediction quality, tradability, and bankroll protection.

## 1) What We Verified First

1. External repos contain useful ideas, but parts are not production-ready as copied.
`sports-betting-model/models/continuous_learning.py` currently has an `IndentationError`, so it is not a drop-in production pattern.

2. Our current market performance is mixed.
`model_artifacts/market_models/market_gating.json` has 35 markets, with 30 marked eligible and 5 ineligible.
Stronger examples by AUC: `dc_1x`, `dc_x2`, `ah_a05`, `ah_h15`, `ah_h05`.
Weaker examples by AUC/BSS: `dc_12`, `c105`, `away_or_o25`, `home_or_o15`.

3. Tradable policy and gating are misaligned.
`model_artifacts/market_models/risk_policy.json` currently includes tradable markets like `dc_12` and `c105` even though they are weak/ineligible in gating output.

4. Odds feed reality matters more than model wishlist.
Current `latest_pre_match` rows in `fixture_odds_markets` are mostly: `1x2`, `dc`, `dnb`, `btts`, `ou`, `corners_ou`, `cards_ou`.
Team-specific goals/corners odds are not reliably present in current SofaScore snapshots.

5. There is an extraction gap for Asian handicap.
Live odds payloads include market `Asian handicap`, but our ingester stores no `ah` rows in `fixture_odds_markets`.
Root cause is likely choice-name mapping for handicap choices with team names (not normalized to `home`/`away`).

6. For player availability, `GET /event/{id}/lineups` is still the correct endpoint, but many scheduled fixtures return 404 until closer to kickoff.
This is a data-timing/coverage issue, not a wrong endpoint issue.

## 2) Priority Actions (Most Impact First)

## P0: Fix Data-to-Market Parity (Do This First)

1. Fix Asian handicap extraction so available odds are actually captured.
Files: [backfill_sofascore_odds_markets_v1.py](C:\Developer\soccer\footy-model\src\ingest\backfill_sofascore_odds_markets_v1.py), [test_sofascore_odds_parse.py](C:\Developer\soccer\footy-model\tests\test_sofascore_odds_parse.py)  
Action: update choice mapping logic for handicap markets where choices are team-name based (for example `(-0.5) Leeds United`).  
Done when: `latest_pre_match` contains non-zero `market_code='ah'` rows for upcoming fixtures with visible AH in raw API payloads.

2. Add a daily odds coverage audit and gate tradability by real availability.
Files: create `src/ingest/report_odds_market_coverage.py`, write artifact to `artifacts/reports/odds_coverage/`.  
Action: report coverage by `market_code`, kickoff horizon, league, and percentage of fixtures with odds.  
Done when: each market has a measurable coverage %, and low-coverage markets are automatically excluded from tradable universe.

3. Align risk policy with gating outputs.
Files: [risk_policy.json](C:\Developer\soccer\footy-model\model_artifacts\market_models\risk_policy.json), [assess_prediction_risk.py](C:\Developer\soccer\footy-model\src\modeling\evaluation\assess_prediction_risk.py)  
Action: tradable set should be intersection of:
`user_required_markets ∩ gating.eligible ∩ coverage_above_threshold`.
Done when: no ineligible market remains in `tradable_markets`.

4. Split markets into `predict_only` vs `tradable`.
Files: `risk_policy.json` and export metadata path in [export_market_outcomes_fixtures_first.py](C:\Developer\soccer\footy-model\src\modeling\export\export_market_outcomes_fixtures_first.py)  
Action: keep generating probabilities for exploratory markets, but only stake on tradable ones.  
Done when: every output row explicitly has `market_mode = tradable|predict_only`.

## P1: Improve Model Signal Quality

1. Add anti-recency anchor features (team-level, not copied blindly from player props repo).
Files: feature build path used by market model training and prediction (start in [build_team_style_clusters.py](C:\Developer\soccer\footy-model\src\features\build_team_style_clusters.py) and training pipeline inputs).  
Action: create features:
`season_baseline_xg`, `recent_xg_mean_5`, `recent_vs_baseline_zscore`, `regressed_recent_xg`.
Done when: walk-forward report shows positive lift in BSS and stable ECE for core markets.

2. Replace binary-only injury impact with weighted absence signal.
Files: [predict_situational_residual.py](C:\Developer\soccer\footy-model\src\modeling\layer2_situational\predict_situational_residual.py), [train_situational_residual.py](C:\Developer\soccer\footy-model\src\modeling\layer2_situational\train_situational_residual.py)  
Action: keep `key_absent` binary, but add continuous feature such as:
`sum(avg_xg_last10 * start_rate_last10)` for missing players.  
Done when: ablation table shows continuous feature improves residual calibration without raising overfit risk.

3. Market-by-market calibration hygiene.
Files: [market_outcome_calibrator.py](C:\Developer\soccer\footy-model\src\modeling\layer2_markets\market_outcome_calibrator.py), [market_gating.json](C:\Developer\soccer\footy-model\model_artifacts\market_models\market_gating.json)  
Action: only promote calibrators when minimum `test_n` and calibration stability are met.  
Done when: tradable markets remain under agreed ECE cap.

## P2: Bankroll-Safe Decision Policy (Given Small Bankroll Constraint)

1. Use stricter stake governance than current loose policy.
Files: [risk_policy.json](C:\Developer\soccer\footy-model\model_artifacts\market_models\risk_policy.json)  
Action: enable hard gates (`hard_precision_gate=true`), lower max stake fraction, and enforce no-bet default when data quality is weak.  
Done when: no bets are emitted on stale/missing-odds fixtures and drawdown volatility declines.

2. Add edge-bucket monitoring and stop-loss controls.
Files: [assess_prediction_risk.py](C:\Developer\soccer\footy-model\src\modeling\evaluation\assess_prediction_risk.py) plus new report artifact.  
Action: track realized performance by edge buckets (`2-4%`, `4-6%`, `6%+`) and auto-pause bucket/market on degradation.  
Done when: underperforming edge buckets are automatically suppressed.

## 3) Recommended Market Universe Right Now

Use two lists until coverage and extraction are fixed.

1. Tradable now (subject to daily coverage check):  
`1x2`, `dc_1x`, `dc_x2`, `o15`, `u35`, `corners_ou` lines with acceptable calibration (`c85`, `c95` currently better than `c105`).

2. Predict-only for now:  
`dc_12`, low-performing combo markets, and team-specific totals/corners where odds are not consistently available in current feed.

3. Conditional tradable after P0 fix:  
Asian handicap variants (`ah_*` / `eh_*`) once extraction and coverage are confirmed in DB.

## 4) Implementation Sequence (Low Risk)

1. P0.1 AH extraction fix + tests.
2. P0.2 coverage report + policy intersection.
3. P0.3 tradable/predict-only split.
4. P1.1 anti-recency features.
5. P1.2 weighted injury signal.
6. P2 bankroll policy tightening + edge-bucket kill-switch.

## 5) Success Criteria

1. Data quality:
`ah` markets appear in `fixture_odds_markets` when present in API payload.

2. Policy integrity:
`tradable_markets` has zero entries that are ineligible or below minimum coverage threshold.

3. Model quality:
BSS and ECE improve or remain stable on walk-forward for tradable markets.

4. Betting safety:
Fewer low-quality bets and lower short-horizon drawdown volatility.

