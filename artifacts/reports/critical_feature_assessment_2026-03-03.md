# Critical Feature Engineering Assessment (No-Fluff)
Date: 2026-03-03
Author: Codex (Senior DS review)

## 1) What was checked

### Data inventory from live DB
- Tables inspected: `fixture_incidents_sofascore`, `fixture_incident_lead_states`, `fixture_player_stats`, `player_availability`, `fixture_odds_markets`.
- Coverage snapshot:
  - `fixtures`: 11,102
  - `fixture_incidents_sofascore` distinct fixtures: 7,734
  - `fixture_incident_lead_states` distinct fixtures: 7,736
  - `fixture_player_stats` distinct fixtures: 7,811
  - `player_availability` distinct fixtures: 7,892

### Current v2 contracts vs actual feature completeness (training frame = 8,082 rows)
- `scoreline` required-complete rows: **91.38%**
- `corners` required-complete rows: **87.78%**
- `anytime` required-complete rows: **1.77%**

Anytime required-feature missingness highlights:
- `adj_lambda_home_final`, `adj_lambda_away_final`: **96.70% missing**
- `home_rolling_xg_p1`, `away_rolling_xg_p1`: **72.79% missing**
- `home_rolling_xg_h2_delta`, `away_rolling_xg_h2_delta`: **72.79% missing**

## 2) Direct answer to your concern about incidents

You are right: we currently do **not** use minute-sequence incident features directly in v2 family contracts.
- Available raw columns in `fixture_incidents_sofascore` include: `minute`, `added_time`, `is_home`, `incident_type`, `home_score`, `away_score`, player IDs/names.
- v2 currently uses only derived lead booleans via labels (`home_led_by_1_any`, etc.), not pre-match historical incident-time features.

So yes: feature engineering is currently under-using the incident dataset.

## 3) Evidence from quick feature-signal and ablation tests

### Univariate signal (anytime markets)
Strongest single features for `h_1up/a_1up/h_2up/a_2up` were mostly:
- `lambda_home_l1`, `lambda_away_l1` (abs AUC about 0.62-0.67)
- Core rolling attack metrics
Weak/non-usable in current form:
- `adj_lambda_*` (too sparse to evaluate)
- `rule_fired_*` (near-random; abs AUC ~0.50)
- `xg_p1/h2_delta` family (weak-moderate and very sparse)

### Incident-history additive test (causal, shifted rolling features)
Added pre-match historical incident-derived features (`rate_led1_last10`, `rate_led2_last10`, `avg_first_lead_min_last10`) to anytime models.
Observed deltas vs baseline anytime set:
- `h_1up`: AUC +0.0037
- `a_1up`: AUC +0.0070
- `h_2up`: AUC +0.0123
- `a_2up`: AUC +0.0183
Brier also improved for all 4 in that run.

Conclusion: incident-history features add real (modest) lift, strongest on 2UP.

### Scoreline/Goals ablation
Adding odds/implied features to scoreline-family predictors:
- `1x2_h`: AUC ~flat (0.63350 -> 0.63362)
- `o15`: AUC up (0.56537 -> 0.57875)
- `u35`: AUC up (0.52767 -> 0.53493), Brier improved

Conclusion: odds are more valuable for totals than for pure side outcome.

### Corners data quality / availability constraint
`fixture_odds_markets` (provider=`sofascore`) corners line coverage:
- `7.5`: 3 fixtures
- `8.5`: 892 fixtures
- `9.5`: 5,164 fixtures
- `10.5`: 1,633 fixtures

This is a major structural constraint. Sparse lines cannot be reliable feature anchors.

## 4) Honest architecture assessment

### What is good
- Family split (scoreline vs corners vs anytime) is directionally correct.
- Coherent market derivation (distributions -> multiple lines) is correct design.
- Leakage-awareness exists and is improving.

### What is not good (and why model strength is capped)
1. **Anytime contract is currently mis-specified for availability**.
   - Required features are largely absent, forcing heavy imputation.
2. **Incident data is under-utilized**.
   - You have rich timeline data but are not yet turning it into strong causal pre-match aggregates.
3. **Corners market data is sparse by line/provider**.
   - Feature quality depends on market availability that is inconsistent.
4. **No full family-level feature selection loop is operational yet**.
   - We still need systematic market-by-market ablation + stability testing as a standard training step.

## 5) On PyTorch/Keras/TensorFlow: should we switch now?

Short answer: **No, not yet**.

Why:
- Current tabular sample sizes (~8k training rows per family after filtering) are usually better served by tree/GLM methods.
- Your current bottleneck is feature coverage and market data quality, not model class capacity.
- Deep learning is justified only after:
  - substantially larger historical data,
  - richer sequence inputs (event/player embeddings), and
  - strict out-of-time validation proving net lift over strong tabular baselines.

Recommended process:
1. Establish strong tabular baseline + calibrated walk-forward metrics.
2. Add engineered incident/player features with causal timestamping.
3. Only then run controlled deep-model benchmark as an experiment, not a rewrite.

## 6) Immediate actions (priority order)

1. **Fix anytime contract** (high priority):
   - Move sparse fields (`adj_lambda_*`, `xg_p1`, `xg_h2_delta`) from required to optional.
   - Add missingness indicators.
2. **Build incident-history feature job** (high priority):
   - Team-level rolling `lead1_rate`, `lead2_rate`, `first_lead_min`, comeback/concession profiles.
   - Strictly shifted by fixture time.
3. **Run family-specific feature ablation pipeline** (high priority):
   - Required report: AUC/Brier/ECE deltas per feature block and per market.
4. **Corners scope policy by availability** (high priority):
   - Gate lines by minimum support; avoid forcing sparse lines as feature anchors.
5. **Model class policy** (medium priority):
   - Keep GBDT/Poisson for now; add deep models only as benchmark after feature/data fixes.

## Bottom line
The current architecture is not fundamentally wrong, but **the feature layer is not yet production-strong**, especially for anytime and sparse corners lines. Your instinct is correct: we need disciplined feature engineering experiments, and incident timelines must be converted into causal historical features before expecting major AUC gains.
