 # Reverse Engineering Analysis: All Markets
 **Date:** 2026-03-20
 **Approach:** Start with outcomes, work backwards to feature combinations
 
 ---
 
 ## 1. ASIAN HANDICAP (AH)
 
 ### Model Features
 - Derived from Poisson score matrix using lambda_home, lambda_away
 - No dedicated AH model - probabilities derived from goal difference
 
 ### Reverse Engineering Findings
 
 #### AH Home -0.5 (Home Win)
 | Lambda_Home | Win Rate | Notes |
 |-------------|----------|-------|
 | 0.0-1.0 | 20.8% | No value |
 | 1.0-1.5 | 35.9% | Below threshold |
 | 1.5-2.0 | 52.5% | Profitable with odds > 1.90 |
 | 2.0-2.5 | 65.1% | Strong favorite |
 | 2.5+ | 77.7% | Heavy favorite |
 
 #### AH Home -1.5 (Home by 2+)
 | Lambda_Home | Win Rate | Notes |
 |-------------|----------|-------|
 | 1.0-1.5 | 14.9% | No value |
 | 1.5-2.0 | 26.5% | Need odds > 3.8 |
 | 2.0-2.5 | 43.2% | Good value territory |
 | 2.5+ | 56.1% | Strong value |
 
 #### AH Away +0.5 (Away DNB)
 | Lambda_Away | Win Rate | Notes |
 |-------------|----------|-------|
 | 0.0-1.0 | 39.8% | Poor value |
 | 1.0-1.5 | 57.3% | Profitable |
 | 1.5-2.0 | 71.6% | Strong value |
 | 2.0+ | 80.4% | Heavy favorite |
 
 #### AH Away +1.5 (Away +1.5 covers)
 | Lambda_Away | Win Rate | Notes |
 |-------------|----------|-------|
 | 0.0-1.0 | 67.2% | Good for short odds |
 | 1.0-1.5 | 80.1% | Strong |
 | 1.5-2.0 | 86.7% | Very strong |
 | 2.0+ | 91.9% | Near certain |
 
 ### Formula
 ```
 p_ah_home_m05 = P(lambda_home >= 2.0) ≈ 65% win rate
 p_ah_away_p05 = P(lambda_away >= 1.5) ≈ 72% win rate
 p_ah_away_p15 = P(lambda_away >= 1.0) ≈ 80% win rate
 ```
 
 ---
 
 ## 2. EUROPEAN HANDICAP (EH)
 
 ### Model Features
 - Derived from Poisson score matrix (same as AH)
 - EH3 0:1 = home -1 (wins by 2+, draws by 1, loses by 0+)
 - EH3 1:0 = away -1 (wins by 2+, draws by 1, loses by 0+)
 
 ### Reverse Engineering Findings
 
 #### EH3 0:1 Home (Home -1, wins by 2+)
 | Lambda_Home | Win Rate |
 |-------------|----------|
 | 1.0-1.5 | 14.9% |
 | 1.5-2.0 | 26.5% |
 | 2.0-2.5 | 43.2% |
 | 2.5+ | 56.1% |
 
 #### EH3 1:0 Away (Away -1, wins by 2+)
 | Lambda_Away | Win Rate |
 |-------------|----------|
 | 1.0-1.5 | 12.6% |
 | 1.5-2.0 | 23.5% |
 | 2.0+ | 35.1% |
 
 ---
 
 ## 3. ANYTIME LEAD
 
 ### Model Features
 - Uses CTMC (Continuous-Time Markov Chain)
 - Derived from lambda_home, lambda_away via matrix exponentials
 - h_1up = P(home leads by 1+ at ANY point)
 - h_2up = P(home leads by 2+ at ANY point)
 - a_1up = P(away leads by 1+ at ANY point)
 - a_2up = P(away leads by 2+ at ANY point)
 
 ### Derivation
 ```python
 from src.pricing.markov import MarkovPricer
 pricer = MarkovPricer(max_goals=8)
 probs = pricer.calculate_lead_probs(lambda_home, lambda_away)
 ```
 
 ### Data Gap
 - We don't have timeline data to validate anytime lead predictions
 - Need match events with timestamps to determine if team ever led by N
 
 ---
 
 ## 4. CORNERS
 
 ### Model Features
 - Dedicated model (totals-first architecture)
 - Features: league_regime, rolling_corners, style, possession, xG
 - Predicts: total_mu (expected corners), home_share
 
 ### Reverse Engineering Findings
 
 #### Total Corners by Lambda Sum
 | Lambda_Sum | Avg Corners | Over 9.5% | Over 8.5% |
 |-------------|-------------|-----------|-----------|
 | 1.0-2.0 | 8.9 | 40.4% | 53.2% |
 | 2.0-2.5 | 9.5 | 47.4% | 59.1% |
 | 2.5-3.0 | 9.6 | 49.4% | 61.6% |
 | 3.0-3.5 | 9.7 | 51.2% | 61.6% |
 | 3.5+ | 10.0 | 56.2% | 68.4% |
 
 #### Home Corners by Lambda_Home
 | Lambda_Home | Avg Home Corners |
 |--------------|------------------|
 | 0.5-1.0 | 4.0 |
 | 1.0-1.5 | 4.9 |
 | 1.5-2.0 | 5.6 |
 | 2.0-2.5 | 6.2 |
 | 2.5+ | 7.3 |
 
 ### Formula
 ```
 expected_corners = 9.5 + 0.3 * (lambda_sum - 3.0)
 home_corners = 5.0 + 1.5 * (lambda_home - 1.5)
 away_corners = 5.0 + 1.5 * (lambda_away - 1.5)
 ```
 
 ---
 
 ## 5. MULTI-FEATURE COMBINATIONS
 
 ### 1X2 Away Win
 | Features | Fixtures | Win Rate |
 |----------|----------|----------|
 | lambda >= 1.5, away_xg_net >= 0 | 2,615 | 20.2% |
 | lambda >= 1.5, away_xg_net >= 0.5 | 1,420 | 18.5% |
 
 ### AH Away +0.5
 | Features | Fixtures | Win Rate |
 |----------|----------|----------|
 | lambda >= 1.5, away_xg_net >= 0 | 2,615 | 42.2% |
 | lambda >= 1.5, away_xg_net >= 0.5 | 1,420 | 39.6% |
 
 ### Over 9.5 Corners
 | Features | Fixtures | Win Rate |
 |----------|----------|----------|
 | lambda_sum >= 2.5, xg_sum >= 2.5 | 4,220 | 50.3% |
 | lambda_sum >= 3.0, xg_sum >= 3.0 | 929 | 51.8% |
 | lambda_sum >= 3.5, xg_sum >= 2.5 | 225 | 54.2% |
 
 ---
 
 ## 6. KEY INSIGHTS
 
 ### Lambda is the Backbone
 - All 4 markets derive from lambda (expected goals)
 - Lambda thresholds are interpretable and actionable
 - Higher lambda = higher scoring rate = more goals/corners/leads
 
 ### Feature Interactions
 - Lambda + xG_net improves predictions
 - Lambda_sum + xG_sum improves corners predictions
 - Style and form features need more investigation
 
 ### Data Gaps
 1. **Anytime Lead**: Need timeline data to validate
 2. **AH/EH Odds**: Limited coverage (642 fixtures vs 8,781)
 3. **Better Odds**: Need Pinnacle/Bet365 for proper ROI analysis
 
 ---
 
 ## 7. NEXT STEPS
 
 1. **Get timeline data** for anytime lead validation
 2. **Get better odds data** for AH/EH/Corners
 3. **Build formulas** for each market
 4. **Integrate into daily report** for forward testing
---

## 8. LAMBDA RATIO (THE CLEANEST METRIC)

Lambda ratio = lambda_away / lambda_home

| Ratio | Away Win Rate | AH +0.5 Win Rate |
|-------|---------------|-----------------|
| 0.5-0.8 | 23.5% | 48.1% |
| 0.8-1.0 | 32.5% | 59.7% |
| 1.0-1.2 | 39.3% | 68.0% |
| 1.2-1.5 | 46.6% | 72.7% |
| 1.5-3.0 | **57.6%** | **81.4%** |

### AH +0.5 with Odds (Small Sample)
| Ratio | Bets | Win Rate | ROI |
|-------|------|----------|-----|
| 0.8-1.0 | 4 | 75.0% | +43.8% |
| 1.0-1.2 | 10 | 80.0% | **+50.3%** |
| 1.2-1.5 | 4 | 75.0% | +39.4% |
| 1.5-3.0 | 8 | 87.5% | **+68.8%** |

---

## 9. FINAL FORMULAS

### 1X2 Away Win
`
lambda_ratio = lambda_away / lambda_home
IF lambda_ratio >= 1.5 AND odds >= 2.0:
    p_win = 57.6%
`

### AH Away +0.5
`
IF lambda_ratio >= 1.0 AND odds >= 1.85:
    p_win = 68-80%
IF lambda_ratio >= 1.5:
    p_win = 81%
`

### AH Away +1.5
`
IF lambda_away >= 1.0:
    p_win = 80%
IF lambda_away >= 1.5:
    p_win = 87%
`

### Over 9.5 Corners
`
lambda_sum = lambda_home + lambda_away
IF lambda_sum >= 3.5:
    p_over = 56%
`

### EH Away -1 (wins by 2+)
`
IF lambda_ratio >= 1.5:
    p_win = 30%
`
