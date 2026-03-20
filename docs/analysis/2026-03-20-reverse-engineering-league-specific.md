 # League-Specific Reverse Engineering Analysis
 **Date:** 2026-03-20
 
 ---
 
 ## 1. 1X2 AWAY WIN BY LAMBDA RATIO - BY LEAGUE
 
 Lambda ratio = lambda_away / lambda_home
 
 | League | Ratio >= 1.5 Fixtures | Win Rate | Notes |
 |--------|----------------------|----------|-------|
 | D1 (Denmark) | 14 | **78.6%** | Very predictable |
 | I1 (Italy) | 35 | **74.3%** | Strong away dominance |
 | F1 (France) | 24 | **75.0%** | Strong away dominance |
 | P1 (Portugal) | 37 | **73.0%** | Strong away dominance |
 | E3 (Eng L2) | 17 | **70.6%** | Good value |
 | EC (Conf) | 42 | **69.0%** | Good sample |
 | CL (Champs Lge) | 44 | **63.6%** | Elite teams |
 | E1 (Eng PL) | 21 | **57.1%** | Competitive |
 | **PL1 (Poland)** | 12 | **16.7%** | AVOID - very low |
 | **E2 (Eng Champ)** | 14 | **35.7%** | AVOID - underperforms |
 
 **Key Insight:** Poland and English Championship underperform significantly. Denmark, Italy, France, Portugal are most predictable.
 
 ---
 
 ## 2. ANYTIME LEAD - VALIDATED WITH ACTUAL DATA
 
 From `fixture_incident_lead_states` table (7,654 fixtures).
 
 ### Away Led by 1+ (Anytime)
 | Lambda_Away | Actual Rate |
 |-------------|--------------|
 | 0.5-1.0 | 32.5% |
 | 1.0-1.5 | 49.2% |
 | 1.5-2.0 | **66.0%** |
 | 2.0+ | **74.5%** |
 
 ### Away Led by 2+ (Anytime)
 | Lambda_Away | Actual Rate |
 |-------------|--------------|
 | 0.5-1.0 | 9.0% |
 | 1.0-1.5 | 19.0% |
 | 1.5-2.0 | **32.4%** |
 | 2.0+ | **44.6%** |
 
 ### Home Led by 1+ (Anytime)
 | Lambda_Home | Actual Rate |
 |-------------|--------------|
 | 0.5-1.0 | 39.2% |
 | 1.0-1.5 | 52.3% |
 | 1.5-2.0 | **69.2%** |
 | 2.0-2.5 | **79.3%** |
 | 2.5+ | **87.4%** |
 
 ### Home Led by 2+ (Anytime)
 | Lambda_Home | Actual Rate |
 |-------------|--------------|
 | 0.5-1.0 | 12.5% |
 | 1.0-1.5 | 19.8% |
 | 1.5-2.0 | **33.6%** |
 | 2.0-2.5 | **50.1%** |
 | 2.5+ | **62.2%** |
 
 ---
 
 ## 3. CORNERS BY LEAGUE
 
 | League | Avg Corners | Over 9.5% | Lambda Sum |
 |--------|-------------|-----------|------------|
 | N1 (Netherlands) | 10.4 | **57%** | 3.27 |
 | EC (Conference) | 10.2 | **57%** | 2.80 |
 | SC0 (Scotland) | 10.0 | **58%** | 2.72 |
 | B1 (Belgium) | 10.2 | **56%** | 2.75 |
 | E1 (Eng PL) | 10.2 | **55%** | 2.57 |
 | I1 (Italy) | 8.8 | 40% | 2.52 |
 | G1 (Greece) | 8.7 | 41% | 2.65 |
 | JP1 (Japan) | 7.9 | 40% | 2.49 |
 
 **Key Insight:** Netherlands, Belgium, Scotland, England have high corners. Italy, Greece, Japan have low corners.
 
 ---
 
 ## 4. LEAGUE-SPECIFIC FORMULAS
 
 ### 1X2 Away Win - Best Leagues
 ```
 IF league IN ('D1', 'I1', 'F1', 'P1') AND lambda_ratio >= 1.5:
     p_win = 73-79%
 
 IF league IN ('E3', 'EC') AND lambda_ratio >= 1.5:
     p_win = 69-71%
 
 AVOID: PL1 (Poland), E2 (Eng Champ) - underperform
 ```
 
 ### Corners Over 9.5 - Best Leagues
 ```
 IF league IN ('N1', 'EC', 'SC0', 'B1', 'E1') AND lambda_sum >= 3.0:
     p_over = 55-58%
 
 AVOID: I1, G1, JP1 - low corners leagues
 ```
 
 ### Anytime Lead (Validated)
 ```
 Away 1up: lambda_away >= 2.0 → p = 74.5%
 Away 2up: lambda_away >= 2.0 → p = 44.6%
 Home 1up: lambda_home >= 2.5 → p = 87.4%
 Home 2up: lambda_home >= 2.5 → p = 62.2%
 ```
 
 ---
 
 ## 5. NEXT STEPS
 
 1. Get odds data by league
 2. Calculate ROI by league
 3. Build league-specific filters into daily report
 4. Forward test on upcoming fixtures