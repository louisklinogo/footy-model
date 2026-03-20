# Betting Strategy Analysis
**Date:** 2026-03-20
**Model:** scoreline_v2 (Scoreline with max_goals=15)

---

## Key Findings

### 1. Model Calibration: GOOD
| Confidence | Actual Win Rate |
|------------|------------------|
| 70%+ | 70.6% |
| 60-70% | 62.6% |
| 50-60% | 55.3% |
| 40-50% | 47.2% |
| <40% | 27.2% |

**Conclusion:** Model is well-calibrated. Predictions match reality.

---

### 2. Bookmaker Margin: ~3%
| Market | Actual Win Rate | Implied Win Rate | Gap |
|--------|-----------------|------------------|-----|
| 1x2_h | 43.8% | 46.7% | -2.9% |
| 1x2_d | 25.2% | 27.3% | -2.1% |
| 1x2_a | 31.0% | 33.4% | -2.4% |

**Conclusion:** Bookmakers have built-in margin. Need to beat odds by ~3% to profit.

---

### 3. The Sweet Spot: AWAY WINS

**Rule:**
`
IF p_model >= 50% AND odds >= 2.5 AND market = '1x2_a':
    BET
`

**Results:**
| Market | Bets | Win Rate | Avg Odds | Profit/Bet |
|--------|------|----------|----------|------------|
| 1x2_a (Away) | 57 | 42.1% | 3.50 | +52.2% |
| 1x2_h (Home) | 138 | 31.9% | 3.04 | -1.1% |

**Combined (p >= 50%, odds >= 2.5):**
- 195 bets, 67 wins (34.4%), **+13.1% ROI**

---

### 4. Confidence + Odds Matrix

| Confidence | Odds Range | Bets | Win% | Profit |
|------------|------------|------|------|--------|
| 60-70% | >= 2.5 | 28 | 39.3% | +25.9% |
| 50-60% | >= 2.5 | 167 | 33.5% | +11.0% |
| < 50% | any | 17,598 | 24.5% | -10.1% |

---

## Betting Rules (Tentative)

### Rule 1: Away Win Value Bets
`
IF market = '1x2_a' AND p_model >= 50% AND odds >= 2.5:
    BET (stake: 1 unit)
`
**Expected ROI: +52%**

### Rule 2: High Confidence Value Bets
`
IF p_model >= 60% AND odds >= 2.5:
    BET (stake: 1 unit)
`
**Expected ROI: +26%**

### Rule 3: General Value Bets
`
IF p_model >= 50% AND odds >= 2.5:
    BET (stake: 1 unit)
`
**Expected ROI: +13%**

---

## Caveats

1. **Sample size is small** - 57 bets for away wins, 195 for general rule
2. **Odds data quality** - SofaScore odds, not real market odds (Pinnacle/Bet365)
3. **Needs validation** - Forward test on new fixtures before committing real money

---

## Next Steps

1. Get better odds data (Pinnacle closing lines)
2. Forward test on upcoming fixtures
3. Track results weekly
4. Validate/refine rules after 100+ bets

---

## Data Source

- Model: scoreline_v2
- Fixtures: 8,781
- Fixtures with odds: ~1,000
- Date range: 2023-09 to 2026-03
