# Betting Strategy Analysis
**Date:** 2026-03-20
**Model:** scoreline_v2 + lambda_xgb

---

## Key Findings

### 1. Model is Calibrated
| Confidence | Actual Win Rate |
|------------|------------------|
| 70%+ | 70.6% |
| 60-70% | 62.6% |
| 50-60% | 55.3% |

**Conclusion:** Model predictions match reality.

---

### 2. Bookmaker Margin: ~3%
| Market | Actual | Implied | Gap |
|--------|--------|---------|-----|
| 1x2_h | 43.8% | 46.7% | -2.9% |
| 1x2_d | 25.2% | 27.3% | -2.1% |
| 1x2_a | 31.0% | 33.4% | -2.4% |

---

### 3. THE SWEET SPOT: LAMBDA-BASED RULES

#### Rule A: Lambda_Away >= 2.0 (STRONGEST)
`
IF lambda_away >= 2.0 AND odds >= 2.0:
    BET on away win
`
**Results:**
- 80 fixtures, 45.0% win rate, avg odds 3.11
- **ROI: +41.8%**
- Avg p_model: 46.3%

#### Rule B: Away Ratio >= 1.5
`
IF (lambda_away / lambda_home) >= 1.5 AND odds >= 2.0:
    BET on away win
`
**Results:**
- 231 fixtures, 45.0% win rate, avg odds 2.82
- **ROI: +21.2%**

#### Rule C: Lambda_Away 1.5-2.0
`
IF lambda_away >= 1.5 AND odds >= 2.0:
    BET on away win
`
**Results:**
- 663 fixtures, 40.1% win rate, avg odds 3.16
- **ROI: +11.2%**

---

### 4. LAMBDA -> P_MODEL CONVERSION

| Lambda Away | Avg P_Model | Win Rate | Profit |
|-------------|-------------|----------|--------|
| >= 2.0 | 46.3% | 45.0% | +41.8% |
| 1.5-2.0 | 36.3% | 40.1% | +11.2% |
| 1.2-1.5 | 31.1% | 32.5% | -4.4% |
| < 1.2 | 24.6% | 21.6% | -16.7% |

**Insight:** Lambda_away >= 2.0 corresponds to p_model ~46%.
The lambda threshold is a simpler, more interpretable rule.

---

### 5. SCORELINE-BASED RULES (for comparison)

#### Rule D: p_model >= 50%, odds >= 2.5
`
IF p_model >= 50% AND odds >= 2.5 AND market = '1x2_a':
    BET on away win
`
**Results:**
- 57 fixtures, 42.1% win rate, avg odds 3.50
- **ROI: +52.2%**

---

## Summary: Ranking Rules by ROI

| Rule | Fixtures | Win% | Odds | ROI |
|------|----------|------|------|-----|
| lambda_away >= 2.0, odds >= 2.0 | 80 | 45.0% | 3.11 | **+41.8%** |
| p_model >= 50%, odds >= 2.5 | 57 | 42.1% | 3.50 | **+52.2%** |
| away ratio >= 1.5, odds >= 2.0 | 231 | 45.0% | 2.82 | **+21.2%** |
| lambda_away >= 1.5, odds >= 2.0 | 663 | 40.1% | 3.16 | **+11.2%** |

---

## Caveats

1. **Sample sizes small** (57-663 fixtures)
2. **Odds data quality** - SofaScore, not Pinnacle
3. **Needs forward validation** before real money

---

## Next Steps

1. Get better odds data (Pinnacle closing lines)
2. Forward test on upcoming fixtures
3. Track weekly, validate after 100+ bets
4. Consider combining lambda + p_model rules

---

## Data Source

- Models: scoreline_v2, lambda_xgb
- Fixtures: 8,781
- Fixtures with odds: ~1,000
- Date range: 2023-09 to 2026-03
