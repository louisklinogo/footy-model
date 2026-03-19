# Next Steps: Using and Maintaining the System

## Immediate Next Steps

### 1. Validate New Markets (1X2, O/U, BTTS)

The new markets were added to settlement but are NOT yet in HIGH_PERFORMANCE_MARKETS.

**Why:** They need to prove themselves in backtests first.

**How to validate:**
```bash
# Run backtest specifically looking for these markets
python src/betting/backtest_match_profiles.py --since-days 30 --strategy balanced
```

**To enable:** Once a market shows 80%+ win rate, add to `HIGH_PERFORMANCE_MARKETS`:
```python
HIGH_PERFORMANCE_MARKETS = {
    "ah2_home_p05",
    "ah2_home_m05",
    "1x2_h",  # Add if 80%+ win rate
    "btts_yes",  # Add if 80%+ win rate
}
```

### 2. Run Daily Predictions

Set up a daily job to generate accas:

```bash
# In the morning, run:
python src/betting/acca_builder.py --today --strategy conservative --output json
```

Output will show:
- Which fixtures have opportunities
- Story signals (why we're betting)
- Combined odds
- Model probability

### 3. Track Results

After matches finish, resolve accas:

```bash
# Run backtest to see how today's accas did
python src/betting/backtest_match_profiles.py --since-days 1 --strategy conservative
```

## Weekly Maintenance

### Monday: Review Last Week

```bash
# 7-day backtest
python src/betting/backtest_match_profiles.py --since-days 7 --strategy conservative
```

Review:
- Which signals won/lost?
- Which markets won/lost?
- Any patterns?

### Tuesday: Update Filters

Based on Monday's review:

1. **If a signal dropped below 70%:** Move to AVOID_SIGNALS
2. **If a market dropped below 80%:** Remove from HIGH_PERFORMANCE_MARKETS
3. **If a new signal/market is performing well:** Add to appropriate list

### Wednesday: Stress Test

```bash
# Run with different strategies
python src/betting/backtest_match_profiles.py --since-days 30 --strategy ultra
python src/betting/backtest_match_profiles.py --since-days 30 --strategy balanced
python src/betting/backtest_match_profiles.py --since-days 30 --strategy aggressive
```

Compare:
- Ultra: Higher win rate, fewer accas
- Balanced: Middle ground
- Aggressive: Lower win rate, more volume

## Monthly Maintenance

### 1. Full Backtest (90 days)

```bash
python src/betting/backtest_match_profiles.py --since-days 90 --strategy conservative
```

Look for:
- Seasonal patterns
- League-specific patterns
- Model drift

### 2. Model Performance Review

Compare GBM vs V2:

```bash
# See match_profile.py for the comparison
python -c "
from src.betting.match_profile import build_match_profile
profile = build_match_profile(20360)
print(profile.predictions)
"
```

### 3. Refresh Model Preference

If V2 starts outperforming GBM on AH2:

1. Edit `src/betting/match_profile.py`
2. Update `GBM_PREFERRED_PREFIXES` or `V2_PREFERRED`
3. Run backtest to validate

## Adding New Features

### Add a New Story Signal

1. **Identify the pattern:** What data indicates this signal?
2. **Add to story_generator.py:** Generate the signal
3. **Add alignment rules:** Map to markets in opportunity_finder.py
4. **Test:** Run backtest, check if HIGH_PERFORMANCE

Example:
```python
# In story_generator.py
if home_position <= 3 and away_position >= 15:
    signals.append("top_vs_bottom")

# In opportunity_finder.py
"top_vs_bottom": {"aligned": ["ah2_home_m15", "1x2_h"], "contradictory": ["1x2_a"]},
```

### Add a New Market

1. **Add settlement logic:** `src/betting/settlement.py`
2. **Add to supported_market_codes:** List the market
3. **Add to SETTLEABLE_MARKETS:** `opportunity_finder.py`
4. **Add alignment rules:** Map signals to market
5. **Test:** Run backtest

## Monitoring & Alerts

### Set up alerts for:

1. **Win rate drops below 60%:** Review immediately
2. **No accas for 3 days:** Check if filters too restrictive
3. **New market shows 80%+ win rate:** Consider adding to HIGH_PERFORMANCE

### Dashboards to consider:

1. **Daily:** Accas generated, fixtures covered
2. **Weekly:** Win rate by signal, by market
3. **Monthly:** ROI, bankroll growth, drawdown

## Known Limitations

1. **Only 2 markets are HIGH_PERFORMANCE:** `ah2_home_p05`, `ah2_home_m05`
2. **New markets (1X2, O/U, BTTS) need validation**
3. **System is conservative:** Might miss opportunities
4. **Requires story alignment:** Won't bet on random high-prob picks

## Future Improvements

1. **League-specific signals:** Different signals for different leagues
2. **Odds movement:** Incorporate closing odds vs opening odds
3. **In-play signals:** Use live data for better predictions
4. **Weather/injuries:** Add external factors
5. **Auto-update HIGH_PERFORMANCE lists:** Based on rolling performance