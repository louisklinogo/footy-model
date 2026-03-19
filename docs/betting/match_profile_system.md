# Match Profile Generator & Backtesting System

## Overview

The Match Profile system generates betting opportunities by:
1. Pulling all data for a fixture (teams, form, predictions, odds)
2. Generating a "story" about the match (style clash, form, expected scoreline)
3. Finding markets where the story aligns with predictions
4. Building accumulators with high win rate
5. Validating historically via backtest

## Architecture

```
Fixture ID
    |
    v
match_profile.py --> Pull data, predictions (best model per market)
    |
    v
story_generator.py --> Generate narrative (style, form, position, scoreline)
    |
    v
opportunity_finder.py --> Rank markets by story alignment
    |
    v
acca_builder.py --> Build accumulators (4 strategies)
    |
    v
backtest_match_profiles.py --> Walk-forward validation
```

## Key Files

| File | Purpose |
|------|---------|
| `src/betting/match_profile.py` | Pull fixture data, predictions |
| `src/betting/story_generator.py` | Generate match narrative |
| `src/betting/opportunity_finder.py` | Find betting opportunities |
| `src/betting/acca_builder.py` | Build accumulators |
| `src/betting/backtest_match_profiles.py` | Validate historically |
| `src/betting/settlement.py` | Settle markets (84 supported) |

## Strategy Guide

| Strategy | Prob Range | Story Aligned | Max Legs | Use Case |
|----------|------------|---------------|----------|----------|
| ULTRA | 80%+ | Required | 2 | Maximum safety |
| CONSERVATIVE | 70%+ | Required | 3 | **Default** |
| BALANCED | 65-80% | Required | 3 | More volume |
| AGGRESSIVE | 60-70% | Required | 3 | Higher odds |

## Model Preference

The system uses the best model per market family:

| Market Family | Model | Reason |
|----------------|-------|--------|
| Anytime (h_1up, a_1up) | V2 | Better calibrated |
| AH2 markets | GBM | Slightly better |
| Corners | V2 | Slightly better |
| Default | V2 | Canonical model |

**Location:** `src/betting/match_profile.py` - `_fetch_predictions()`

## Story Signals

### HIGH_PERFORMANCE_SIGNALS (70%+ win rate)

```python
HIGH_PERFORMANCE_SIGNALS = {
    "title_race_home",      # 85.7% win rate
    "expected_2_0",         # 100% win rate
    "expected_1_0",         # 83.3% win rate
    "relegation_battle",    # 85.7% win rate
    "expected_2_1",         # 100% win rate
    "hot_home_vs_cold_away", # 100% win rate
    "home_dominant",        # Needs validation
}
```

### AVOID_SIGNALS (below 70% win rate)

```python
AVOID_SIGNALS = {
    "expected_0_1",  # 60% win rate
    "expected_1_1",  # 66.7% win rate
    "cold_vs_cold",  # 66.7% win rate
}
```

**Location:** `src/betting/opportunity_finder.py`

## Market Alignment

### HIGH_PERFORMANCE_MARKETS (80%+ win rate)

```python
HIGH_PERFORMANCE_MARKETS = {
    "ah2_home_p05",  # 90% win rate
    "ah2_home_m05",  # High win rate
}
```

### STORY_MARKET_ALIGNMENT Rules

| Signal | Aligned Markets | Contradictory |
|--------|-----------------|---------------|
| `title_race_home` | `ah2_home_m05`, `ah2_home_p05`, `1x2_h` | `ah2_away_m05`, `1x2_a` |
| `expected_2_0` | `ah2_home_m15`, `1x2_h`, `ou_1.5_over`, `btts_no` | `1x2_a`, `btts_yes` |
| `expected_1_0` | `ah2_home_m05`, `ah2_home_p05`, `1x2_h`, `ou_2.5_under` | `1x2_a`, `btts_yes` |
| `expected_1_1` | `ah2_home_p05`, `ah2_away_p05`, `1x2_d`, `btts_yes` | `1x2_h`, `1x2_a` |
| `hot_vs_hot` | `ou_2.5_over`, `ou_3.5_over`, `btts_yes` | `ou_2.5_under`, `btts_no` |

**Location:** `src/betting/opportunity_finder.py`

## Supported Markets (84 total)

| Category | Markets |
|----------|---------|
| **AH2** | `ah2_home_m05`, `ah2_home_p05`, `ah2_home_m15`, `ah2_home_p15`, `ah2_away_*` |
| **AH** | `ah_h05`, `ah_a05`, `ah_h15`, `ah_a15` |
| **1X2** | `1x2_h`, `1x2_d`, `1x2_a` |
| **O/U** | `ou_0.5_*` through `ou_5.5_*` |
| **BTTS** | `btts_yes`, `btts_no` |
| **Corners** | `c75`, `c85`, `hc25`, `ac25`, etc. |
| **Incident** | `h_1up`, `a_1up`, `h_2up`, `a_2up` |

**Location:** `src/betting/settlement.py` - `supported_market_codes()`

## Daily Operations

### Run predictions for today's fixtures

```bash
# Build profiles for all today's fixtures
python src/betting/match_profile.py --today

# Build accas for today
python src/betting/acca_builder.py --today --strategy conservative
```

### Check a specific fixture

```bash
# Full profile
python src/betting/match_profile.py --fixture 20360 --output markdown

# Story
python src/betting/story_generator.py --fixture 20360

# Opportunities
python src/betting/opportunity_finder.py --fixture 20360
```

### Run backtest

```bash
# Last 7 days
python src/betting/backtest_match_profiles.py --since-days 7 --strategy conservative

# Last 30 days
python src/betting/backtest_match_profiles.py --since-days 30 --strategy conservative
```

## Weekly Maintenance

### 1. Run backtest to validate performance

```bash
python src/betting/backtest_match_profiles.py --since-days 30 --strategy conservative
```

### 2. Review results

Check:
- Win rate by story signal
- Win rate by market
- Any new signals/markets performing well

### 3. Update HIGH_PERFORMANCE lists

If a signal/market shows 70%+ win rate over 30+ days:

1. Edit `src/betting/opportunity_finder.py`
2. Add to `HIGH_PERFORMANCE_SIGNALS` or `HIGH_PERFORMANCE_MARKETS`
3. Commit and push

### 4. Add new markets to settlement

If you need a new market:

1. Add settlement logic to `src/betting/settlement.py`
2. Add to `supported_market_codes()`
3. Add to `SETTLEABLE_MARKETS` in `opportunity_finder.py`
4. Add alignment rules to `STORY_MARKET_ALIGNMENT`
5. Run backtest to validate

## Troubleshooting

### Win rate dropped

1. Check if any signals moved from HIGH_PERFORMANCE to AVOID
2. Check if market alignment rules changed
3. Run backtest with different date ranges

### No accas generated

1. Check if HIGH_PERFORMANCE_MARKETS is too restrictive
2. Lower probability threshold
3. Try BALANCED or AGGRESSIVE strategy

### Market not being picked

1. Check if market is in SETTLEABLE_MARKETS
2. Check if market has story alignment
3. Check if market is in HIGH_PERFORMANCE_MARKETS

## Backtest Results (30 days)

| Metric | Value |
|--------|-------|
| Win Rate | 71.43% |
| ROI | 434.26% |
| `ah2_home_p05` win rate | 90% |
| Max Drawdown | 1.99% |