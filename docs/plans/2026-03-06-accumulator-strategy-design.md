# Small-Bankroll Accumulator Strategy Design

Date: 2026-03-06
Owner: Modeling / Betting

## Objective
Use the existing model stack to run a defendable accumulator campaign for a `100 cedis` bankroll.

The optimization target is not generic long-run ROI alone. The operating target is the probability of reaching bankroll milestones before ruin, using only whitelisted markets that already have acceptable predictive quality and risk gating.

## Why This Pivot
The repo already has a walk-forward accumulator backtester with bankroll simulation, settlement, and correlation controls. Building another engine would duplicate logic and slow down the actual trading question.

The missing layer is strategic:
1. Which markets are allowed into tickets
2. What policy profile is acceptable for a small bankroll
3. How to evaluate accumulator paths against bankroll milestones

## Recommended Architecture
1. Keep `src/betting/backtest_accumulators.py` as the core historical engine.
2. Add a dedicated small-bankroll policy JSON for tighter leg selection and slightly more aggressive staking.
3. Add a market whitelist JSON with `tradable`, `watchlist`, and `blocked` sets.
4. Add a campaign runner that:
   1. intersects whitelist with eligible gated markets
   2. runs the historical walk-forward backtest
   3. bootstraps ticket outcomes into milestone-hit probability estimates

## Market Universe
Initial tradable universe is intentionally narrow:
1. stronger non-draw double-chance legs
2. stronger anytime legs
3. stronger team-corners and basic goals legs

Draw-heavy or currently weak markets stay blocked or watchlist-only.

## Output Contract
The campaign runner should write:
1. JSON summary
2. Markdown report
3. CSV of realized historical slips
4. JSON of slip payloads

The report should show:
1. resolved market universe
2. walk-forward slip count, hit rate, ROI, drawdown
3. simulated ruin probability
4. milestone hit probabilities

## Decision Rule
After the first campaign run:
1. keep only markets that survive both predictive and ticket-level evidence
2. demote weak slices to watchlist or blocked
3. only then decide whether remaining model improvements are worth immediate engineering effort
