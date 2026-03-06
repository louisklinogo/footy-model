# Small-Bankroll Accumulator Strategy Tasks
Date: 2026-03-06
Owner: Modeling / Betting
Goal: Convert current model strengths into a defendable `100 cedis` accumulator campaign.

## Strategy Decision
1. We are not building a new betting engine.
2. We are building a strategy layer on top of the existing walk-forward accumulator backtester.
3. Primary objective is not generic long-run ROI.
4. Primary objective is step-up bankroll target hit probability before ruin.

## What Already Exists
1. [x] Walk-forward accumulator backtester in `src/betting/backtest_accumulators.py`.
2. [x] Bankroll path simulation with stake fraction and drawdown reporting.
3. [x] Correlation controls: same fixture, same team, and same-league caps.
4. [x] Market gating and risk assessment flow for tradable vs predict-only separation.

## New Strategy Layer
1. [x] Create a dedicated small-bankroll accumulator policy file.
2. [x] Create a small-bankroll market whitelist with `tradable`, `watchlist`, and `blocked`.
3. [x] Create a small-bankroll campaign runner with bootstrap milestone simulation.
4. [x] Run first campaign backtest with current promoted model stack.
5. [x] Review resolved market universe and prune any weak slices that still leak into ticket construction. (`o15` is the only currently tradable market with usable scored odds coverage.)
6. [ ] Decide whether to include `watchlist` markets in a second aggressive profile after odds coverage improves.

## Initial Tradable Intent
1. Corners: stronger team corners only.
2. Anytime: `h_1up`, `h_2up`, `a_2up`; hold `a_1up` in watchlist.
3. Scoreline-derived: non-draw, high-stability legs only.

## First Campaign Outcome
1. Actual tradable universe collapsed to `o15` only because corners, anytime, and double-chance markets do not currently have usable scored odds coverage in `prediction_scores`.
2. `o15` was marked `limited_to_one_leg` in gating, which blocked all multi-leg tickets until the campaign policy explicitly allowed repeated `o15` across different fixtures.
3. First working campaign result:
   1. slip size: `2`
   2. slip count: `4`
   3. hit rate: `0.75`
   4. ROI: `0.4391`
   5. ending bankroll: `103.59`
4. Bootstrap milestone output is low-support because only `4` slips were observed; use it directionally, not as decision-grade evidence.

## Repair Queue
1. [x] Repair historical scoring so `prediction_scores` backfills odds for already-scored settled rows when odds arrived later.
2. [x] Rebuild the market universe after scoring repair; `1x2` and `dc` expanded materially, corners partially expanded.
3. [x] Re-run campaign after repair before widening whitelist or changing strategy assumptions.

## Post-Repair State
1. [x] Default tradable core promoted to `dc_1x`, `dc_x2`, `o15`.
2. [x] New default campaign result:
   1. slip size: `2`
   2. slip count: `109`
   3. hit rate: `0.6055`
   4. ROI: `0.3219`
   5. ending bankroll: `243.79`
3. [x] Support is now adequate for bootstrap interpretation (`109` observed slips).
4. [ ] Review whether `c85` and `u35` should be promoted from watchlist after further support or stricter sub-filters.

## Acceptance
1. Campaign runner writes JSON/MD/CSV artifacts.
2. Campaign report shows:
   1. walk-forward slip metrics
   2. bankroll ruin probability
   3. milestone hit probability
3. Strategy uses explicit whitelist intersection with eligible markets.
4. Remaining model work is re-prioritized based on campaign evidence, not generic metric instinct.
