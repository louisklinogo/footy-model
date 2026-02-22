# New Model Blueprint (Profit-First, Fixtures-First)

Date: 2026-02-20

This document is the single source of truth for the next-generation model and trading system.

## Goal

Build a deployable football betting system optimized for long-run positive EV and risk-adjusted returns.

Core principle: one coherent match model -> multiple market probabilities -> EV -> risk-managed bet selection.

## Scope

IN (Phase 1)
- Leagues: start with a quality-gated set (expected: `E0,E1,E2,E3,D1,D2,I1,SC0`).
- Markets: start with O/U 2.5, then Asian Handicap.
- Decision-time operation: trade only inside defined windows close to kickoff.

OUT (Phase 1)
- Long-horizon betting (>72h before kickoff) unless explicitly launched as a separate, lower-confidence product.
- SportyBet 1X2/1UP production trading until SportyBet-specific pricing capture and rule-set support are implemented and validated.
- Narrative/subjective psychology features.

## Data Contracts (Non-Negotiable)

- Every feature is point-in-time: for any prediction at decision time `as_of_ts`, every upstream source timestamp must satisfy `source_ts <= as_of_ts`.
- No post-kickoff inputs in training or prediction. Any detected post-kickoff odds/stat snapshots are hard failures.
- Placeholder premium stats (`NULL` and/or obvious placeholders like `0,0` xG) must be excluded or explicitly flagged and never treated as genuine match performance.

Primary DB tables relied upon
- `fixtures`, `fixture_results`
- `fixture_stats_premium` (post-match ground truth; do not use as pre-match features)
- `fixture_odds_snapshots` (pre-match snapshots; multiple snapshot types)
- `team_premium_snapshots` (pre-match rolling team snapshots)
- `execution_logs` (immutable ledger of model probabilities vs bookmaker odds at exactly the `decision_window` for true CLV/EV tracking)

Optional (Phase 2)
- Lineups / absences (coverage is inconsistent; treat as optional alpha overlay)

## Decision Windows and Tradability

We operate multiple decision windows. Each window has its own dataset build, model run, and evaluation.

Default windows (adjustable)
- `T-60m`
- `T-15m`
- `T-5m`

Tradability rules
- If required odds are missing at the cutoff for a window, the fixture is `NOT_TRADABLE` for that window.
- Do not forward-fill odds from a later snapshot into an earlier window.
- If a league fails readiness gates (coverage/fidelity), it is `NOT_TRADABLE` until it passes.

Readiness gate references
- `scrapers/check_premium_readiness.py` (coverage + fidelity thresholds)
- `daily_pipeline.py` (orchestrates readiness -> snapshots -> predict -> export -> score)

## Modeling Architecture

### Core targets (foundation)

- Predict expected goal intensities using **XGBoost / LightGBM Regressors** (We use tree-based ML over traditional Bayesian/Gaussian dynamic state-space models to better capture non-linear feature interactions):
  - `lambda_home`
  - `lambda_away`

Rationale: a single coherent latent state supports consistent probabilities across O/U, AH, BTTS, DNB, and 1X2.

### Probability engine

- Score distribution: Poisson baseline with Dixon-Coles correction for low-score dependence.
- From score distribution, derive:
  - O/U probabilities for configured totals
  - BTTS
- Advanced Market Modeling (Phase 2+):
  - **Continuous-Time Markov Chains (CTMC)**: Simulate minute-by-minute transition states (0-0 -> 1-0) to accurately price SportyBet 1UP early-payout markets.
  - **Monte Carlo Simulations**: Simulate the `lambda` outputs 10,000+ times to calculate probabilities for complex Asian Handicaps and correlated bets where standard calculus breaks down.

### Feature sets

P0 (must-have, high coverage)
- `team_premium_snapshots` rolling features (attack/defense proxies)
- schedule proxies derivable from fixtures (days rest, congestion)
- odds snapshot features available at the decision window cutoff (O/U and AH ladders)

P1 (add only if stable lift)
- market microstructure: dispersion across books, line movement, opening-to-cutoff movement
- segmentation: league/season/liquidity buckets; optional team-style clusters derived from rolling stats

P2 (optional alpha, never mandatory)
- lineup and absences overlay (coverage gated by league and hours-to-kickoff)
- H2H features only with minimum sample + strong regularization (default expectation: low ROI)

## Evaluation and Promotion Gates

Primary metrics (profit-first)
- CLV (per window, per league, per market)
- realized EV and ROI after vig and estimated execution costs

Secondary metrics (model health)
- log loss, Brier
- calibration slope/intercept and reliability curves

Promotion gate (default)
- Coverage threshold: >= 70% of fixtures in the target window are tradable
- Stability: CLV positive across at least 2 rolling periods (e.g., month-on-month) and not dominated by one league
- Calibration: acceptable drift bounds for key odds buckets

Feature inclusion gate
- Coverage passes threshold in the target decision window
- No leakage against declared as-of timestamp
- Out-of-sample lift in CLV/ROI is stable across leagues/seasons

## Risk and Execution Policy (Phase 1 defaults)

- Start with fractional Kelly: 0.10 to 0.25
- Hard caps:
  - max stake per fixture
  - max exposure per day
  - correlated exposure cap across the same fixture (avoid stacking highly correlated markets)
- Kill switches:
  - disable a league/market slice if CLV turns negative beyond a threshold over a rolling sample
  - reduce stake fraction on drawdown triggers

## Roadmap (Practical)

1) Data readiness
- Ensure `fixture_results` completeness for settled FT fixtures.
- Ensure odds snapshots are refreshed inside the target windows.
- Enforce point-in-time contracts end-to-end.

2) Baseline model
- Build and validate `lambda_home/lambda_away`.
- Implement score distribution + O/U 2.5 probabilities.
- Add AH probabilities next.

3) Trading simulation
- Windowed backtests with as-of snapshots.
- CLV + EV logging and reporting.

4) Web App & API Integration
- Expose the finalized model probabilities, odds, and EV edges via a clean `src/api` layer (e.g., FastAPI).
- The API must support querying the `execution_logs` and live predictions to strictly serve the downstream client-facing web application.

5) Controlled pilot
- Limited leagues and one market family.
- Expand only after promotion gates pass.

## Open Questions (to answer before implementation)

- SportyBet: confirm exact 1UP rule-set and eligibility (market types, leagues, singles/accas, void/abandoned).
- SportyBet: how will we capture the actual price you can bet (SportyBet odds) at each decision window?
- Exact launch universe: confirm leagues to include/exclude.
- Confirm decision windows and whether any >72h product is desired.
