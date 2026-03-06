# Footy Model Re-Architecture Report (Ideal State)
Date: 2026-03-03
Author: Senior DS review

## 1) Direct Conclusion
Yes, the current architecture is structurally wrong for your objective.

The core issue is not a single weak algorithm; it is a design mismatch:
1. one broad feature contract applied across very different markets,
2. per-line binary models where a shared distribution model is more correct,
3. market availability and model design not aligned,
4. deployment policy mixed with model development too early.

A proper redesign is required.

## 2) Goal Definition (What "best" means)
Target objective (in order):
1. Reliable probability quality for selected markets,
2. stable discrimination (AUC/PR-AUC) under rolling time splits,
3. strict deployability gates per market,
4. controlled drawdown at decision layer.

Not a goal:
- "Win every stake" (statistically impossible).

## 3) Market Scope (Your Linchpins)
Keep these families:
1. 1X2 + Double Chance
2. Goals totals (O1.5, U3.5)
3. Corners totals + home/away corners
4. Asian/European handicaps
5. 1UP/2UP (home/away)

Remove everything else from core training and reporting.

## 4) Ideal Target Architecture

## A. Data Layer (Point-in-time feature store)
Build one canonical point-in-time table per fixture snapshot (no leakage):
- entity keys: fixture_id, timestamp, league, teams
- football state features only (form, xG/xGA, shot profile, tempo, tactical, availability)
- optional market state features separated by namespace (not mixed by default)

Hard rules:
1. Every feature has an availability timestamp <= prediction timestamp.
2. Missingness must be explicit (indicator columns), not silently imputed away.
3. Feature lineage tracked (source table + extraction method + freshness).

## B. Modeling Layer (Family heads, not one generic model)
Use a shared base representation + family-specific heads.

1) Scoreline Head (Primary generative head)
- Predict full score distribution P(H, A) via bivariate Poisson / Dixon-Coles style / low-rank count model.
- This single head should generate:
  - 1X2
  - DC
  - AH/EH lines
  - O/U goal lines
  - 1UP/2UP approximations (with path model extension)

2) Goals Head (for O1.5/U3.5 refinement)
- Binary residual head for O1.5/U3.5 on top of scoreline features.
- Purpose: capture errors not modeled in scoreline base.

3) Corners Head (separate count process)
- Model total corners distribution directly (Negative Binomial preferred over Poisson due dispersion).
- Model team corners split (home/away) jointly or with coupled heads.
- Derive all corner lines from same distribution:
  - totals: 7.5, 8.5, 9.5, 10.5
  - team lines: 2.5, 3.5, ...
- Do not train separate disconnected binary models per corner line.

4) Anytime Lead Head (1UP/2UP)
- Use event-path / hazard approximation fed by score intensities and tempo features.
- Keep as separate head due path-dependence.

5) Handicap Head
- Derive from score-difference distribution from scoreline head.
- Optional correction head if systematic bias remains.

## C. Calibration Layer
Per family and per market:
1. Time-ordered calibration (holdout after train period).
2. Choose isotonic/sigmoid/temperature by holdout Brier + ECE.
3. Reject calibration method if it degrades discrimination materially.

## D. Decision Layer (separate from prediction)
Prediction system outputs probabilities and uncertainty.
Decision system applies:
- tradability eligibility,
- odds availability,
- edge thresholds,
- stake policy.

No decision logic should contaminate model training.

## 5) Feature Strategy (What to engineer)

## Core football features (required)
1. Team strength state:
- season baseline xG/xGA, rolling xG windows (3/5/10), attack-defense interaction
2. Shot quality profile:
- shots on target, box touches, big chances, conversion suppression
3. Tempo/state features:
- pace proxies, possession pressure, crossing profile
4. Tactical features:
- formation clusters, style delta, tactical stability index
5. Availability features:
- missing starter burden, xG-lost burden, defense-loss burden, uncertainty flags

## Market-state features (optional, modular)
- Odds-derived features can be used in a dedicated "price-informed" variant,
  but never as the only backbone.

Recommended deployment modes:
1. `core_model` (football features only) -> primary truth model
2. `core_plus_price` (football + odds residual) -> optional enhancement when coverage is strong

## 6) Evaluation Framework (non-negotiable)
1. Rolling walk-forward by time (no random split).
2. Purged windows around season boundaries / duplicates.
3. Metrics per market:
- AUC / PR-AUC
- Brier
- ECE
- Log loss
- calibration slope/intercept
4. Stability checks:
- std across folds
- league-wise variance
- provider availability sensitivity

Promotion gate per market:
1. AUC lift threshold
2. Brier non-worse
3. calibration under cap
4. minimum support sample

## 7) Why current corners setup fails
Current approach learns each line separately while line coverage is uneven.
This creates line-fragmented models with high missingness and unstable calibration.

Correct approach:
- one corners distribution model,
- derive all lines from same probability mass,
- train once, price many lines.

## 8) Re-Architecture Program (Execution Plan)

## Phase 0 (1 week): Scope + contracts
1. Freeze market scope to linchpins only.
2. Define feature contract v2 by family.
3. Define strict baseline artifacts per selected market.

## Phase 1 (2 weeks): Data + feature store v2
1. Build point-in-time training snapshot builder.
2. Add availability burden features and tactical stability features.
3. Add missingness indicators and freshness flags.

## Phase 2 (2 weeks): Family model rebuild
1. Build scoreline head + derived 1X2/DC/AH/OU outputs.
2. Build corners count head (totals + team split).
3. Build anytime lead head.

## Phase 3 (1 week): Calibration + gating
1. Per-family calibration search.
2. Promotion registry with transparent pass/fail reasons.
3. Tradable vs predict-only market routing.

## Phase 4 (1 week): Shadow + cutover
1. Run in shadow mode for 1-2 weeks.
2. Compare against incumbent by market and league.
3. Promote only passing markets.

## 9) Immediate Next 10 Tasks
1. Create `model_v2/market_scope.yaml` (linchpins only).
2. Create `model_v2/feature_contracts/{scoreline,corners,anytime}.yaml`.
3. Implement unified point-in-time dataset builder for v2.
4. Implement scoreline head training script.
5. Implement corners distribution head training script.
6. Implement market-derivation functions (1X2/DC/AH/OU from scoreline; lines from corners).
7. Implement per-family calibration pipeline.
8. Implement v2 evaluation runner (walk-forward + calibration metrics).
9. Implement promotion registry schema v2.
10. Run first offline benchmark and publish v2 model card.

## 10) Expected Outcome
If executed correctly:
1. better coherence across related markets,
2. fewer contradictory probabilities,
3. improved calibration stability,
4. clearer deploy/no-deploy decisions per market.

This is the right long-term architecture for your objective.
