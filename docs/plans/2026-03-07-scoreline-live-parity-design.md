# Scoreline Live-Parity Contract Design

Date: 2026-03-07  
Owner: Augment Agent  
Status: approved

## Purpose

Define a two-track scoreline parity experiment that separates:
- scoreline model-shape quality,
- live-side information advantage,
- scoreline-only availability/lineup advantage.

## Decision

Add two experimental scoreline feature contracts and keep trainer/runtime code unchanged for the first pass.

## Why this exists

The current comparison is not apples-to-apples:
- live uses a wide direct-market stack with Layer 1 lambdas, Layer 2 adjusted lambdas, odds, and broad fixture context,
- current scoreline uses a shared latent goal head with Layer 1 lambdas plus scoreline-only availability/lineup and baseline features,
- current scoreline does not consume adjusted lambdas, rule-fired flags, or goal-market odds.

## Experiment variants

### 1. `scoreline_live_parity_plus`

Goal:
- give scoreline the major live-side goal-market signals,
- keep current scoreline-only extras.

Includes:
- current scoreline required features,
- `adj_lambda_home_final`, `adj_lambda_away_final`,
- `rule_fired_home`, `rule_fired_away`,
- goal odds, implied probabilities, and odds gaps,
- existing availability/lineup/missing-player/baseline features.

Question answered:
- how good can shared-head scoreline get when it is not starved of live-side information?

### 2. `scoreline_live_parity_strict`

Goal:
- approximate live’s signal family while preserving scoreline’s shared-head modeling shape.

Includes:
- core xG / SOT / sample-size features,
- `lambda_home_l1`, `lambda_away_l1`,
- `adj_lambda_home_final`, `adj_lambda_away_final`,
- `rule_fired_home`, `rule_fired_away`,
- goal odds, implied probabilities, and odds gaps,
- formation/style fields aligned with active live artifact use.

Excludes:
- availability-known / lineup-known indicators,
- freshness-hour features,
- missing-player counts,
- season-baseline / recent / regressed xG extras not present in the active live artifact.

Question answered:
- under near-equal information, is shared-head scoreline stronger or weaker than the live direct-market stack?

## Contract policy

- add new YAMLs under `model_v2/feature_contracts/experiments/`
- do not modify `scoreline.yaml`
- do not change trainer logic in the first pass
- treat `adj_lambda_*_final` as required parity features
- keep corners-only odds out of scoreline parity contracts

## Validation plan

1. Add contract-loading tests for both new YAMLs.
2. Add scoreline feature-selection tests for plus/strict inclusion rules.
3. Run targeted v2 pytest coverage for contracts and scoreline train utils.

## Training and benchmark sequence

Train three scoreline variants:
1. baseline current contract,
2. parity plus,
3. parity strict.

Then compare:
- live vs current scoreline,
- live vs parity plus,
- live vs parity strict.

## Interpretation guide

- if strict closes the gap, the old benchmark was mostly information asymmetry,
- if plus improves but strict does not, scoreline gains mostly from its extra availability/lineup block,
- if neither improves enough, live’s edge is more likely direct-market modeling plus odds anchoring,
- if strict wins, shared-head scoreline becomes a serious replacement candidate.