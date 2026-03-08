# Handicap Market Contract Design

Date: 2026-03-06  
Status: approved design captured in repo  

## Purpose

Define the canonical football handicap market contract before further scoreline diagnosis or replacement decisions.

## Decision

- **European handicap** is a true **3-way** market after handicap adjustment.
- **Asian handicap** is a true **2-way** market with push and half-outcome settlement behavior.
- Current repo handicap codes are **not all canonical sportsbook markets**.
- `model_v2/handicap_contract.yaml` is the canonical v2 handicap spec.

## Canonical math

Let `d = home_goals - away_goals`.

### European handicap

For displayed line `H:A`, define `threshold = A - H`.

- Home wins when `d > threshold`
- Draw wins when `d = threshold`
- Away wins when `d < threshold`

Examples:

- `0:1` → Home = win by 2+, Draw = win by exactly 1, Away = draw or away win
- `1:0` → Home = home win or draw, Draw = away by exactly 1, Away = away by 2+

### Asian handicap

For a selected side with handicap `h`, define:

- `selected_net = selected_goals + h - opponent_goals`
- full win when `selected_net > 0`
- push when `selected_net = 0`
- full loss when `selected_net < 0`

Quarter lines split stake across adjacent lines.

## Compatibility findings

- `ah_h05`, `ah_h15` align with Home `-0.5` and Home `-1.5`.
- `ah_a05`, `ah_a15` align with Away `-0.5` and Away `-1.5` in training/settlement, but current odds resolution appears sign-misaligned.
- `eh_h1` and `eh_a1` are **legacy binary proxies**, not full European handicap market captures.

## Files updated

- `model_v2/handicap_contract.yaml`
- `model_v2/market_scope.yaml`
- `model_v2/feature_contracts/scoreline.yaml`

## Required next alignment

1. scoreline derivation
2. target generation and evaluation
3. odds resolution
4. export naming and surface area