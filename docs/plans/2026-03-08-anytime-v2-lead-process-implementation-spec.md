# Anytime v2 Lead-Process Implementation Spec

Date: 2026-03-08  
Status: active redesign spec  
Depends on: `docs/plans/2026-03-08-anytime-v2-state-ladder-recommendation.md`

## Objective

Redesign the v2 anytime family so it models the actual market object: whether home or away reaches a 1-goal or 2-goal lead at any point in regulation.

The redesign must stay inside the existing v2 family structure:

- contract-driven features,
- PIT-safe train data,
- train / predict / derive separation,
- normal family artifact layout,
- standard evaluation / promotion / live-replacement tooling.

## Market semantics

The four anytime markets are absorbing-state lead events:

- `h_1up`: home leads by at least 1 at any point
- `a_1up`: away leads by at least 1 at any point
- `h_2up`: home leads by at least 2 at any point
- `a_2up`: away leads by at least 2 at any point

This is a **lead-path** problem, not just a final total-goals problem.

## Diagnosis of the current architecture

Current anytime v2 trains home/away goal-rate heads and derives the four markets via a constant or phase-split Markov head.

That architecture is useful but insufficient because it does not cleanly separate:

1. who gets the first lead,
2. who protects or extends that lead,
3. home/away asymmetry after state changes.

Repo evidence now says bounded feature tweaks improve the family but do not eliminate the live-overlap home-ladder weakness.

## Design principle

Preserve the v2 family framework and replace only the anytime family core.

New family mode:

- `path_version = state_ladder`

The family will learn latent state-aware scoring parameters, then derive the four displayed markets through a deterministic ladder-state pricer.

## Proposed family architecture

### Latent outputs

Predict these family-level latent quantities:

- `lambda_home_tied_p1`
- `lambda_away_tied_p1`
- `lambda_home_tied_p2`
- `lambda_away_tied_p2`
- `mult_home_lead`
- `mult_away_lead`
- `mult_home_trail`
- `mult_away_trail`

Interpretation:

- tied-state phase intensities capture the race to first lead
- lead multipliers capture extension pressure once ahead
- trail multipliers capture equalizer pressure once behind

### State space

Use a compact score-difference ladder:

- `-2`, `-1`, `0`, `+1`, `+2`

with `±1` and `±2` hit events treated as absorbing for the relevant market derivations.

### Derivation engine

Create `src/modeling/v2/families/anytime/state_pricer.py`.

Responsibilities:

- convert latent outputs into state transition rates by phase
- compute hit probabilities for `h_1up`, `a_1up`, `h_2up`, `a_2up`
- guarantee bounds and monotonicity

Preferred implementation:

- deterministic DP or matrix-exponential CTMC over the compact ladder state space

## Data and supervision

### Reuse existing incident truth

The repo already has regulation-safe lead-state truth in `fixture_incident_lead_states` and exposes:

- `home_led_by_1_any`, `away_led_by_1_any`
- `home_led_by_2_any`, `away_led_by_2_any`
- `first_home_lead_minute`, `first_away_lead_minute`

### New anytime labels helper

Create `src/modeling/v2/families/anytime/labels.py`.

Use it to derive supervision for:

- first-lead direction,
- extension after first lead,
- optional coarse early-vs-late lead timing buckets.

Phase 1 should avoid DB/schema changes if existing incident columns are enough.

## File-level workstreams

### Workstream 1 — Data and labels

Modify:

- `src/modeling/layer2_markets/market_outcome_calibrator.py`
- `src/modeling/v2/data/build_pit_dataset.py`

Create:

- `src/modeling/v2/families/anytime/labels.py`

Goal:

- ensure PIT/offline anytime training frame contains the lead-state labels needed by `state_ladder`.

### Workstream 2 — New pricer and derivation

Create:

- `src/modeling/v2/families/anytime/state_pricer.py`

Modify:

- `src/modeling/v2/families/anytime/derive_markets.py`

Goal:

- add `derive_and_validate_anytime_state_ladder(...)` while preserving current modes for comparison and rollback.

### Workstream 3 — Trainer redesign

Modify:

- `src/modeling/v2/families/anytime/train_anytime.py`

Goal:

- support `--path-version state_ladder`
- fit the latent heads
- write standard artifact outputs plus richer config/metadata

### Workstream 4 — Prediction redesign

Modify:

- `src/modeling/v2/families/anytime/predict_anytime.py`

Goal:

- load `state_ladder` artifacts
- derive the same four market rows
- keep downstream CSV/DB contracts stable

### Workstream 5 — Acceptance and evaluation

Add or tighten anytime-specific gating via:

- `model_v2/promotion_policies/anytime_core.yaml`
- optional small updates in evaluation tooling if needed

Anytime redesign success requires:

- positive or non-regressive matched live-overlap on `h_1up` and `h_2up`
- no material away-ladder giveback
- standard holdout / walkforward quality retained

## Validation plan

### Unit tests

- state pricer monotonicity, symmetry, and bounds
- label derivation correctness from synthetic incident rows
- `state_ladder` artifact save/load integrity

### Integration tests

- `train_anytime.py` emits standard holdout/walkforward artifacts in `state_ladder` mode
- `predict_anytime.py` preserves output row shape and metadata discipline
- evaluation flow still reads the family cleanly

### Model validation

1. smoke train on PIT subset
2. full anytime candidate train
3. evaluation flow
4. live replacement compare
5. inspect per-market live deltas with home ladders as hard gates

## Exit criteria

The redesign is considered successful only if it beats the current anytime leader on matched live-overlap in a way that is clearly visible on:

- `h_1up`
- `h_2up`

without materially regressing:

- `a_1up`
- `a_2up`

## Delivery order

1. labels/data plumbing
2. state pricer and derivation
3. trainer `state_ladder` mode
4. predictor `state_ladder` mode
5. tests
6. candidate training and evaluation
7. live-overlap decision

## Non-goal reminder

This is not a repo-wide architectural rewrite and not a free-form simulation project.

It is a structural upgrade of the anytime family core so the model class finally matches the market being priced.