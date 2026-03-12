# Bounded Corners Autoresearch Loop Design

Date: 2026-03-12

## Goal

Add a bounded autoresearch loop for the existing corners family that searches over a fixed library of PIT-safe interaction blocks and scores candidates using the repo's real corners evaluation path.

This is research tooling only. It must not change schedulers, production aliases, baseline registries, or promotion state.

## Recommended Approach

Build on top of `autoresearch_corners/` instead of replacing it.

The existing runner already knows how to:

- train one named corners challenger through `train_corners.py`
- run `live_replacement_compare.py`
- log a run summary

The new loop should add:

- a fixed interaction-block library
- candidate manifest generation
- temporary contract/config materialization
- bounded search execution across multiple candidates
- normalized result summaries and a simple leaderboard

## Scope

### Included

- a named interaction library for corners research
- a search runner that executes a bounded set of candidates
- deterministic candidate naming
- temporary experiment config / contract generation
- result ranking using live replacement as the primary objective
- focused tests around generation and dry-run command planning

### Excluded

- free-form code editing by the search loop
- scheduler or runtime wiring changes
- automatic promotion or baseline mutation
- unrestricted pairwise interaction search
- new non-PIT-safe feature seams

## Search Objective

Primary objective:

- improve matched `corners_overlap` live replacement behavior

Secondary objectives:

- preserve or improve holdout Brier and log-loss where available
- avoid cosmetic gains concentrated in a single weak market

Guardrails:

- reject candidates with broad regressions on `c85`, `c95`, `c105`
- reject complexity growth without meaningful overlap improvement
- reject unstable candidates that only improve `c75`

## Proposed Architecture

### `autoresearch_corners/interaction_library.py`

Defines the allowed PIT-safe interaction blocks. Each block is a named bundle of features to enable.

Initial families:

- `style_matchup_possession`
- `style_matchup_box_touches`
- `style_matchup_sot`
- `style_matchup_xg_against`
- `axis_attack_vs_defense`
- `axis_press_vs_territory`

### `autoresearch_corners/search_runner.py`

Runs bounded search over combinations of interaction blocks.

Responsibilities:

- enumerate candidate manifests from the library
- write temporary contracts/configs
- call the existing `run_experiment.py` plan or reuse its helpers
- collect result artifacts
- build a ranked leaderboard

### `autoresearch_corners/program.md`

Carries the search mission, objective, stop rules, and allowed changes.

### `autoresearch_corners/results/`

Stores:

- per-run logs
- candidate manifests
- generated contracts/configs
- leaderboard summaries

## Data Flow

For each candidate:

1. choose a small combination of named interaction blocks
2. materialize a temporary corners contract/config
3. train the candidate corners artifact
4. run live replacement compare
5. parse outputs into a normalized scorecard
6. mark the candidate as `keep`, `reject`, or `promising_but_unstable`

## Safety Model

- no production artifact alias changes
- no scheduler edits
- no automatic baseline rebuilds
- no promotion-side effects
- no feature creation outside the approved library

## Testing Plan

- unit test interaction-library expansion
- unit test manifest to contract/config generation
- unit test deterministic candidate tag generation
- runner dry-run test for multi-candidate command planning
- keep subprocess-heavy tests mocked where possible

## Implementation Notes

- prefer reusing `autoresearch_corners/run_experiment.py` helpers instead of duplicating path resolution and subprocess logic
- keep the first version manifest-driven
- design the search runner so beam-search or memory can be added later without breaking the v1 contract

## Success Condition

The loop is successful when it can run a bounded batch of PIT-safe corners challengers reproducibly, rank them by live replacement outcome, and leave behind interpretable summaries without touching production state.
