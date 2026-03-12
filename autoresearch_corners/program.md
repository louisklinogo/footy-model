# corners autoresearch program

## Mission

Use the existing corners family to test a small number of disciplined challengers.

## Optimize for

1. better matched live-replacement performance
2. stable totals corners behavior
3. minimal added complexity

## Allowed changes

- `experiment.local.json`
- `search.local.json`
- corners feature-contract choice
- named interaction-block choice from the bounded library
- corners `path_version`
- candidate naming and run notes

## Not allowed

- scheduler changes
- production cutover
- promotion registry edits
- broad repo refactors disguised as research

## Stop rules

- stop if the candidate fails to train cleanly
- stop if live replacement cannot run
- stop if gains are only cosmetic and do not improve the overlap cohort enough to matter

## Bias for corners

Treat corners as a support-limited family. Prefer seam tests over architecture novelty.

## Search discipline

- prefer small named interaction bundles over anonymous pairwise sweeps
- optimize for `corners_overlap` live replacement first
- use holdout summaries only as secondary tie-breakers
- stop quickly when gains come only from `c75` or when `c85/c95/c105` regress together
