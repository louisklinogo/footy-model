# Corners PMF Surface v2 Design

## Goal

Build a new corners path version that fixes the current corners family's weakest seam:

- upper-tail totals: `c95`, `c105`
- away ladders: `ac25`, `ac35`, `ac55`
- threshold ranking instability across nearby corners markets

The current `totals_first` branch is useful but too mean-centric. It predicts a total mean, predicts or derives a home share, and then converts those means into ladder markets. That works for broad shape but leaves tail and team-market ranking underpowered.

The next model should represent the corners distribution more directly.

## Current Evidence

Best current corners challenger:

- `corners_totals_first_style_matchup_possession_box_touches_corners_against_featurepass_autoresearch_v1_candidate_20260312`

Observed behavior:

- beats legacy on some corners markets
- still fails overall promotion
- strongest remaining failures are tail and away-team ladders
- incremental feature sweeps and path tweaks did not close that seam

The repo already contains a `pmf_surface_blended` path, which is the right conceptual seam. The next step is to strengthen that path rather than continue broad interaction search.

## Recommended Approach

Implement `pmf_surface_blended_v2` as a new corners path version.

Core idea:

1. Keep the best current `totals_first` branch as the anchor prior.
2. Train count-distribution heads for `home_corners` and `away_corners`.
3. Build a coherent probability surface from those count PMFs.
4. Derive all corners ladder markets from that surface.
5. Blend the PMF-derived probabilities back toward the anchor prior using walk-forward-selected weights.

This keeps the current branch's stability while giving the model a richer representation of distribution shape.

## Why This Over Alternatives

### 1. PMF surface v2

Pros:

- attacks the exact remaining failure mode
- keeps all corners markets coherent
- models totals and team ladders from one shared surface
- matches the existing repo structure

Cons:

- more implementation work
- requires careful clipping, normalization, and blend governance

### 2. Direct ladder heads

Pros:

- can optimize ranking for each market directly
- simplest way to push AUC on specific ladders

Cons:

- incoherent market surfaces are likely
- patches symptoms, not the underlying distribution problem
- hard to keep monotone and internally consistent

### 3. Mixture or dispersion-only upgrade to totals-first

Pros:

- smaller change set
- easier to compare against the current branch

Cons:

- still indirect for team ladders
- likely improves the mean and some tails but leaves away-ladder ranking unresolved

Recommendation: choose option 1.

## Architecture

### Anchor prior

Use the current best corners branch as the prior backbone:

- total corners mean
- home share
- existing interaction-enriched feature set

This prior remains the fallback and blend target.

### PMF heads

Train a bounded count distribution for:

- `home_corners`
- `away_corners`

Initial support should remain bounded by count grid, but the v2 design should make the grid configurable rather than fixed by a single low constant.

Target outputs:

- per-row categorical probabilities for counts `0..K`
- clipped and renormalized row sums

### Surface derivation

From the home and away PMFs:

- derive home ladder probabilities from the home marginal
- derive away ladder probabilities from the away marginal
- derive total ladder probabilities from convolution of home and away PMFs

This yields a coherent `c*`, `hc*`, `ac*` surface.

### Blending

For each corners market:

- take PMF raw probability
- optionally calibrate
- blend toward the prior branch probability

Blend weights should be chosen on walk-forward validation, market by market, not globally.

## Data Flow

1. Load the existing corners PIT dataset and current best feature contract.
2. Fit the prior backbone exactly as today.
3. Fit PMF heads for home and away counts.
4. Generate PMF-derived market probabilities.
5. Compare PMF probabilities against prior probabilities on validation folds.
6. Fit calibrators and blend weights per market.
7. Persist all artifacts and metadata.
8. At prediction time:
   - run prior backbone
   - run PMF heads
   - derive coherent corners surface
   - apply calibrators and blend
   - emit final market probabilities

## Required Code Changes

### `src/modeling/v2/families/corners/train_corners.py`

- add a new explicit path version, likely `pmf_surface_blended_v2`
- increase flexibility of PMF count grid handling
- fit PMF heads with a richer configuration than the current lightweight path
- persist any new config needed by prediction

### `src/modeling/v2/families/corners/predict_corners.py`

- add loading and inference support for the new PMF path version
- ensure market derivation and blending remain coherent with training

### Supporting utilities

- reuse existing PMF helpers where possible
- only extract shared helpers if duplication becomes material

## Evaluation Plan

Primary targets:

- `c95`
- `c105`
- `ac25`
- `ac35`
- `ac55`

Guardrails:

- do not lose `c75`
- do not lose `c85`
- do not lose `hc25`
- do not lose `ac45`

Success criteria:

1. beat the current best corners challenger on the failed seam
2. improve matched live replacement on corners overlap
3. avoid degrading the already-working corners ladders
4. only run full formal evaluation if the focused challenger wins first

## Risks

- PMF heads may overfit rare tail counts
- wider count support can create unstable probabilities if not regularized
- direct PMF outputs may still need strong blending on sparse ladders
- if the PMF seam still fails, the issue is likely feature/data ceiling rather than path shape

## Out of Scope

- generic interaction sweeps
- new neural architecture research unrelated to the corners seam
- direct team heads as the primary fix
- production promotion or alias changes

## Implementation Order

1. inspect and baseline the current `pmf_surface_blended` path
2. define `pmf_surface_blended_v2` artifact contract
3. upgrade training path
4. upgrade prediction path
5. run focused challenger evaluation against the current best corners branch
6. materialize only if it clearly improves the failed markets

## Expected Outcome

Best-case:

- upper totals and away ladders improve enough to create the first corners challenger that is broadly competitive with legacy

Likely-case:

- corners becomes meaningfully stronger on the remaining seam, even if still short of full promotion

Failure-case:

- if `pmf_surface_blended_v2` does not improve the failed seam, then the next diagnosis should shift from architecture to data ceiling and feature availability rather than more path experimentation.
