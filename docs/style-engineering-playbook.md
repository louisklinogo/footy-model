# Style Engineering Playbook (Layer 2)

This is a practical guide for adding style-on-style features without data leakage or overfitting.

## 1) Start with style axes (before clustering)

Build 4-6 point-in-time style axes from rolling team stats:

- `territory_control = z(rolling_box_touches - rolling_box_touches_against)`
- `chance_volume = z(rolling_big_chances - rolling_big_chances_against)`
- `shot_threat = z(rolling_sot - rolling_sot_against)`
- `defensive_suppression = z(-rolling_xg_against)`
- `game_openness = z(rolling_xg + rolling_xg_against)`
- `width_directness = z(rolling_crosses / (rolling_box_touches + 1))`

Notes:
- Use rolling windows (for example, last 8-10 matches) and optionally time decay.
- Compute all inputs as-of kickoff only (no post-kickoff values).
- Normalize by league-season (z-score) using train-fold parameters only.

## 2) Engineer matchup identity features

For each axis, create directional matchup features:

- `style_delta_axis = home_axis - away_axis`

Then add a small set of interactions:

- `home_territory_control * away_defensive_suppression`
- `home_width_directness * away_shot_threat_allowed`

Keep this set small at first to limit overfitting.

## 3) Add tactical archetypes after deltas

Only after deltas are stable:

- Fit clustering (for example KMeans with k=4..6) on team-match style vectors.
- Generate: `home_style_cluster`, `away_style_cluster`, `style_pair`.
- Fit cluster model on training period only; transform test/inference with frozen centroids.

## 4) H2H ghosting with shrinkage (do not memorize noise)

Use pair-specific residual offsets only with safeguards:

- Minimum sample threshold (for example `n >= 5`).
- Time decay so older meetings matter less.
- Shrinkage to league prior:
  - `ghost = (n/(n+k))*pair_residual + (k/(n+k))*league_prior`

Do not use raw pair averages without shrinkage.

## 5) Leakage and integrity guards

- All style features must be computed from matches strictly before kickoff.
- For same-kickoff fixtures, use shared pre-kickoff state (no within-timestamp contamination).
- Add static/data audits for as-of joins and timestamp rules.

## 6) Validation protocol (mandatory)

Run temporal ablation in this order:

1. Baseline (Layer 1 only)
2. + style deltas
3. + archetypes
4. + H2H ghosting

Report per stage:

- Residual RMSE lift
- Out-of-time stability (rolling folds)
- Feature importance and prevalence

If a stage does not improve out-of-time metrics, remove it.

## 7) Common failure modes

- Too many interactions too early
- Cluster definitions refit on full history (leakage)
- Rare-feature explosion without prevalence checks
- Pair-memory overfitting in H2H features

Keep the pipeline simple, leakage-safe, and measured by temporal ablation.
