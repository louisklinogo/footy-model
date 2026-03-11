## Corners v2 signal-limit review (2026-03-10)

### Question
- After the full totals-first structural sprint, is corners still mostly architecture-limited, or is it now signal-limited under the current valid pre-match PIT setup?

### Executive conclusion
- The repo evidence now points to corners being **signal/support-limited more than architecture-limited**.
- The strongest current corners challenger remains `model_artifacts/v2/corners_totals_first_signal_expanded_league_style_v1_candidate_20260310/`, but it is still non-promotable and still slightly worse than live on matched overlap.
- The meaningful uplift in this sprint came from **feature-side expansion** (`signal_expanded*`), not from the later architectural redesigns.
- Every architecture-first follow-up after the feature-expanded leader regressed live: direct team heads, calibrated totals surface, league-share residual allocation, and coherent PMF surface.

### Best current challenger reference
- Artifact: `model_artifacts/v2/corners_totals_first_signal_expanded_league_style_v1_candidate_20260310/`
- Evaluation: `model_artifacts/v2/evaluation_corners_totals_first_signal_expanded_league_style_v1_candidate_20260310/`
- Live-overlap compare: `artifacts/v2/family_replacement/live_replacement_20260310_corners_totals_first_signal_expanded_league_style_v1/`
- Best matched live-overlap result in the sprint: `ΔAUC -0.003596`, `ΔBrier +0.000492`, `Δlog-loss +0.001019`

### Architecture sprint evidence
- `totals_first_v1`: `ΔBrier +0.000877`
- `totals_first_residual_v1`: `ΔBrier +0.004817`
- `totals_first_market_heads_v1`: `ΔBrier +0.003118`
- `signal_expanded_v1`: `ΔBrier +0.000886`
- `signal_expanded_noodds_v1`: `ΔBrier +0.001012`
- `signal_expanded_league_style_v1`: `ΔBrier +0.000492` (**best**)
- `team_market_calibrated_league_style_v1`: `ΔBrier +0.001204`
- `totals_surface_calibrated_league_style_v1`: `ΔBrier +0.000813`
- `totals_first_league_share_residual_v1`: `ΔBrier +0.000684`
- `pmf_surface_blended_v1`: `ΔBrier +0.003619`
- Result: all major challengers stayed negative on matched live overlap; none passed more than `1/12` corners markets in evaluation.

### What actually improved the most
- The biggest step in the sprint was from the plain totals-first baseline to the feature-expanded league/style variant:
  - `totals_first_v1` holdout mean Brier `0.215582` -> `signal_expanded_league_style_v1` holdout mean Brier `0.214142`
  - matched live-overlap `ΔBrier +0.000877` -> `+0.000492`
- By contrast, later architectural variants built on the stronger backbone all moved backward on live:
  - league-share residual: `+0.000684`
  - calibrated totals surface: `+0.000813`
  - direct team-market calibrated: `+0.001204`
  - PMF surface: `+0.003619`

### Signal-support evidence
- Totals odds support in the current PIT audit dataset is extremely sparse:
  - `c75`: `0.00%` with both sides present
  - `c85`: `0.477%`
  - `c95`: `2.497%`
  - `c105`: `0.808%`
  - any complete totals line pair across `c75/c85/c95/c105`: `3.647%`
  - all four totals line pairs complete on the same row: `0.00%`
- Team-corners odds support is absent in the active PIT dataset snapshot:
  - team-corners odds columns present: `0`
- Practical implication: there is very little trustworthy market support available to calibrate or validate richer corners surfaces, especially for team-corners.

### Contract-safe ceiling sanity check
- A quick direct per-market `HistGradientBoostingClassifier` probe on the **valid active corners contract** (`41` usable features after regime application) reached only:
  - mean AUC `0.586761`
  - mean Brier `0.218371`
  - mean log-loss `0.627945`
- That probe is weaker than the current best structured corners artifact, which is what we would expect if the current leader is already extracting most of the available signal from the valid feature set.
- A broader "use all numeric non-odds columns" probe immediately produced fake perfect scores because the PIT dataset still contains target-derived columns outside the contract boundary, including:
  - `home_corners`
  - `away_corners`
  - `total_corners`
  - `corners_net_diff`
  - `corners_gap_75`, `corners_gap_85`, `corners_gap_95`, `corners_gap_105`
- Practical implication: the trustworthy ceiling has to be judged inside the valid contract, not by unconstrained sweeps over the raw PIT table.

### Bottom-line decision
- Stop architecture-only corners branching for now.
- Keep `corners_totals_first_signal_expanded_league_style_v1_candidate_20260310` as the research reference, but do not promote it.
- Keep the current bundled corners reference unchanged until a genuinely new signal source exists.

### What would justify revisiting corners
- New pre-match corners-specific signal, for example:
  - materially better corners/team-corners odds support
  - stable pre-match attacking-width / crossing / territorial-pressure features
  - lineup-derived set-piece and wing-back availability features with demonstrated PIT safety
  - richer league/team tactical style priors that transfer on matched live overlap
- Without new signal/support, more internal reshuffling of the current corners architecture is unlikely to beat the present leader.