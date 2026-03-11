## Corners new-signal acquisition plan (2026-03-10)

### Objective
- Define the next **justified** corners work after the architecture sprint stop signal.
- Focus only on signals that can realistically beat the current corners ceiling under PIT-safe evaluation.

### Current anchor
- Keep `model_artifacts/v2/corners_totals_first_signal_expanded_league_style_v1_candidate_20260310/` as the research reference.
- Do **not** start another architecture-only corners branch until new signal/support is added.

### Phase A execution result (completed 2026-03-10)
- Full Phase A experimental contract:
  - contract: `model_v2/feature_contracts/experiments/corners_signal_phasea_v1.yaml`
  - artifact: `model_artifacts/v2/corners_signal_phasea_v1_candidate_20260310/`
  - live compare: `artifacts/v2/family_replacement/live_replacement_20260310_corners_signal_phasea_v1/`
  - result: **clear no-go** at corners overlap `ΔAUC -0.008589`, `ΔBrier +0.002084`, `Δlog-loss +0.004831`
- Bounded salvage variant without crosses:
  - contract: `model_v2/feature_contracts/experiments/corners_signal_phasea_core_v1.yaml`
  - artifact: `model_artifacts/v2/corners_signal_phasea_core_v1_candidate_20260310/`
  - live compare: `artifacts/v2/family_replacement/live_replacement_20260310_corners_signal_phasea_core_v1/`
  - result: **better than the full Phase A bundle, but still not good enough** at corners overlap `ΔAUC -0.005568`, `ΔBrier +0.001119`, `Δlog-loss +0.002465`
- Current leader remains better:
  - `signal_expanded_league_style_v1` at `ΔAUC -0.003596`, `ΔBrier +0.000492`, `Δlog-loss +0.001019`
- Practical implication: the contract-only signal pass is now exhausted as a promotion path. If corners work continues, move to **Phase B** rather than iterating more small contract reshuffles.

### Phase B execution result (completed 2026-03-10)
- Defensive-pressure extraction was implemented end-to-end:
  - training/PIT query updated in `src/modeling/layer2_markets/market_outcome_calibrator.py`
  - live prediction query updated in `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py`
  - experimental contract: `model_v2/feature_contracts/experiments/corners_signal_phaseb_pressure_v1.yaml`
  - candidate artifact: `model_artifacts/v2/corners_signal_phaseb_pressure_v1_candidate_20260310/`
  - live compare: `artifacts/v2/family_replacement/live_replacement_20260310_corners_signal_phaseb_pressure_v1/`
- PIT coverage on rebuilt Phase B snapshot (`8170` rows):
  - `rolling_errors_lead_to_shot*` missing about `32.1%`
  - `rolling_tackles_pct*` missing about `14.5%–14.6%`
  - added missingness indicators were populated with binary variance
- Result: **clear no-go** on matched live corners overlap:
  - `ΔAUC -0.017072`
  - `ΔBrier +0.004888`
  - `Δlog-loss +0.012008`
- This was materially worse than both:
  - current leader `signal_expanded_league_style_v1`: `-0.003596 / +0.000492 / +0.001019`
  - Phase A core salvage variant: `-0.005568 / +0.001119 / +0.002465`
- Practical implication: the strongest currently justified corners extraction seam has now been tested and failed. Corners should move to **parked / blocked** status unless upstream data/support materially improves.

### Repo-grounded evidence
- Current active corners contract: `model_v2/feature_contracts/corners.yaml`
- Current v2 PIT builder still comes from `src/modeling/v2/data/build_pit_dataset.py` -> `src/modeling/layer2_markets/market_outcome_calibrator.py`
- Richer snapshot metrics already exist in `src/features/build_team_premium_snapshots_v1.py`
- Style cluster artifacts already exist in `model_artifacts/style_clusters/`
- Availability join logic already exists in `src/modeling/availability_features.py`

### Highest-priority signal candidates

#### Tier 1 — Existing PIT columns not yet used by corners contract
- **Add `xgot/xa` block first**
  - Available now in PIT dataset.
  - Currently **not** in `model_v2/feature_contracts/corners.yaml`.
  - Direct chrono-safe probe vs current 41-feature base:
    - base mean Brier: `0.218371`
    - `+ xgot/xa`: `0.217807` (`-0.000563`)
- **Add `big_chances` next, but only with `xgot/xa`**
  - Alone it was weak/mixed.
  - Combined with `xgot/xa`, it gave the best compact block result:
    - `+ xgot/xa + big_chances`: `0.217674` (`-0.000697` vs base)
- **Re-enable / expand `crosses` only as secondary support**
  - Crosses already exist in PIT, but corners contract currently disables them.
  - Alone: small Brier gain (`-0.000280`) with slight AUC loss.
  - Combined with `xgot/xa`, still positive but weaker than `xgot/xa + big_chances`.
- Coverage in current PIT audit snapshot:
  - `xgot/xa` block complete on `67.809%` of rows
  - `big_chances` block complete on `84.871%`
  - `crosses` block complete on `91.432%`
- Post-benchmark read:
  - `xgot/xa + big_chances` is the least-bad compact addition and did improve over the full bundle once crosses were removed.
  - But even that cleaner variant still stayed materially behind the current leader on matched live overlap.
  - Treat Tier 1 as **informative but insufficient** under the current PIT signal regime.

#### Tier 2 — Existing upstream signals not yet surfaced into active PIT dataset
- **Add defensive pressure block from `team_premium_snapshots`**
  - Source already exists in `src/features/build_team_premium_snapshots_v1.py`:
    - `rolling_tackles_pct`
    - `rolling_errors_lead_to_shot`
  - These are **not** exposed by the active PIT extraction in `market_outcome_calibrator.fetch_dataset(...)`.
  - Rationale: they are better proxies for territorial pressure and forced defensive actions than another mean/split redesign.
- **Surface richer snapshot completeness metadata when adding these fields**
  - Add explicit missingness indicators if coverage is materially below current core blocks.
- Post-benchmark read:
  - This seam is now exhausted for immediate corners promotion.
  - The extraction worked, coverage was real, and the candidate still regressed materially on live overlap.
  - Do not continue corners iteration through more snapshot-field additions unless there is a fresh, materially different support unlock.

#### Tier 3 — Style priors with real upside, but incomplete pipeline wiring
- **Integrate fixture-level style clusters into v2 PIT only after coverage is fixed**
  - Builder exists: `src/features/build_team_style_clusters.py`
  - Artifact exists and passes internal gates:
    - silhouette `0.2397`
    - temporal ARI `0.5810`
  - But active PIT audit coverage is currently too low:
    - `home_style_cluster`: `27.589%`
    - `away_style_cluster`: `27.625%`
    - `style_matchup`: `31.028%`
  - Practical read: promising medium-term signal family, but not ready for direct corners adoption until coverage is raised and PIT joining is formalized.

#### Tier 4 — Operational unblockers, not immediate modeling wins
- **Availability / lineup block is currently non-functional for corners**
  - PIT snapshot columns exist, but the audit dataset shows all-zero counts and null freshness fields.
  - Current sample behavior:
    - `home_missing_players`, `away_missing_players`, role counts, `*_lineup_known`, `*_availability_known` all constant at `0`
    - freshness fields all null
  - Conclusion: do not spend modeling effort here until ingestion is actually populating before the chosen prediction cutoff.
- **Do not prioritize lineup-derived corners features until role data becomes real**
  - Useful corners variants would need wing-back / wide-player / set-piece role information.
  - The current availability block is too empty to support that safely.

#### Tier 5 — Market-support upgrades that would reopen richer corners modeling
- **Improve corners odds support before retrying market-surface architectures**
  - Current PIT snapshot:
    - any complete totals pair across `c75/c85/c95/c105`: `3.647%`
    - all four totals pairs complete: `0.00%`
    - team-corners odds columns: `0`
  - The repo does show latent market-code awareness in `src/betting/audit_model_architecture.py` for:
    - `home_corners_ou`, `away_corners_ou`
    - `corners_home_ou`, `corners_away_ou`
    - `team_corners_home_ou`, `team_corners_away_ou`
  - That makes market-support recovery a valid medium-term data objective, but not a current modeling dependency.

### Recommended implementation order

#### Phase A — Fastest high-ROI pass
1. Create an experimental corners contract that adds:
   - `xgot/xa`
   - `big_chances`
   - `crosses` (include both for/against fields)
2. Run corners-only chrono ablation and full benchmark.
3. Promote only if it improves both:
   - holdout mean Brier vs `signal_expanded_league_style_v1`
   - matched live-overlap corners segment
4. **Status:** completed and failed to clear the bar. Keep both Phase A contracts as research artifacts only.

#### Phase B — Expand PIT extraction from existing snapshot warehouse
1. Extend `market_outcome_calibrator.fetch_dataset(...)` to surface:
   - `rolling_tackles_pct`
   - `rolling_errors_lead_to_shot`
2. Add missingness indicators and coverage audit.
3. Re-run corners ablation before any new architecture work.
4. **Status:** completed and failed clearly. Keep Phase B code and artifact as evidence, but do not iterate further on this seam for corners.

#### Phase C — Style-cluster productionization
1. Rebuild / backfill style-cluster coverage to acceptable PIT levels.
2. Add PIT-safe join into the v2 dataset build path.
3. Test cluster labels as priors or one-hot matchup blocks.

#### Phase D — Operational recovery for late lineup signals
1. Diagnose why availability features are all fallback/zero in the PIT snapshot.
2. Only after nontrivial coverage exists, test lineup-role corners blocks.

#### Phase E — Market support upgrade
1. Verify whether team-corners odds are actually stored in `fixture_odds_markets` but not extracted, or absent upstream.
2. If support becomes real, revisit calibrated market-surface corners families.

### Stop/go rules
- **Go** only if a new signal block improves both holdout and matched live-overlap.
- **Stop** if the candidate block only helps on holdout but not on matched live-overlap.
- **Stop** if coverage is too thin or mostly fallback values.
- **Do not** reopen PMF / direct-surface / split-prior architectures until Tier 1–3 signal work is exhausted.
- **After Phase B:** treat corners as parked unless one of the following changes materially:
  - team-corners odds support becomes real
  - totals corners odds support becomes much denser
  - style-cluster PIT coverage becomes operationally strong
  - availability / lineup features become genuinely populated pre-match
  - a new upstream signal family arrives that is not just another small contract reshuffle

### Senior recommendation
- The next corners pass should be a **signal acquisition program**, not a model-architecture program.
- Updated immediate bet after execution: skip further Tier 1 contract-only reshuffles and move straight to **Phase B defensive-pressure extraction** (`rolling_tackles_pct`, `rolling_errors_lead_to_shot`) if corners work continues.
- Updated final recommendation after Phase B execution: **park corners research**. The current best research reference remains `corners_totals_first_signal_expanded_league_style_v1_candidate_20260310`, but it is still non-promotable and materially closer to live than either Phase A or Phase B follow-ups. Reopen corners only after a real upstream support change.