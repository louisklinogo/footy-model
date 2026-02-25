# Project Todo List - Soccer Predictive Model

## 🛠️ Model Diagnostics & Calibration
- [ ] **Poisson Model Calibration Check**
    - **Goal:** Verify if predicted probabilities (1X2, U/O) align with actual hit rates.
    - **Deliverable:** `src/modeling/calibration_check.py`
    - **Logic:** 
        1. Extract history from `data/v1/daily_slip_YYYYMMDD.csv`.
        2. Cross-reference with `fixture_results` from DB/JSON.
        3. Bin probabilities and calculate **Expected Calibration Error (ECE)**.
        4. Generate a reliability diagram (calibration curve).

## 📊 Data Ingestion & Engineering
- [x] **Key Player Absence Flag (`is_key_absent`)**
    - **Goal:** Add a boolean flag catching when a core starter is out.
    - **Logic:** Calculate via SQL if a missing player started >70% of available matches in the last 10 games.
- [x] **European Cups (CL/EL/ECL) Pipeline**
    - **Goal:** Ensure European matches are seamlessly integrated.
    - **Logic:** Confirm fixture seeders pull these leagues and map `team_id` perfectly so domestic and European rolling stats flow into a single unified timeline.
- [ ] **Historical Odds Convergence (`odds_model_gap`)**
    - **Goal:** Let the Layer 2 Situational Model know what the betting market thinks.
    - **Logic:** Backfill historical closing 1X2, AH, and O/U odds into `fixture_odds_snapshots`.
    - **Implementation Details:**
        - Source: Sofascore `/event/{match_id}/odds/1/all` (Provider 1: bet365).
        - Format: Convert fractional odds (`fractionalValue` and `initialFractionalValue`) to Implied Probabilities.
        - Calculate divergence between Poisson projection and Market implied probability.

## 🧬 Model Improvements (Future)
- [ ] **H1 vs H2 Dynamics (Senior Grade)**
    - **Goal:** Leverage granular period stats to detect fatigue and tactical switches.
    - **Features:** Fatigue Index (H2/H1 xG ratio), Manager "Half-Time Talk" effect, Game-State Normalization.
    - **Ref:** `docs/halftime_logic_proposal.md`
- [ ] **Matchup Signature & Tactical Classification (Senior Grade)**
    - **Concept:** Moving beyond simple averages to model "Style-on-Style" interactions based on advanced stats.
    - **Resource:** `docs/style-engineering-playbook.md`
    - **Implementation:**
        - Cluster teams into **Tactical Archetypes** (e.g., *High-Pressing*, *Low-Block/Counter*, *Possession-Dominant*, etc).
        - Feature engineering for **Matchup Identity**:
            - `style_delta`: How a "High-Press" team performs specifically against a "Weak-Build-up" team.
        - **H2H Ghosting:** A historical delta offset for specific fixture pairs that consistently deviate from the Poisson baseline (the "Deja Vu" signatures).



## 📈 Layer 2 Situational: Next DS Steps

### Feature Health Checks
- [ ] Run feature coverage report on training set — identify columns that are constant (0 variance) or >90% missing
- [ ] Check prevalence of rare flags: `is_derby`, `home_lame_duck`, `away_lame_duck` should each be <5% of rows (XGBoost won't split on them until sample size grows)
- [ ] Verify `odds_model_gap` column is NOT constant before retraining

### Odds Model Gap (High ROI)
- [ ] **Backfill historical odds** into `fixture_odds_snapshots` (or a new table) with `snapshot_time_utc`
- [ ] Convert odds to implied probabilities (1X2 at minimum; handle vig removal if possible)
- [ ] **Leakage guard**: join uses latest snapshot where `snapshot_time_utc <= match_datetime_utc`
- [ ] Compute `odds_model_gap = market_implied_home_prob - poisson_home_prob` (add draw/away variants too)
- [ ] **Retrain** and check feature importance — odds gap is likely the biggest incremental lift

### Player Availability Coverage
- [ ] Verify `player_availability` coverage for historical seasons — confirm backfill has run for 2024/2025
- [ ] If coverage is sparse, flag as "expected low importance" and track % populated per league/season
- [ ] Add a DB check (extend `audit_layer2_situational_leakage.py`) to flag rows where `recorded_at > match_datetime_utc`

### Evaluation & Ablation
- [ ] **Temporal CV**: split training by date (e.g., train on first 80% of matches chronologically, test on last 20%) — report RMSE and hit rates per segment
- [ ] **Ablation table**:
  1. Baseline (Layer 1 Poisson only)
  2. + Standings/points gap
  3. + Schedule (congestion, upcoming tier)
  4. + Player impact
  5. + Odds model gap
- [ ] Report RMSE lift and feature importance at each stage

### Monitoring (Production)
- [ ] Track `odds_model_gap` distribution drift over time — retrain when mean shifts >0.05
- [ ] Add alert if `player_availability` coverage drops below 50% for upcoming matches
- [ ] Log same-kickoff fixture groups and verify they share identical pre-kickoff state

---

### Why some features show 0.0000 importance (Feb 2026 training run)

From `.sisyphus/evidence/task-6-train.txt` (Home model):
```
home_upcoming_tier: 0.0000
home_lame_duck: 0.0000
away_lame_duck: 0.0000
is_derby: 0.0000
home_xg_lost: 0.0000
away_xg_lost: 0.0000
home_key_absent: 0.0000
away_key_absent: 0.0000
injury_impact: 0.0000
home_playing_top4: 0.0000
away_playing_top4: 0.0000
derby_position_gap: 0.0000
odds_model_gap: 0.0000
```

**What this means:**
- `odds_model_gap` is hardcoded to `0.0` in training — not empty, just a constant. It will light up once we backfill market odds.
- Player impact features (`xg_lost`, `key_absent`, `injury_impact`) exist but appear mostly zero/unpopulated in the current historical window — likely backfill still in progress.
- Derby / lame duck flags are rare (<2% of matches). XGBoost needs far more samples to learn these edge cases.

**Verdict:** The core foundation works and finds real alpha (~4.4% RMSE lift on home residuals). Once odds backfill finishes and we retrain, expect the bottom half of this list to light up.

### Hephaestus DS Guardrails (Added)
- [ ] **Make split chronology explicit**: enforce `max(train_match_datetime_utc) < min(test_match_datetime_utc)` and fail fast if violated
- [ ] **Add rolling temporal CV** (minimum 3 folds) and report mean/std RMSE lift vs baseline
- [ ] **Odds leakage audit**: assert every odds row used for training/inference satisfies `snapshot_time_utc <= match_datetime_utc`
- [ ] **Cold-start / promoted teams check**: evaluate fixtures where either team has `<6 prior games` and report separate metrics
- [ ] **Artifact reproducibility sidecar**: write JSON next to model artifact with train date range, row counts, feature list/hash, hyperparams, and git commit hash (if available)
