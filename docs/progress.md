# Progress Tracker - 3-Layer Syndicate Architecture

Last updated: 2026-02-22

---

## Completed

### Infrastructure
- [x] `team_league_standings` table created and populated (512 teams)
- [x] `fixtures.season` column added (all 9,800 fixtures = "2025-26")
- [x] Postgres trigger `trg_update_standings` on `fixture_results` for incremental updates

### Layer 1: Poisson/Dixon-Coles Baseline
- [x] `src/modeling/train_lambda.py` - Trains XGBoost Poisson models for lambda_home, lambda_away
- [x] `src/modeling/predict_lambda.py` - Generates lambda predictions and saves to DB
- [x] Model artifacts: `models/artifacts/xgb_lambda_home_v1.json`, `xgb_lambda_away_v1.json`
- [x] Dixon-Coles rho calibration in `calibration_params.json`
- [ ] **Missing: Time decay** - Research done, implementation pending

### Layer 2: Situational ML Residual
- [x] `src/modeling/train_situational_residual.py` - Feature builder + trainer
- [x] `models/v2_situational/situational_features.csv` - 6,282 fixtures with rest/congestion/motivation
- [x] `models/v2_situational/residual_model.pkl` - Ridge regression (R2=0.02, 456 samples)
- [ ] **Missing: Form data features** - rolling features available in team_premium_snapshots
- [ ] **Missing: Blowup detection** - partial patterns exist

### Layer 3: AI Research Overlay
- [x] `src/modeling/ai_research_pass.py` - Tavily-based research (original)
- [x] `src/modeling/ai_research_pass_exa.py` - Exa-based research (NOT TESTED)
- [x] `src/modeling/apply_research_overlay.py` - Applies verdicts to predictions (NOT TESTED)

### Diagnostics
- [x] `src/modeling/calibration_check.py` - ECE + reliability diagrams (NOT TESTED)

---

## Research Findings (2026-02-22)

### 1. Time Decay for Poisson

**Dixon-Coles formula:** `w = exp(-xi * days_ago)`

**Recommended xi values:**
- xi=0.001: Slow decay (90-day half-life)
- xi=0.002: Moderate decay (45-day half-life) 
- xi=0.003: Fast decay (30-day half-life)

**Implementation:**
```python
def compute_time_decay_weights(match_dates, as_of=None, xi=0.002):
    if as_of is None:
        as_of = pd.to_datetime(match_dates).max()
    dates = pd.to_datetime(match_dates)
    days_ago = (as_of - dates).dt.days.astype(float)
    return np.exp(-xi * days_ago)
```

**Where to add:** `train_lambda.py` - apply weights to XGBoost sample_weight parameter

---

### 2. Form Data Features (Available in team_premium_snapshots)

| Feature | Description |
|---------|-------------|
| rolling_xg | Expected goals for |
| rolling_xg_against | Expected goals against |
| rolling_xgot | xG on target |
| rolling_xa | Expected assists |
| rolling_box_touches | Box touches |
| rolling_big_chances | Big chances created |
| rolling_sot | Shots on target |
| rolling_corners | Corners |
| rolling_rest_days | Rest between matches |

**Gap:** No combined "form_rating" score

**Implementation:** Extend `train_situational_residual.py` to pull these from `build_features.py`

---

### 3. Blowup Detection

**Existing patterns:**
- `ai_research_pass.py`: `PersonnelKPIs` (absent_minutes_impact, star_player_void, defensive_pillar_out)
- `live_edge_detector.py`: EV threshold detection
- `train_situational_residual.py`: Residual computation

**Recommended new file:** `blowup_detector.py`
1. Flag fixtures with large residuals (z-score > 2)
2. Check for red cards via AI research
3. Cluster unusual scorelines (high-scoring, unexpected results)

---

## Data Counts

| Resource | Count |
|----------|-------|
| Total fixtures | 9,800 |
| FT fixtures | 6,282 |
| Results with goals | 5,877 |
| Team snapshots | 18,242 |
| Lambda predictions | 956 (478 fixtures) |
| Premium GBM predictions | 220 |
| Team standings | 512 |

---

## Pending Tasks

### High Priority
1. Generate all lambda predictions - `python src/modeling/predict_lambda.py`
2. Test Exa research code - `python src/modeling/ai_research_pass_exa.py`
3. Test AI overlay - `python src/modeling/apply_research_overlay.py --dry-run`

### Medium Priority
4. Add time decay to `train_lambda.py`
5. Add form features to Layer 2
6. Create blowup detector

### Low Priority
7. Test calibration check
8. Integrate 3 layers into unified pipeline
9. Build disagreement detector (Poisson vs GBM)

---

## Architecture

```
Layer 1: Poisson/Dixon-Coles
  train_lambda.py -> lambda_home, lambda_away
  poisson.py -> score matrix -> all market prices

Layer 2: Situational ML
  train_situational_residual.py
  Features: rest_delta, congestion_flag, motivation_score
  Output: residual adjustments to lambda

Layer 3: AI Research
  ai_research_pass_exa.py -> research verdict
  apply_research_overlay.py -> final adjustments
```
