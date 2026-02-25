# Proposal: H1 vs H2 Situational Modeling

The Situational Model (Layer 2) is currently "Half-Blind"—it treats a 2.0 xG performance as a single block. By integrating our new `_p1` and `_p2` (SofaScore) data, we can unlock a higher-fidelity "Senior Grade" understanding of match dynamics.

## The Problem: "Averaging the Noise"
Currently, if a team creates 1.8 xG, the model sees a "Strong Attack". But there is a massive difference between:
- **Scenario A**: Creating 1.5 xG in the first 30 mins, then park the bus (Front-loaders/Fatigue).
- **Scenario B**: Creating 0.2 xG in H1, then chasing the game with 1.6 xG in H2 (Tactical Adjustments/Stamina).

## Proposed New Features (Senior Data Scientist Perspective)

### 1. The "Fatigue/Caffeine" Index
**Calculated as**: `(Rolling H2 xG) / (Rolling H1 xG)`
- **Goal**: Identify teams that consistently "die" in the second half vs. teams that have the fitness to dominate late.
- **Why**: This helps adjust the Poisson residual when a "Fast Starting" home team faces a "Strong Finishing" away team.

### 2. Tactical Flexibility Score (The "Half-Time Talk" Effect)
**Calculated as**: Rolling average of `(H2 xG - H1 xG)`
- **Goal**: Quantify which managers are actually effective at making tactical adjustments at the break.
- **Why**: Some squads consistently improve after 45 minutes of observation (Data-driven profiling of managers).

### 3. Game State xG Normalization (De-Noising)
**Calculated as**: Interaction between `H1_Score` and `H2_xG`.
- **Goal**: If a team is up 2-0 at H1, their H2 xG production often drops intentionally. Our model currently "punishes" their offensive rating for this, which is a mistake.
- **Action**: Use H1 results to weight the importance of H2 statistics in the rolling window.

### 4. Early-Season "Tempo" Detection
**Calculated as**: `H1_Box_Touches` vs `H2_Box_Touches`.
- **Goal**: Early in the season, goals are noisy. H1 Tempo is a much "sharper" indicator of a team's true intended style before fatigue sets in.

## Implementation Roadmap
1. [ ] **Update Snapshots**: Refactor `team_premium_snapshots` to include `rolling_h1_xg` and `rolling_h2_xg`.
2. [ ] **Feature Builder**: Add the "Fatigue Index" and "Half-Time Delta" to `situational_features_v2.py`.
3. [ ] **Training**: Retrain the model on the `_p1/_p2` residuals to see if the RMSE lift improves beyond the current 4%.

---
> [!TIP]
> From a betting perspective, the **Fatigue Index** is often the "hidden" variable that market odds miss, especially during congested winter fixtures.
