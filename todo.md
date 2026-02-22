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

## 🧬 Model Improvements
- [ ] **Poisson-Odds Integration**
    - Integrate bookmaker odds as input features into the `src/modeling/train_lambda.py` pipeline to capture market sentiment and late team news.
- [ ] **Situational ML Overlay (Layer 2)**
    - Build a model to predict the "residual" errors of the Poisson model based on travel, rest cycles, and motivation factors.
- [ ] **Matchup Signature & Tactical Classification (Senior Grade)**
    - **Concept:** Moving beyond simple averages to model "Style-on-Style" interactions.
    - **Implementation:**
        - Cluster teams into **Tactical Archetypes** (e.g., *High-Pressing*, *Low-Block/Counter*, *Possession-Dominant*).
        - Feature engineering for **Matchup Identity**:
            - `style_delta`: How a "High-Press" team performs specifically against a "Weak-Build-up" team.
            - `transition_efficiency`: Success rate of converting turnovers into Big Chances.
        - **H2H Ghosting:** A historical delta offset for specific fixture pairs that consistently deviate from the Poisson baseline (the "Deja Vu" signatures).

## 🚀 Automation & Orchestration
- [ ] **AI Research Automated Workflow & JSON Schema**
    - Automate `research_queue.md` and implement Pydantic validation for the following KPIs:
        - **Personnel:** `absent_minutes_impact`, `star_player_void`, `defensive_pillar_out`.
        - **Schedule/Context:** `rest_delta_days`, `travel_distance_km`, `fixture_congestion_flag`.
        - **Environmental:** `weather_wind_speed_ms`, `pitch_surface_type`, `heavy_pitch_flag`.
        - **Market Sentiment:** `market_steam_status`, `clv_projection`.
        - **Referee Tendencies:** Yellow/Red card rates, penalty tendency, and home/away bias (critical for card/penalty markets).
