# Unified Model Architecture Design (Override Mode)

Date: 2026-02-26  
Status: proposed (decision lock: `override`)  
Owner: In-house Senior DS loop  
Scope: Layer 1 + Layer 2 global + Layer 2 rule-layer + market outcome stack + ops orchestration

## 1) Executive Decision
We will run a single coordinated stack with `override` semantics for overlapping sparse shock signals.

Canonical backbone:
1. Layer 1 (`lambda_xgb`) produces base lambdas.
2. Layer 2 global (`situational_xgb`) applies policy-gated residual correction.
3. Layer 2 rule-layer applies deterministic shock adjustments.
4. For overlapping shock families (starting with `key_absent`), rule-layer uses `override` behavior instead of additive stacking.

Market layer:
5. `market_outcome_gbm` consumes canonical backbone state plus premium/odds features.
6. Market scoring/export stay under a single model identity contract.

## 2) Current-State Reality (As-Is)
1. Layer 2 is league-gated and currently enabled for `E1` and `RO1`.
2. Rule-layer is implemented and promoted (`keyabs_home_only_minus0.120`), but only active when `--enable-rule-layer` is passed.
3. Tick predict flow currently runs market prediction/export directly and does not consistently run Layer 1 + Layer 2 + rule-layer in the same cycle.
4. Market model quality is mixed across markets (usable, but not uniformly strong), and live calibration evidence is still sample-limited.

## 3) Target Interaction Contract (To-Be)

### 3.1 Backbone Computation Contract
For each fixture side (`home`, `away`):
1. Base lambda: `lambda_l1`.
2. Global residual output: `res_global`.
3. Policy shrinkage: `alpha(league)` from `layer2_deployment_policy.json`.
4. Global adjusted lambda candidate:
   - `lambda_global = max(0.01, lambda_l1 + alpha * res_global)`.
5. Rule-layer deterministic candidate:
   - `lambda_rule = rule(lambda_global, feature_state, config)`.

### 3.2 Override Semantics (Chosen)
If a rule fires in an overlap family (`key_absent` now):
1. Do not double-count equivalent global contribution for that side.
2. Use rule-led corrected value for that family under capped constraints.
3. Preserve non-overlapping global contributions.

Operationally: rule has priority on overlap families; global remains active for broad smooth effects (odds/schedule/standings).

### 3.3 Serving Output Contract
Persist for each fixture-side:
1. `lambda_l1`
2. `lambda_after_global`
3. `lambda_final` (after override rules)
4. `rule_fired` + `rule_family` + `rule_pct_capped`
5. `overlap_mode='override'`

All outputs are point-in-time auditable via metadata lineage fields.

## 4) Market Outcome Model Contract

### 4.1 Role
`market_outcome_gbm` remains the market mapper head (probabilities by market).

### 4.2 Required Feature Inputs
1. Existing premium and odds features.
2. Backbone features added as first-class inputs:
   - `lambda_home_l1`, `lambda_away_l1`
   - `adj_lambda_home_final`, `adj_lambda_away_final`
   - `rule_fired_home`, `rule_fired_away`
   - `rule_family_home`, `rule_family_away` (encoded)

### 4.3 Fallback Policy
If market model artifact is missing/invalid for a market:
1. fallback to Poisson/backbone-derived market probability,
2. mark prediction metadata with `fallback_used=true`,
3. keep scoring pipeline intact.

## 5) Ops Architecture (Single Path)

### 5.1 Canonical Tick/Daily Predict Sequence
1. Build/refresh premium snapshots.
2. Run Layer 1 lambda prediction for scheduled horizon.
3. Run Layer 2 global prediction (league policy gating).
4. Run Layer 2 rule-layer in `override` mode.
5. Build market features (including backbone outputs).
6. Predict market outcomes.
7. Export predictions.
8. Score settled predictions.
9. Run monitoring/audit loop.

No split-brain between daily and tick for prediction path.

### 5.2 Reliability Guardrails
1. Hard fail on post-kickoff odds in train/predict.
2. Hard fail on lineage/timing violations.
3. DB retry/backoff wrapper for tick critical phases.
4. Phase-level run logging into `pipeline_runs`.
5. Lockfile + stale-lock cleanup for schedulers.

## 6) Governance, Gating, and Rollout

### 6.1 League Governance
Layer 2 global remains policy-gated by league.  
Rules run under two scopes:
1. `enabled_league` scope when Layer 2 is enabled and confidence gates pass.
2. `disabled_league_safety` scope (conservative, odds-confirmed) when Layer 2 is disabled but safety mode is enabled in rule config.

### 6.2 Promotion Gates
Any promotion must pass:
1. leakage/timing checks = pass,
2. non-negative global lift vs baseline,
3. triggered-segment lift for targeted rule families,
4. no away-side degradation beyond tolerance,
5. calibration sample at decision-grade minimum.

### 6.3 Rollout Plan
1. Phase A: enforce unified orchestration path + override semantics.
2. Phase B: retrain market head with backbone features.
3. Phase C: champion/challenger run for 1-2 weeks.
4. Phase D: promote champion and lock serving contract.

## 7) Risks and Mitigations
1. Risk: hidden double-counting persists in edge cases.
   - Mitigation: explicit overlap family map + rule/global attribution logs.
2. Risk: market head overfits to backbone artifacts.
   - Mitigation: strict time split, per-league holdout, calibration checks.
3. Risk: operational divergence between tick and daily.
   - Mitigation: shared command graph and contract test in CI.
4. Risk: insufficient scored sample for robust calibration.
   - Mitigation: freeze broader rollout until sample gate is met.

## 8) Definition of Done (Architecture Track)
1. Tick and daily use the same backbone->market graph.
2. Override mode is implemented and test-covered for overlap families.
3. Market model retrained with backbone features and benchmarked.
4. Fallback path is active and observable.
5. Monitoring includes backbone/rule firing stats and market calibration.
6. Promotion memo updated with signed pass/fail against all gates.

## 9) Immediate Questions (None Blocking)
No blocking questions for execution start.  
Default implementation will treat `key_absent` as first overlap family and extend via config if future overlap families are added.
