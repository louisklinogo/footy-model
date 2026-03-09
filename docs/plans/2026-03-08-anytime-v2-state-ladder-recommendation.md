# V2 Anytime Family Redesign Recommendation

## 1. Overview
- **Recommendation:** replace the current `markov_head_v1` anytime family (`src/modeling/v2/families/anytime/train_anytime.py` + `src/pricing/markov.py`) with a **state-aware lead-process family** exposed through a new anytime path mode: `state_ladder_v1`.
- **Market semantics:** these are not goalscorer markets. `h_1up / a_1up / h_2up / a_2up` are absorbing-state lead events: probability a side leads by 1 or by 2 at any point in regulation.
- **Why this class:** the current family maps final home/away scoring rates into anytime via a homogeneous CTMC. That is too weak because it cannot represent “who gets the first lead” separately from “who extends a 1-goal lead into 2”, nor can it express home/away continuation asymmetry cleanly.
- **Observed failure mode in this repo:** the latest live replacement artifact improved away ladders but still regressed home ladders (`artifacts/v2/family_replacement/live_replacement_20260308_anytime_h1up_rich_snapshot_v1/per_market_comparison.csv` shows `h_1up` and `h_2up` losing live AUC/Brier/log-loss/ECE vs the champion, while `a_1up` and `a_2up` improved).
- **Goals / success criteria:**
  - fix the live-overlap home ladder weakness without breaking family monotonicity (`h_2up <= h_1up`, `a_2up <= a_1up`),
  - stay inside the existing `train / predict / derive / evaluate` family architecture,
  - keep artifact outputs compatible with `src/modeling/v2/run_evaluation_flow.py`, `src/modeling/v2/run_prediction_flow.py`, and `src/modeling/v2/eval/live_replacement_compare.py`.
- **Included:** anytime family only, four anytime markets, PIT/offline training, v2 evaluation/promotions.
- **Excluded:** scoreline/corners redesign, active scheduler changes, dependency additions.

## 2. Prerequisites
- **Data source:** the PIT builder already routes through `src/modeling/v2/data/build_pit_dataset.py`, which calls `legacy_calibrator.fetch_dataset()` and `add_targets_and_derived()` in `src/modeling/layer2_markets/market_outcome_calibrator.py`.
- **Timeline labels available:** `fixture_incident_lead_states` already stores `first_home_lead_minute`, `first_away_lead_minute`, `home_led_by_1_any`, `home_led_by_2_any`, `away_led_by_1_any`, `away_led_by_2_any` via `src/modeling/evaluation/build_incident_lead_state_features.py`.
- **No package install is required.**
- **Prefer no DB migration for v1 of this redesign:** first derive new labels from existing incident columns during dataset assembly; only add schema if a truly missing label is required.

## 3. Implementation Steps

### Step 1: Freeze the target contract and promotion standard
- **Files:** `model_v2/feature_contracts/anytime.yaml`, create `model_v2/promotion_policies/anytime_core.yaml`, optionally `docs/v2_upgrade_tracker.md`.
- **What to do:** define the anytime core as exactly `h_1up`, `a_1up`, `h_2up`, `a_2up`.
- **Key detail:** do **not** promote on aggregate anytime-family averages alone; require explicit passes on the two home ladder markets because that is the known weak spot.
- **Testing:** add/extend policy tests similar to `tests/v2/test_promotion_policy.py`.

### Step 2: Add incident-derived supervision for the ladder path
- **Files:** `src/modeling/layer2_markets/market_outcome_calibrator.py`, `src/modeling/v2/data/build_pit_dataset.py`, create `src/modeling/v2/families/anytime/labels.py`.
- **What to do:** enrich the anytime training frame with ladder-specific labels derived from `fixture_incident_lead_states`.
- **Recommended labels:**
  - `home_first_to_1up`, `away_first_to_1up`,
  - `home_1up_then_2up`, `away_1up_then_2up`,
  - `first_home_lead_minute`, `first_away_lead_minute`,
  - retain `home_goals_p1`, `away_goals_p1`, then derive `home_goals_p2`, `away_goals_p2` as today.
- **Key detail:** keep all labels regulation-only and timeline-safe, consistent with `build_incident_lead_state_features.py`.
- **Testing:** add unit tests for label derivation and PIT leakage safety.

### Step 3: Replace the current anytime derivation core with a ladder-state pricer
- **Files:** create `src/modeling/v2/families/anytime/state_pricer.py`, modify `src/modeling/v2/families/anytime/derive_markets.py`, optionally leave `src/pricing/markov.py` untouched for backward compatibility.
- **Best structural class:** a **2-phase score-difference CTMC / dynamic-programming lead-process model** over score-diff states `{-2, -1, 0, +1, +2}`.
- **Recommended latent outputs:**
  - `lambda_home_tied_p1`, `lambda_away_tied_p1`,
  - `lambda_home_tied_p2`, `lambda_away_tied_p2`,
  - `mult_home_lead`, `mult_away_lead`,
  - `mult_home_trail`, `mult_away_trail`.
- **Derivation strategy:**
  - tied state uses the phase base rates,
  - from `+1`, use `home -> +2` via `lambda_home * mult_home_lead` and `away -> 0` via `lambda_away * mult_away_trail`,
  - from `-1`, mirror the same logic,
  - compute market hit probabilities for `±1` and `±2` thresholds by matrix exponential or exact DP over the 2 phases.
- **Why this is the right compromise:** it stays structural and monotone like the current family model, but it introduces the missing “reach lead” vs “extend lead” separation that bounded feature tweaks cannot supply. It is stronger than a plain phase split, but still fits the repo’s family-artifact pattern without becoming a free-form simulator.
- **Testing:** add pricer tests for monotonicity, symmetry, probability bounds, and regression against the current `constant` / `phase_split` modes.

### Step 4: Train a multi-head anytime artifact instead of just two or four goal heads
- **Files:** modify `src/modeling/v2/families/anytime/train_anytime.py`, create `tests/v2/test_anytime_state_train.py`.
- **What to do:** add a new `--path-version state_ladder` mode while retaining `constant` and `phase_split` as baselines.
- **Training recipe:**
  - fit phase rate heads from `home_goals_p1/away_goals_p1/home_goals_p2/away_goals_p2`,
  - fit lead/trail multiplier heads from the new incident-derived labels,
  - save all heads plus feature/imputation artifacts into the same family artifact directory contract used today.
- **Key detail:** avoid four direct market classifiers as the primary model; they fit the symptoms but break the family’s latent-to-derived design and fragment calibration.
- **Testing:** verify artifact I/O, model config persistence, and walk-forward summary generation remain compatible.

### Step 5: Keep prediction and DB write paths artifact-compatible
- **Files:** modify `src/modeling/v2/families/anytime/predict_anytime.py`.
- **What to do:** load the new latent heads, derive market probabilities through `state_pricer.py`, and persist the same four market rows.
- **Metadata to write:** include all latent outputs and `derivation: state_ladder_ctmc_v1` in `metadata_json`.
- **Key detail:** keep the outer prediction row shape unchanged so `run_prediction_flow.py` and downstream DB consumers do not need to change.
- **Testing:** extend prediction-flow tests and smoke-test CSV/DB row contents.

### Step 6: Make live-overlap evidence part of the anytime acceptance bar
- **Files:** optionally modify `src/modeling/v2/run_evaluation_flow.py`, `src/modeling/v2/eval/live_replacement_compare.py`, `tests/v2/test_evaluation_flow_runner.py`, `tests/v2/test_live_replacement_compare.py`.
- **Evaluation standard:**
  - existing holdout + walk-forward metrics remain required,
  - add a required live replacement compare artifact for anytime candidates,
  - candidate must be non-regressive on **both** `h_1up` and `h_2up` in live overlap Brier and log-loss,
  - family aggregate may improve, but home-ladder regressions block promotion,
  - away ladders cannot materially worsen while fixing home ladders.
- **Key detail:** treat `promotion_registry.py` as the offline gate and live replacement as the operational gate.
- **Testing:** verify the runner can emit or at least document the live-compare artifact path for anytime promotion decisions.

### Step 7: Only add a residual overlay if the structural gap becomes small and stable
- **Files:** optionally create `src/modeling/v2/families/anytime/train_anytime_residuals.py`, `src/modeling/v2/families/anytime/residual_utils.py`.
- **Recommendation:** do this only after `state_ladder_v1` lands. Do not lead with a residual-only fix.
- **Reason:** the current issue is structural; a residual overlay should be reserved for small, repeatable calibration bias after the path model is corrected.
- **Testing:** mirror the scoreline residual pattern only if needed.

## 4. File Changes Summary
- **Create:**
  - `docs/plans/2026-03-08-anytime-v2-state-ladder-recommendation.md`
  - `src/modeling/v2/families/anytime/labels.py`
  - `src/modeling/v2/families/anytime/state_pricer.py`
  - `model_v2/promotion_policies/anytime_core.yaml`
  - `tests/v2/test_anytime_state_pricer.py`
  - `tests/v2/test_anytime_state_train.py`
- **Modify:**
  - `src/modeling/layer2_markets/market_outcome_calibrator.py`
  - `src/modeling/v2/data/build_pit_dataset.py`
  - `src/modeling/v2/families/anytime/train_anytime.py`
  - `src/modeling/v2/families/anytime/predict_anytime.py`
  - `src/modeling/v2/families/anytime/derive_markets.py`
  - `model_v2/feature_contracts/anytime.yaml`
  - optionally `src/modeling/v2/run_evaluation_flow.py` and `src/modeling/v2/eval/live_replacement_compare.py`
- **Delete:** none initially.

## 5. Testing Strategy
- **Unit tests:**
  - ladder-state derivation monotonicity and bounds,
  - label builder correctness from synthetic incident sequences,
  - artifact save/load for `state_ladder` mode.
- **Integration tests:**
  - `train_anytime.py` produces the canonical holdout/walk-forward files,
  - `predict_anytime.py` writes the same four markets with richer metadata,
  - evaluation-flow wiring still includes anytime cleanly.
- **Manual / model validation:**
  - run focused pytest in `tests/v2/test_anytime_* tests/v2/test_evaluation_flow_runner.py tests/v2/test_live_replacement_compare.py`,
  - train a candidate artifact on the latest PIT snapshot,
  - run `src/modeling/v2/eval/live_replacement_compare.py` against the same matched cohort used by the current anytime benchmark,
  - inspect `per_market_comparison.csv` and confirm home ladder no longer regresses.

## 6. Rollback Plan
- Keep `constant` and `phase_split` paths intact while developing `state_ladder`.
- Promotion remains blocked until the new artifact clears both offline and live gates.
- If the redesign underperforms, revert the anytime artifact directory/model version to the current champion and remove the new promotion policy from use.
- Because the first version avoids schema migration, rollback is code/artifact-only.

## 7. Estimated Effort
- **Effort:** ~3 to 5 focused engineering days.
- **Complexity:** **medium-high**.
- **Risk profile:** moderate modeling risk, low operational risk if kept behind a new `path_version` and promotion gate.

## Bottom line
- The best redesign is **not** another bounded feature tweak and **not** four direct classifiers.
- The best fit for this repo is a **shared latent state-aware lead-process model** that predicts phase scoring pressure plus lead-extension / trail-response multipliers, then derives the four anytime markets through a deterministic state pricer inside the existing family framework.

## Senior modeling conclusion
- In abstract, an even richer minute-level state-space or point-process model could be superior.
- In this repo, with prematch PIT data and the existing v2 family architecture, the strongest scientifically defensible next step is **state-aware lead-process modeling inside the family framework**, not more static-rate tuning and not a separate bespoke simulator.

