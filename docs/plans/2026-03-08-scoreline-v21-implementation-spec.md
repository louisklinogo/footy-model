# Scoreline v2.1 Implementation Spec

Date: 2026-03-08  
Status: active Phase 1 spec  
Depends on: `docs/plans/2026-03-08-v2-production-standard-plan.md`

## Objective

Make the scoreline family the first production-grade v2 family by improving the shared core markets that matter most on live overlap while preserving scoreline-family coherence and maintainability.

Primary promotion target for this phase:

- `scoreline_directional`: `1x2_*`, `dc_*`, `ah_h05`, `ah_a05`, `ah_h15`, `ah_a15`, `eh_h1`, `eh_a1`
- `scoreline_totals_core`: `o15`, `u35`

Not a Phase 1 promotion blocker:

- `scoreline_multigoals`
- `scoreline_multiscore`

## Benchmark grounding

Current matched-cohort truth from the latest fresh benchmark:

- calibrated scoreline directional is slightly positive vs live in aggregate,
- calibrated scoreline totals core is slightly negative overall,
- runtime calibration materially helps the challenger,
- strongest live-overlap wins are concentrated in `dc_1x`, `1x2_a`, `1x2_d`, `dc_12`,
- weakest shared scoreline markets are `1x2_h`, `dc_x2`, `ah_h05`, `eh_h1`.

Implication: scoreline is the closest family to promotion, but the current base artifact is still not strong enough for a clean handoff.

## Design principle

Keep the existing scoreline structure as the backbone and add a small number of production-grade layers around it:

1. structural scoreline base,
2. stronger direct-odds-aware features,
3. residual correction on top of the base outputs,
4. runtime calibration in serving,
5. explicit feature-tier reporting.

## Existing file map to build on

### Current backbone files

- `src/modeling/v2/families/scoreline/train_scoreline.py`
- `src/modeling/v2/families/scoreline/predict_scoreline.py`
- `src/modeling/v2/families/scoreline/derive_markets.py`
- `src/modeling/v2/run_prediction_flow.py`
- `src/modeling/v2/run_evaluation_flow.py`
- `src/modeling/v2/calibration/run_calibration.py`
- `src/modeling/v2/eval/live_replacement_compare.py`
- `src/modeling/v2/eval/promotion_registry.py`

### Existing config / contract files

- `model_v2/feature_contracts/scoreline.yaml`
- `src/modeling/v2/market_family_registry.py`
- `model_v2/promotion_policies/scoreline_core.yaml`

### Existing reference patterns worth reusing

- `src/modeling/v2/families/scoreline/train_scoreline_o15_stacked.py`
- `src/modeling/v2/families/scoreline/train_scoreline_o15_aux.py`

These already show the right chrono-safe stacked pattern: create base scoreline outputs first, then train a second-stage classifier on top.

## Phase 1 scope

### In scope

- direct-odds-aware scoreline contract expansion,
- scoreline residual overlays for shared core markets,
- runtime application of `calibrators.joblib`,
- feature-tier metadata in scoreline predictions,
- promotion gating that includes live-replacement evidence for scoreline core markets.

### Out of scope

- corners family changes,
- anytime family changes,
- full tail-market optimization,
- full repo-wide market-tier rollout beyond what scoreline needs first.

## Proposed implementation

## Workstream 1 — Strengthen the scoreline feature contract

Update `model_v2/feature_contracts/scoreline.yaml` so the production scoreline contract explicitly admits the strongest already-available scoreline-relevant priors from the fixtures-first / PIT dataset:

- totals odds: `odds_over_15`, `odds_under_15`, `odds_over_25`, `odds_under_25`, `odds_over_35`, `odds_under_35`
- implied totals priors: `implied_over15`, `implied_under15`, `implied_over25`, `implied_under25`, `implied_over35`, `implied_under35`
- odds-gap features: `odds_gap_15`, `odds_gap_25`, `odds_gap_35`
- optional backbone refinement fields already present in the dataset: `adj_lambda_home_final`, `adj_lambda_away_final`, `rule_fired_home`, `rule_fired_away`

Do **not** add corners-specific fields from the forbidden list.

Also add missingness indicators for any newly admitted scoreline-relevant priors that can be absent at prediction time.

## Workstream 2 — Add scoreline residual overlays

Add a second-stage trainer:

- new file: `src/modeling/v2/families/scoreline/train_scoreline_residuals.py`

Purpose:

- train chrono-safe residual models on top of the base scoreline outputs from `train_scoreline.py`.

Minimum residual markets for v2.1:

- `1x2_h`, `1x2_d`, `1x2_a`
- `o15`, `u35`

Derived, not independently trained in v2.1:

- `dc_*` from adjusted `1x2_*`

Conditional extension if first rerun still misses on margin markets:

- add residual heads for margin primitives that support `ah_*` / `eh_*` without training every displayed alias independently.

Residual feature inputs should include:

- base structural probabilities from `derive_markets.py`,
- `lambda_home_pred`, `lambda_away_pred`, total lambda,
- scoreline base outputs like `p_home`, `p_draw`, `p_away`, `p_o15`, `p_u35`,
- direct totals odds / implied totals priors / odds gaps,
- existing availability and lineup indicators from the scoreline contract,
- backbone refinement flags (`adj_lambda_*`, `rule_fired_*`) where available,
- explicit missingness indicators,
- league code / simple tier-safe context features.

Recommended modeling pattern:

- multiclass residual model for `1x2_*`,
- binary residual models for `o15` and `u35`,
- chrono-safe OOF generation from the base scoreline trainer before fitting residual heads.

Expected artifacts in the scoreline artifact dir:

- `residual_models.joblib`
- `residual_features.json`
- `residual_imputation.json`
- `residual_training_report.json`
- `residual_holdout_predictions.csv`

## Workstream 3 — Preserve serving coherence

Update `src/modeling/v2/families/scoreline/predict_scoreline.py` so serving becomes:

1. load base scoreline artifact,
2. generate structural score matrix and base market probabilities,
3. if present, load and apply scoreline residual models,
4. recompute derived `dc_*` from adjusted `1x2_*`,
5. apply family calibrators from `calibrators.joblib`,
6. write final `p_model` plus metadata showing which layers were used.

Runtime metadata should include at least:

- `scoreline_base_only` / `scoreline_residual_applied`,
- `scoreline_calibrated`,
- `scoreline_calibration_method`,
- `scoreline_feature_tier`,
- `lambda_home_pred`, `lambda_away_pred`,
- direct-odds availability flags for the row.

`src/modeling/v2/run_prediction_flow.py` should default to runtime calibration on, with an explicit opt-out flag only for debugging.

## Workstream 4 — Add explicit feature-tier handling

For scoreline Phase 1, define row-level tiers inside `predict_scoreline.py`:

- Tier A: direct odds + backbone refinement + availability features present
- Tier B: direct odds present, refinement partial
- Tier C: no direct odds but structural/base features present
- Tier D: minimal structural fallback only

Persist the tier in prediction metadata and summarize tier counts in a small scoreline runtime report.

## Workstream 5 — Make evaluation reflect the real promoted object

The promoted scoreline candidate should be the final runtime object, not just the raw base trainer.

Required changes:

- `train_scoreline.py` remains the base artifact builder,
- `train_scoreline_residuals.py` must write final scoreline holdout predictions for the markets it owns,
- `src/modeling/v2/eval/run_walkforward.py` and `src/modeling/v2/run_evaluation_flow.py` should continue reading the family artifact dir, but the artifact contents must now represent the post-residual family state,
- `src/modeling/v2/eval/promotion_registry.py` should remain the baseline/holdout/walkforward gate,
- the scoreline production decision must additionally read `live_replacement_compare.py` outputs for the shared core markets.

New scoreline promotion rule:

- internal baseline gate still required,
- live-replacement gate required for shared scoreline core markets,
- no scoreline promotion if runtime calibration is absent but evaluation used it.

## Workstream 6 — Tests and verification

Add or extend targeted tests:

- `tests/v2/test_scoreline_runtime_calibration.py`
- `tests/v2/test_scoreline_residual_train.py`
- `tests/v2/test_scoreline_residual_predict.py`
- `tests/v2/test_scoreline_feature_tiers.py`
- extend `tests/v2/test_live_replacement_compare.py` for residual-aware scoreline artifacts

Verification sequence:

1. focused unit tests for new scoreline files,
2. small smoke train on PIT subset,
3. scoreline-only evaluation flow rerun,
4. scoreline live-replacement rerun,
5. compare calibrated final artifact vs current live on shared scoreline core markets.

## Exit criteria for Phase 1

Scoreline v2.1 is ready for controlled promotion only if all are true:

- directional aggregate remains positive after residual + runtime-calibration parity,
- totals core is no longer net negative on matched live overlap,
- `1x2_h`, `dc_x2`, `ah_h05`, and `eh_h1` are no longer clear weak spots,
- shared core-market live replacement is at least neutral-to-positive by aggregate Brier and log loss,
- calibrated serving behavior matches evaluated behavior,
- feature-tier degradation is visible and acceptable.

## Delivery order

1. update scoreline contract,
2. add residual trainer,
3. wire runtime residual + calibration application,
4. add feature-tier metadata,
5. rerun scoreline evaluation,
6. rerun live replacement,
7. decide whether AH/EH margin overlays are still needed before canary.

## Non-goal reminder

This spec is for the **first production-grade scoreline upgrade**, not the final perfect scoreline model. The goal is to create the smallest maintainable scoreline v2.1 that can credibly take ownership of the shared core scoreline markets.