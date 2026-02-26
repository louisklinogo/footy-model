# Layer 1 Stabilization Sweep Review

Date: 2026-02-26  
Scope: Layer 1 (`lambda_xgb`) only

## Objective
Validate two potential Layer 1 boosters before more Layer 2 iteration:
1. Time-decay tuning (`xi`)
2. Add `shots_inside_box` rolling features

## What Was Run
- Sweep script: `src/modeling/layer1_poisson/run_layer1_feature_sweep.py`
- Artifact outputs:
  - `artifacts/reports/layer1_sweep/layer1_feature_sweep.json`
  - `artifacts/reports/layer1_sweep/layer1_feature_sweep.md`

Configurations tested:
- `xi` in `{0.000, 0.001, 0.002, 0.003}`
- each with and without `shots_inside_box` rolling feature block

## Headline Result
Best config:
- `xi_0.001_core`
- Mean RMSE: `1.1243` (home `1.1738`, away `1.0748`)

Focused hyperparameter follow-up (full frame):
- `regB_xi001_depth3_mcw25_lr025`
- Mean RMSE: `1.1241` (home `1.1738`, away `1.0743`)
- Source: `artifacts/reports/layer1_sweep/layer1_hyperparam_sweep_full.md`

## Decision on `shots_inside_box`
Current answer: **do not include in Layer 1 production feature set yet**.

Reason:
- Across this sweep, `+shots_inside_box` variants were slightly worse on average than core variants.
- Coverage was not the issue (100% in this sweep frame); incremental signal quality was.

## Time-Decay Decision
Current answer: **set Layer 1 default `xi` to `0.001`**.

Reason:
- It was the best-performing setting in this controlled sweep.
- Gains are modest but directionally consistent versus heavier decay.

## Hyperparameter Promotion Decision
Current answer: **promote the winning regularization profile as Layer 1 defaults**.

Promoted defaults:
- `learning_rate=0.025`
- `max_depth=3`
- `min_child_weight=25`
- `subsample=0.7`
- `colsample_bytree=0.7`

## Implementation Updates Applied
1. `src/modeling/layer1_poisson/train_lambda.py`
   - Added full CLI parameterization for training controls:
     - `--xi`
     - `--learning-rate`
     - `--max-depth`
     - `--min-child-weight`
     - `--subsample`
     - `--colsample-bytree`
     - `--num-boost-round`
     - `--early-stopping-rounds`
   - Default `xi` set to `0.001`
   - Promoted regularization defaults listed above
   - Persists `xi` in `calibration_params.json`
   - Persists booster parameters in `calibration_params.json`
   - Normalized logging output to ASCII
   - Writes model binaries to canonical inference path `model_artifacts/poisson_model/` (and legacy mirror path for compatibility)

2. Trained Layer 1 artifacts using:
   - `python src/modeling/layer1_poisson/train_lambda.py`

3. Refreshed predictions partially (chunk run):
   - `python src/modeling/layer1_poisson/predict_lambda.py --limit 4000 --batch-size 400`
   - Saved predictions: `7092`

4. Completed full-universe refresh:
   - `python src/modeling/layer1_poisson/predict_lambda.py --batch-size 500`
   - Saved predictions: `20578`
   - Post-refresh anatomy coverage improved from `6906` to `7137` pre-kickoff fixture pairs.

5. Confirmatory residual anatomy (post-promotion):
   - `python src/modeling/layer2_situational/audit_layer1_residual_anatomy.py`
   - Full pre-kickoff residuals (`n=7137`):
     - home bias `+0.0149`
     - away bias `-0.0080`
     - RMSE home/away `1.1959 / 1.0756`
   - Recent pre-kickoff residuals (`n=6447`):
     - home bias `+0.0108`
     - away bias `+0.0101`
     - RMSE home/away `1.1750 / 1.0653`

## Remaining Layer 1 Stabilization Work
1. Continue monitoring league-specific error pockets via scheduled validation runs.
2. Revisit Layer 1 features only if new data assets improve quality (not just quantity), with one-block-at-a-time additions.
