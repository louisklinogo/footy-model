# V2 Evaluation Workflow

## Current entrypoint

Use the single runner:

- `python src/modeling/v2/run_evaluation_flow.py`

This currently orchestrates the v2 steps that exist in the repo today:

1. `train_scoreline.py`
2. `train_corners.py`
3. `train_anytime.py`
4. `eval/run_walkforward.py`
5. `eval/promotion_registry.py`

By default, promotion compares the evaluated artifacts against the supplied baseline file as a frozen champion registry.

`io/rebuild_baseline_metrics.py` is available, but only when you opt in with `--rebuild-baseline`.

## Common variants

- Smoke run with capped rows:
  - `python src/modeling/v2/run_evaluation_flow.py --max-rows 5000`
- Reuse existing family artifacts and only rebuild evaluation outputs:
  - `python src/modeling/v2/run_evaluation_flow.py --skip-train`
- Rebuild the baseline registry explicitly from current artifacts:
  - `python src/modeling/v2/run_evaluation_flow.py --rebuild-baseline`
- Allow incomplete holdout coverage when explicitly rebuilding the baseline:
  - `python src/modeling/v2/run_evaluation_flow.py --rebuild-baseline --allow-missing-holdout`
- Preview commands without executing:
  - `python src/modeling/v2/run_evaluation_flow.py --dry-run`

## Champion vs challenger usage

- Champion baseline file should remain frozen, for example:
  - `model_artifacts/v2/baselines/metrics_baseline_v2.json`
- Challenger family artifacts should be written to separate directories and passed via:
  - `--scoreline-dir`
  - `--corners-dir`
  - `--anytime-dir`

Do not rebuild the champion baseline as part of a challenger evaluation run unless you are intentionally refreshing the frozen baseline itself.

For prediction-time isolation, challengers should also carry a distinct `model_version` inside their artifact directory. Example training shape:

- `python src/modeling/v2/families/scoreline/train_scoreline.py --output-dir model_artifacts/v2/scoreline_dc_candidate_v1 --model-version scoreline_dc_candidate_v1`
- `python src/modeling/v2/families/corners/train_corners.py --output-dir model_artifacts/v2/corners_live_scope_candidate_v1 --model-version corners_live_scope_candidate_v1`

The family prediction scripts now read `artifact_metadata.json` automatically, so DB writes can coexist for champion and challenger runs without sharing the same `(model_name, model_version)` key.

## Main outputs

- Family artifacts:
  - `model_artifacts/v2/scoreline/`
  - `model_artifacts/v2/corners/`
  - `model_artifacts/v2/anytime/`
- Aggregated evaluation:
  - `model_artifacts/v2/evaluation/evaluation_report.json`
  - `model_artifacts/v2/evaluation/walkforward_summary.json`
  - `model_artifacts/v2/evaluation/walkforward_summary_by_league.json`
  - `model_artifacts/v2/evaluation/holdout_summary.json`
  - `model_artifacts/v2/evaluation/promotion_registry.json`
  - `model_artifacts/v2/evaluation/evaluation_flow_report.json`
- Baseline registry:
  - `model_artifacts/v2/baselines/metrics_baseline_v2.json`

## Note on calibration

Planning notes mention `src/modeling/v2/calibration/run_calibration.py`, but that file is not present in the current workspace.

So the current runner intentionally stops at promotion gating based on:

- holdout metrics
- walk-forward summaries
- baseline comparison

When a v2 calibration runner is added, it should slot between:

1. `eval/run_walkforward.py`
2. `eval/promotion_registry.py`
