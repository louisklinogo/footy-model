# V2 Evaluation Workflow

> Canonical for the v2 evaluation/challenger workflow.
> This workflow is not currently the active scheduled prediction path; see `docs/current_state.md`.

## Current entrypoint

Use the single runner:

- `python src/modeling/v2/run_evaluation_flow.py`

This currently orchestrates the v2 steps that exist in the repo today:

1. `train_scoreline.py`
2. `train_corners.py`
3. `train_anytime.py`
4. `eval/run_walkforward.py`
5. `calibration/run_calibration.py`
6. `eval/promotion_registry.py`

By default, promotion compares the evaluated artifacts against the supplied baseline file as a frozen champion registry.

`io/rebuild_baseline_metrics.py` is available, but only when you opt in with `--rebuild-baseline`.

The evaluation runner now preflight-validates frozen baselines before promotion runs:

- every scoped market must have a baseline row present
- every effective required market (CLI + promotion policy) must be inside scope
- every effective required market must also have a baseline row present

Important:

- this is a **presence** check, not a metric-completeness check
- explicit baseline rows with `null` metrics are still allowed for markets that intentionally use absolute gates
- if you run with `--rebuild-baseline`, the runner defers this validation until after the rebuild step

## Common variants

- Smoke run with capped rows:
  - `python src/modeling/v2/run_evaluation_flow.py --max-rows 5000`
- Reuse existing family artifacts and only rebuild evaluation outputs:
  - `python src/modeling/v2/run_evaluation_flow.py --skip-train`
- Rebuild the baseline registry explicitly from current artifacts:
  - `python src/modeling/v2/run_evaluation_flow.py --rebuild-baseline`
- Allow incomplete holdout coverage when explicitly rebuilding the baseline:
  - `python src/modeling/v2/run_evaluation_flow.py --rebuild-baseline --allow-missing-holdout`
- Evaluate a challenger but make the top-level promotion decision depend only on explicit core markets:
  - `python src/modeling/v2/run_evaluation_flow.py --skip-train --required-market 1x2_h --required-market 1x2_d --required-market 1x2_a`
- Use the repo-backed scoreline core-market policy instead of repeating markets on the CLI:
  - `python src/modeling/v2/run_evaluation_flow.py --skip-train --promotion-policy model_v2/promotion_policies/scoreline_core.yaml`
- Preview commands without executing:
  - `python src/modeling/v2/run_evaluation_flow.py --dry-run`

## Champion vs challenger usage

- Champion baseline file should remain frozen, for example:
  - `model_artifacts/v2/baselines/metrics_baseline_v2.json`
- Challenger family artifacts should be written to separate directories and passed via:
  - `--scoreline-dir`
  - `--corners-dir`
  - `--anytime-dir`

Recommended practice after the 2026-03-08 stale-baseline incident:

- treat generic baseline files as mutable registries, not as unquestioned truth
- prefer explicit named frozen baseline snapshots for important promotion decisions
- if preflight validation fails, refresh the frozen baseline intentionally or fix the required-market/policy list before trusting any promotion result

Current repo-backed scoreline challenger reference:

- scoreline artifact: `model_artifacts/v2/scoreline_v21_total_intensity_snap_20260308/`
- passing evaluation: `model_artifacts/v2/evaluation_scoreline_v21_total_intensity_snap_20260308_eps/`
- pinned named baseline snapshot: `model_artifacts/v2/baselines/metrics_baseline_v2_scoreline_v21_total_intensity_snap_20260308_eps.json`

Use `--baseline-path model_artifacts/v2/baselines/metrics_baseline_v2_scoreline_v21_total_intensity_snap_20260308_eps.json` when you want future scoreline comparisons to anchor to this passing named snapshot instead of the mutable generic baseline registry.

Do not rebuild the champion baseline as part of a challenger evaluation run unless you are intentionally refreshing the frozen baseline itself.

## Current repo-backed bundle reference

- Scoreline artifact: `model_artifacts/v2/scoreline_v21_total_intensity_snap_20260308/`
- Corners artifact: `model_artifacts/v2/live_verification_corners_20260306/`
- Anytime artifact: `model_artifacts/v2/anytime_direct_monotone_v1_candidate_20260308/`
- Baseline snapshot: `model_artifacts/v2/baselines/metrics_baseline_v2_scoreline_v21_total_intensity_snap_20260308_eps.json`
- Latest bundled evaluation output: `model_artifacts/v2/evaluation_bundle_refresh_anytime_direct_monotone_v1_20260309/`
- Latest bundled live-replacement output: `artifacts/v2/family_replacement/live_replacement_20260309_bundle_anytime_direct_monotone_v1/`

Reference command shape for re-evaluating the current bundled challenger without retraining families:

- `python src/modeling/v2/run_evaluation_flow.py --skip-train --scope model_v2/market_scope.yaml --scoreline-dir model_artifacts/v2/scoreline_v21_total_intensity_snap_20260308 --corners-dir model_artifacts/v2/corners --anytime-dir model_artifacts/v2/anytime_direct_monotone_v1_candidate_20260308 --baseline-path model_artifacts/v2/baselines/metrics_baseline_v2_scoreline_v21_total_intensity_snap_20260308_eps.json --evaluation-dir model_artifacts/v2/evaluation_bundle_refresh_anytime_direct_monotone_v1_20260309 --promotion-policy model_v2/promotion_policies/scoreline_core.yaml`

The `--corners-dir model_artifacts/v2/corners` selector currently resolves to `model_artifacts/v2/live_verification_corners_20260306/` in the emitted bundle artifacts.

Current corners research leader for future challenger work:

- `model_artifacts/v2/corners_totals_first_style_matchup_possession_box_touches_corners_against_featurepass_autoresearch_v1_candidate_20260312/`
- evaluation: `model_artifacts/v2/evaluation_corners_totals_first_style_matchup_possession_box_touches_corners_against_featurepass_autoresearch_v1_candidate_20260312/`

This is the current best non-promoted corners branch (`67 / 76` scoped markets passed). Use it explicitly via `--corners-dir` for further challenger experiments, but keep the official repo-backed bundle reference on `live_verification_corners_20260306/` until promotion actually passes.

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
  - `model_artifacts/v2/evaluation/calibration_report.json`
  - `model_artifacts/v2/evaluation/walkforward_summary.json`
  - `model_artifacts/v2/evaluation/walkforward_summary_by_league.json`
  - `model_artifacts/v2/evaluation/holdout_summary.json`
  - `model_artifacts/v2/evaluation/promotion_registry.json`
  - `model_artifacts/v2/evaluation/evaluation_flow_report.json` (now includes a compact `promotion_summary` block when promotion runs)
  - `model_artifacts/v2/evaluation/promotion_recommendation.json`
  - `model_artifacts/v2/evaluation/promotion_recommendation.md`
- Baseline registry:
  - `model_artifacts/v2/baselines/metrics_baseline_v2.json`

## Note on calibration

The runner now includes `src/modeling/v2/calibration/run_calibration.py` between walk-forward aggregation and promotion.

Calibration consumes row-level `holdout_predictions.csv` artifacts, prefers chronological split/eval when timestamps are available, and writes:

- family-level `calibration_report.json`
- family-level `calibrators.joblib` when a non-identity method is selected
- aggregated `model_artifacts/v2/evaluation/calibration_report.json`

Promotion will prefer calibrated holdout metrics when available and otherwise fall back to raw holdout metrics.

If `--required-market` is supplied (repeatable), the generated promotion registry still reports pass/fail for every scoped market, but also emits a top-level `decision` section based only on those required markets.

If `--promotion-policy` is supplied, the registry loads `required_markets` from that YAML file, merges them with any repeated `--required-market` values, and records the policy metadata in the output rules block.

The evaluation flow also writes a derived promotion recommendation artifact pair (`promotion_recommendation.json` and `.md`) that converts the registry decision into an explicit recommendation plus checklist. If the required gate passes but non-required scoped markets still fail, the recommendation scope is `required_markets_only` rather than `full_scope`.
