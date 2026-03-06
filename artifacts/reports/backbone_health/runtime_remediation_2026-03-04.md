# Runtime Backbone Remediation Report
Date: 2026-03-04
Owner: Codex

## Scope
- Execute step 7: resolve policy-enabled league failures under strict adjusted-lambda coverage gate.
- Execute all remaining safe automated steps that can run immediately.

## Chronological Work Log

1. Baseline check of policy-enabled leagues
- Source: `model_artifacts/situational_model/layer2_deployment_policy.json`
- Enabled leagues detected: `E3`, `RO1`

2. Pre-patch strict gate validation (3-day horizon)
- `E3`: **FAIL**
  - `total=1`, `l1_pairs=1`, `adj_pairs=0`
  - Failure reason: `adj_coverage_enabled_league_below_threshold(actual=0.0000, threshold=0.9500, league=E3)`
  - Evidence: `artifacts/reports/backbone_health/backbone_health_E3_pre_patch.json`
- `RO1`: **PASS**
  - `total=0` (no scheduled fixtures in horizon)
  - Evidence: `artifacts/reports/backbone_health/backbone_health_RO1_pre_patch.json`

3. Policy remediation applied
- Updated `model_artifacts/situational_model/layer2_deployment_policy.json`:
  - `E3.enabled: true -> false`
  - `E3.alpha: 0.5 -> 0.0`
  - `E3.reason: manual_runtime_disable_adj_coverage_below_95pct_2026-03-04`
  - Appended gate fail reason marker: `runtime_adj_coverage_below_95pct`

4. Post-patch strict gate validation
- `E3`: **PASS**
  - `total=1`, `l1_pairs=1`, `adj_pairs=0`, `layer2_enabled=false`
  - Evidence: `artifacts/reports/backbone_health/backbone_health_E3_post_patch.json`
- Global backbone check: **PASS**
  - `total=60`, `l1_pairs=60`, `adj_pairs=30`, `l1_late=0`, `l1_unknown_asof=0`
  - Evidence: `artifacts/reports/backbone_health/backbone_health_global_post_patch.json`

## Current Policy State (key leagues)
- `E3`: disabled (runtime safety override)
- `RO1`: still enabled (no active fixtures in 3-day horizon during this check)

## What Was Completed vs What Is Left

### Completed now
- Step 7 completed for active failing enabled league (`E3`).
- Strict gate behavior verified pre/post.
- Evidence reports saved under `artifacts/reports/backbone_health/`.

### You still need to run
1. Full non-dry-run tick in your normal production window:
   - `python src/jobs/tick_due_fixtures_v1.py --predict-days 3`
2. Confirm generated per-league backbone reports:
   - `artifacts/reports/backbone_health/backbone_health_<LEAGUE>.json`
3. If any enabled league fails adjusted coverage in live run:
   - either backfill/fix situational generation coverage for that league
   - or disable that league in policy until coverage is stable.

## Recommendation
- Keep strict gate enabled as-is.
- Re-enable `E3` only after repeated runs show adjusted-lambda coverage >= 95% for active scheduled fixtures plus no negative lift regressions.
