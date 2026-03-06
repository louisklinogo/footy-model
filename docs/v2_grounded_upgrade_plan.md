# V2 Grounded Upgrade Plan

Updated: 2026-03-06
Owner: Augment Agent
Status: Active working plan

## Purpose
Upgrade v2 scoreline and corners modeling using live-DB evidence, strict PIT safety, and rollback-safe promotion rules. Changes must be reversible: no overwrite of current champion artifacts, no silent scope changes, and no promotion without shadow comparison.

## Ground truth from repo + live DB
- Current v2 baseline tests are green: `pytest -q tests/v2` and focused odds/corners/leakage tests passed.
- Corners are overdispersed; plain Poisson is too thin.
- Home/away corners are negatively correlated overall in the live DB, so do not assume positive shared-pressure covariance.
- Aggregate draw rate does not by itself prove the scoreline family is broken.
- Live SofaScore odds support is concentrated at corners totals `8.5`, `9.5`, `10.5`; `7.5` is effectively unsupported.
- Persisted team-corners odds anchors are effectively absent, so team-corners are research-first, not promotion-first.
- SofaScore lineups expose `missingPlayers`; repo ingestion already supports availability-style data.

## Non-negotiable guardrails
1. PIT-safe only: every new feature must respect prediction-time availability.
2. Reversible rollout only: challengers get distinct artifact paths and model versions.
3. Promotion by evidence only: walk-forward + holdout + calibration + support gates.
4. No destructive edits to current champion config/artifacts.

## Artifact and rollout strategy
- Champion artifacts remain untouched under existing `model_artifacts/v2/*` paths.
- Each challenger gets a unique path, e.g.:
  - `model_artifacts/v2/scoreline_dc_candidate_v1`
  - `model_artifacts/v2/corners_live_scope_candidate_v1`
- Each challenger gets a unique `model_version` string.
- Promotion is a config/registry decision, not a file overwrite.

## Phase 1: Evaluation harness first
Goal: make champion vs challenger comparisons explicit and repeatable.

Work:
- Add/complete `src/modeling/v2/eval/run_walkforward.py`.
- Add/complete `src/modeling/v2/eval/metrics.py`.
- Add/complete `src/modeling/v2/eval/promotion_registry.py`.
- Extend baseline registry if needed for explicit missing-baseline failures.

Outputs:
- Per-market metrics: AUC, PR-AUC, Brier, log loss, ECE, support.
- Per-fold and per-league drilldowns.
- Explicit promotion verdicts with reason codes.

## Phase 2: Corners live-scope challenger
Goal: improve corners with low risk by aligning to actual odds support.

Work:
- Keep current mean-model pattern and auto model selection.
- Create a candidate scope prioritizing `c85`, `c95`, `c105`.
- Treat `c75` as non-critical unless coverage improves.
- Keep team-corners outputs research-only until odds support exists.
- Benchmark current global dispersion vs league-aware or shrinkage-aware dispersion.

Promotion-critical markets:
- `c85`, `c95`, `c105`

## Phase 3: Scoreline challenger ladder
Goal: improve scoreline conservatively and only where evidence supports it.

Order:
1. Champion baseline.
2. Same score matrix with better PIT-safe features.
3. Low-score dependence challenger (Dixon-Coles style or equivalent).
4. Richer dependence model only if phase 3 evidence demands it.

Promotion-critical focus:
- draw calibration
- low-score slice quality (`0-0`, `1-0`, `0-1`, `1-1`)
- no broad degradation across 1x2/DC/goals derivatives

## Phase 4: Availability features
Goal: incorporate missing-player signal safely.

Possible features:
- home/away missing-player counts
- role-group counts
- lineup-known vs lineup-unknown flags
- freshness relative to kickoff

Requirements:
- no post-prediction leakage
- safe defaults when lineups are unavailable

## Phase 5: Shadow mode and promotion
- Train challengers separately.
- Predict with distinct versions.
- Compare champion vs challenger offline before any active switch.
- Promote only if predictive, calibration, stability, and support gates pass.

## Evaluation gates
A challenger passes only if:
- no PIT violations
- no unit/integration regression
- better or equal predictive quality on promotion-critical markets
- calibration is not materially worse
- support is sufficient
- gains are not isolated to one fold or one league

## Testing standards
- Unit: math, monotonicity, artifact metadata, promotion reason codes, PIT feature behavior.
- Integration: capped-data train/eval runs write expected artifacts and preserve model isolation.
- Offline eval: walk-forward, holdout, per-league drilldowns, slice analysis.
- Trading-readiness: CLV proxy and ROI only as secondary confirmation, not primary promotion gates.

## Recommended immediate implementation order
1. Build evaluation/promotion plumbing.
2. Add candidate artifact/version plumbing.
3. Add corners live-scope challenger.
4. Add tests for scope gating and promotion logic.
5. Benchmark corners challengers.
6. Benchmark scoreline low-score challenger.
7. Add availability features after PIT-safe evaluation seams exist.

