# V2 Ambitious Challengers Branch Plan

Date: 2026-03-06
Branch: `research/v2-ambitious-challengers`
Status: active working plan

## 1. Overview

This branch should produce a promotion-safe challenger ladder for v2 with four priorities, in order:
1. scoreline redesign,
2. calibration as a first-class v2 stage,
3. PIT-safe availability features,
4. corners totals redesign only if evidence still justifies it.

Success means:
- challenger artifacts and `model_version` values remain isolated from the current champion,
- the frozen baseline at `model_artifacts/v2/baselines/metrics_baseline_v2.json` stays unchanged,
- every candidate is evaluated through the existing v2 harness,
- promotion decisions are evidence-based and reversible.

Out of scope:
- broad legacy market retraining,
- team-corners promotion before odds support improves,
- destructive overwrite of champion artifacts or baselines.

## 2. Prerequisites

- Keep using the current v2 conventions already present in the repo:
  - scope contracts in `model_v2/`
  - family artifacts in `model_artifacts/v2/<candidate_name>/`
  - aggregated eval outputs in `model_artifacts/v2/evaluation_<candidate_name>/`
  - frozen champion baseline in `model_artifacts/v2/baselines/metrics_baseline_v2.json`
- Reuse the existing harness files instead of creating parallel flows:
  - `src/modeling/v2/run_evaluation_flow.py`
  - `src/modeling/v2/eval/run_walkforward.py`
  - `src/modeling/v2/eval/promotion_registry.py`
- Treat current facts from repo docs as starting assumptions:
  - scoreline currently uses an independent Poisson score matrix,
  - calibration is planned but not yet implemented in v2,
  - corners live-scope candidate exists but has not shown uplift yet,
  - availability is historically repaired but serving-readiness remains a gate.

## 3. Implementation Steps

### Step 1: Persist branch governance before new modeling work
- Create and maintain this branch doc as the source of truth for candidate names, hypotheses, gates, and stop/go decisions.
- Persist in the doc:
  - champion reference paths,
  - challenger naming scheme,
  - promotion-critical markets and slices,
  - required evidence artifacts,
  - rollback rule: promotion is registry/config only.
- Candidate naming to reserve now:
  - `scoreline_pit_candidate_v1`
  - `scoreline_calibrated_candidate_v1`
  - `scoreline_dc_candidate_v1`
  - `scoreline_availability_candidate_v1`
  - `corners_dispersion_candidate_v1` (conditional)

### Step 2: Make calibration a first-class stage before promotion
- Add `src/modeling/v2/calibration/run_calibration.py` and `src/modeling/v2/calibration/methods.py`.
- Insert calibration into the existing flow between walk-forward and promotion, matching the gap already called out in `docs/v2_evaluation_workflow.md`.
- Update `src/modeling/v2/run_evaluation_flow.py` and `src/modeling/v2/eval/promotion_registry.py` so promotion reads calibrated holdout outputs, selected method, and calibration diagnostics.
- Persist per-family calibration artifacts under each challenger artifact dir plus summarized outputs in `evaluation_<candidate>/`.
- Gate on Brier, log loss, ECE, and "do not materially harm discrimination".

Update (2026-03-06): implemented. Real rerun on `corners_live_scope_candidate_v1` showed `c105` selects sigmoid on the calibration eval split, improving Brier/log loss/ECE there, but promotion now only prefers calibrated holdout metrics when the calibrated sample is comparable to raw holdout. This avoids apples-to-oranges gating when calibration diagnostics are computed on an eval-only subset.

### Step 3: Re-baseline scoreline with slice-first diagnostics
- Keep the current scoreline champion frozen, but document that `src/modeling/v2/families/scoreline/train_scoreline.py` is still an independent-Poisson head using legacy dataset plumbing.
- Add explicit scoreline evaluation slices for:
  - draw rate,
  - `0-0`, `1-0`, `0-1`, `1-1`,
  - 1X2/DC/goals derivative stability,
  - by-league low-score behavior.
- Persist these slice reports in the evaluation outputs so scoreline redesign is judged on the actual failure modes, not aggregate metrics alone.

### Step 4: Build the scoreline challenger ladder in strict order
1. `scoreline_pit_candidate_v1`
   - Keep the same independent score matrix, but move training onto the v2 PIT dataset/contract path.
   - Goal: separate data-plumbing gains from distributional-model gains.
   - Update (2026-03-06, earlier): initial real PIT dataset build at `artifacts/v2/datasets/scoreline_pit_candidate_v1/` failed validation on 7,594 / 8,170 rows because `odds_snapshot_time_utc` commonly reflected closing/kickoff timing after the nominal prediction timestamp. Added a safety guard so scoreline training refuses PIT dataset artifacts whose sibling validation report is failed.
   - Update (2026-03-06, later): after fixing both the odds cutoff and the remaining uncapped lambda-predictions seam, the rerun candidate `scoreline_pit_candidate_v2` became the real reference result. It now passes 59 / 68 scoped promotion checks, including all core scoreline markets (`1x2_*`, `dc_*`, `ah_*`, `eh_*`, `o15`, `u35`). The remaining failures are concentrated in 9 tail scoreline bucket markets (`mg_*`, `hmg_*`, `ms_other_awaywin`) that fail on AUC only while Brier / log-loss / ECE improve.
2. `scoreline_calibrated_candidate_v1`
   - Apply first-class calibration to the PIT-safe scoreline challenger.
   - Goal: see whether calibration closes draw/low-score gaps before changing the model family.
3. `scoreline_dc_candidate_v1`
   - Add a low-score dependence correction layer (Dixon-Coles-style or equivalent) on top of the same PIT-safe feature base.
   - Goal: improve low-score and draw slices without broad regression in derived markets.
4. Optional richer scoreline challenger
   - Only if the calibrated low-score challenger still leaves clear residual failure modes.
   - Examples: richer dependence or count-family redesign, but only after stepwise evidence.

### Step 5: Add PIT-safe availability features only after the scoreline seam is clean
- Extend `src/modeling/v2/data/build_pit_dataset.py` and scoreline contracts to represent availability with explicit timing semantics.
- Start with conservative features only:
  - home/away missing-player counts,
  - role-group counts,
  - lineup-known flags,
  - freshness-to-kickoff features,
  - safe-default missingness flags.
- Do not use availability features that require post-prediction knowledge or assume upcoming coverage is always present.
- First target family should be scoreline, not corners.

Update (2026-03-06): the scoreline PIT seam is now clean enough to proceed. Odds snapshots and lambda-prediction joins are both capped to prediction time for PIT builds, and `scoreline_pit_candidate_v2` provides the current clean baseline for later availability-enriched scoreline challengers.
- Promote only if scoreline gains remain stable when upcoming availability coverage is sparse.

### Step 6: Decide whether corners totals need redesign
- Treat corners as conditional work, not equal priority with scoreline.
- First re-run corners through the stronger calibration stage and current live-scope convention (`c85`, `c95`, `c105`).
- Only open `corners_dispersion_candidate_v1` if one of these holds:
  - promotion-critical corners lines still underperform after calibration,
  - league-specific dispersion clearly beats global dispersion,
  - live odds support changes enough to justify a wider scope.
- Keep team-corners research-only until persisted odds support exists.

Update (2026-03-06): real rerun completed for the isolated live-scope corners challenger. Result remains: `c85`, `c95`, and `c105` all pass promotion gates, but the candidate does not show uplift versus the frozen champion on the production-critical corners lines. Corners redesign therefore remains conditional, not immediate priority.

### Step 7: Promotion and shadow comparison
- For every serious challenger, require:
  - isolated artifact dir,
  - distinct `model_version`,
  - `evaluation_<candidate>/` output set,
  - comparison against frozen baseline,
  - explicit registry verdict with reason codes.
- Promotion order should be:
  1. calibrated scoreline challenger,
  2. dependent scoreline challenger if it beats the calibrated-only version,
  3. availability-enriched scoreline challenger,
  4. corners challenger only if conditional trigger is met.
- Practical update (2026-03-06): `scoreline_pit_candidate_v2` is now the leading clean scoreline reference candidate. Promotion/use can be recommended for the core scoreline markets without waiting on the remaining tail bucket clean-up, but those 9 residual markets should stay on a separate optimization track.
- Policy update (2026-03-06): the promotion registry now supports optional repeated `--required-market` arguments so future evaluation runs can separate the top-level promotion decision for core markets from the informational pass/fail reporting of secondary tail buckets.

## 4. File Changes Summary

### Create
- `docs/plans/2026-03-06-v2-ambitious-challengers-branch-plan.md`
- `src/modeling/v2/calibration/run_calibration.py`
- `src/modeling/v2/calibration/methods.py`
- targeted `tests/v2/` calibration and slice-eval tests

### Modify
- `src/modeling/v2/run_evaluation_flow.py`
- `src/modeling/v2/eval/promotion_registry.py`
- `src/modeling/v2/eval/run_walkforward.py`
- `src/modeling/v2/data/build_pit_dataset.py`
- `src/modeling/v2/families/scoreline/train_scoreline.py`
- `src/modeling/v2/families/scoreline/predict_scoreline.py`
- `model_v2/feature_contracts/scoreline.yaml`
- optionally `src/modeling/v2/families/corners/train_corners.py` if Step 6 triggers

### Delete
- none planned

## 5. Testing Strategy

- Unit tests:
  - calibration method selection,
  - calibrated-vs-raw metric gating,
  - scoreline low-score slice summaries,
  - PIT availability timing guards,
  - artifact metadata and model-version isolation.
- Integration tests:
  - capped `run_evaluation_flow.py` challenger run writes isolated artifacts,
  - promotion registry fails cleanly on missing baselines or failed calibration gates.
- Manual/offline checks:
  - compare challenger vs champion on holdout, walk-forward, and by-league reports,
  - inspect low-score scoreline slices before claiming dependence-model wins,
  - inspect corners only after calibration-first rerun.

## 6. Rollback Plan

- No rollback migration is needed if the branch follows current conventions.
- Revert by leaving champion paths and frozen baseline untouched and discarding challenger dirs/config references.
- If a candidate was shadowed, disable it by registry/config selection rather than deleting artifacts.

## 7. Estimated Effort

- Effort: medium-high
- Complexity: high for scoreline + calibration, medium for availability features, conditional for corners.
- Recommended execution order by week-block:
  1. governance + calibration plumbing,
  2. scoreline PIT challenger,
  3. calibrated scoreline challenger,
  4. low-score dependence challenger,
  5. availability-enriched scoreline challenger,
  6. corners redesign only if still justified by evidence.

