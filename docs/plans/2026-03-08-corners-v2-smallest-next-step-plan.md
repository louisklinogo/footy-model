## 1. Overview

Run exactly one bounded challenger: re-test the archived `corners_attack_block` feature contract on the current audit cohort, then score it with the current v2 evaluation harness and the live-replacement harness.

Why this is the smallest evidence-backed next step:
- Current corners challenger is broadly behind live on the matched cohort: `model_artifacts/v2/live_replacement_audit_20260308_cal/summary.json` shows `corners_overlap.delta.auc=-0.00233`, `delta.brier=+0.000284`, and only `2/12` AUC market wins.
- The archived attack-block experiment already exists and showed the only repo-backed positive signal for corners: `artifacts/reports/v2_feature_block_experiments/20260305T025316Z/compare_corners_attack_block.md` has `auc_delta_mean=+0.000385`, `brier_delta_mean=-0.000202`, with clearer lift on team-corners (`ac25`, `ac35`, `hc25`, `hc35`) than on totals (`c75/c85/c95/c105`).

Goal:
- Answer one question only: does the archived team-corners-friendly signal still reproduce on the current audit/live cohort?

Out of scope:
- New feature engineering blocks
- New model families or architecture changes
- Promotion/cutover decisions

## 2. Prerequisites

- Existing audit PIT dataset: `artifacts/v2/datasets/audit_20260308/pit_dataset_20260308T083848Z.csv`
- Existing archived contract: `model_v2/feature_contracts/experiments/corners_attack_block.yaml`
- Existing frozen comparison artifacts for other families:
  - `model_artifacts/v2/audit_scoreline_20260308`
  - `model_artifacts/v2/audit_anytime_20260308`
- DB access for `live_replacement_compare.py` (`DATABASE_URL` or repo fallback env vars), because it loads live predictions and actuals.

## 3. Implementation Steps

### Step 1: Train one named corners challenger
- Candidate name:
  - artifact dir: `model_artifacts/v2/corners_attack_block_candidate_20260308`
  - model version: `corners_attack_block_candidate_20260308`
- Command:
  - `python src/modeling/v2/families/corners/train_corners.py --dataset-path artifacts/v2/datasets/audit_20260308/pit_dataset_20260308T083848Z.csv --contract model_v2/feature_contracts/experiments/corners_attack_block.yaml --scope model_v2/market_scope.yaml --output-dir model_artifacts/v2/corners_attack_block_candidate_20260308 --model-version corners_attack_block_candidate_20260308`
- Why this name:
  - matches existing v2 artifact conventions (`challenger_*`, `live_verification_*`)
  - keeps the tested feature block explicit in the artifact identity
- Testing note:
  - inspect `training_report.json`, `metrics_walkforward.json`, and `metrics_holdout.json`

### Step 2: Evaluate it in the current v2 harness
- Command:
  - `python src/modeling/v2/run_evaluation_flow.py --skip-train --scoreline-dir model_artifacts/v2/audit_scoreline_20260308 --corners-dir model_artifacts/v2/corners_attack_block_candidate_20260308 --anytime-dir model_artifacts/v2/audit_anytime_20260308 --evaluation-dir model_artifacts/v2/evaluation_corners_attack_block_candidate_20260308 --baseline-path model_artifacts/v2/baselines/metrics_baseline_v2.json --promotion-policy model_v2/promotion_policies/scoreline_core.yaml`
- What to read:
  - `model_artifacts/v2/evaluation_corners_attack_block_candidate_20260308/evaluation_report.json`
  - `.../holdout_summary.json`
  - `.../walkforward_summary.json`
- Important nuance:
  - the top-level promotion gate is scoreline-core, so use this step for comparable offline metrics, not for a corners go/no-go by itself.

### Step 3: Check whether it actually narrows the live gap
- Command:
  - `python src/modeling/v2/eval/live_replacement_compare.py --scoreline-dir model_artifacts/v2/audit_scoreline_20260308 --corners-dir model_artifacts/v2/corners_attack_block_candidate_20260308 --anytime-dir model_artifacts/v2/audit_anytime_20260308 --output-dir artifacts/v2/family_replacement/corners_attack_block_candidate_20260308_cal --days 3 --backfill-days 60 --apply-calibrators`
- What to read:
  - `artifacts/v2/family_replacement/corners_attack_block_candidate_20260308_cal/summary.json`
  - `.../per_market_comparison.csv`
  - `.../league_family_comparison.csv`
- Focus segments:
  - team corners: `ac25/ac35/ac45/ac55/hc25/hc35/hc45/hc55`
  - totals: `c75/c85/c95/c105`

### Step 4: Apply the keep/stop rule
- Keep investing in the attack-block path only if both are true:
  1. **Offline replication:** team-corners (`ac*` + `hc*`) improve vs current `audit_corners_20260308` on both mean AUC and mean Brier, with at least `5/8` team-corners markets better on Brier and at least `4/8` better on AUC.
  2. **Live-gap narrowing:** the live comparison improves materially from today’s baseline (`delta_auc=-0.00233`, `delta_brier=+0.000284`) to either:
     - `corners_overlap.delta_auc >= -0.0010`, or
     - `corners_overlap.delta_brier <= 0.0` with at least `6/12` corners markets better on Brier.
- Stop this path after this run if either is false.
- Guardrail:
  - even if team-corners improve, stop if totals materially regress (`c*` mean Brier worse and more than `1/4` totals markets lose `>0.005` AUC).

## 4. File Changes Summary

Created by the run:
- `model_artifacts/v2/corners_attack_block_candidate_20260308/`
- `model_artifacts/v2/evaluation_corners_attack_block_candidate_20260308/`
- `artifacts/v2/family_replacement/corners_attack_block_candidate_20260308_cal/`

Modified source files:
- None

Deleted files:
- None

## 5. Testing Strategy

- No new unit tests for this bounded step; this is an evidence-gathering challenger run.
- Validate that training exits `0` and emits holdout/walkforward artifacts.
- Compare candidate vs current corners on:
  - team-corners subgroup
  - totals subgroup
  - pooled corners overlap vs live

## 6. Rollback Plan

- No code rollback needed.
- If the candidate is not promising, keep serving the current corners artifact and discard the new candidate directories from future references.

## 7. Estimated Effort

- Effort: low
- Complexity: low-to-medium
- Expected outcome: a fast yes/no on whether the only evidence-backed corners feature path is still worth pursuing on current data.
