# Model V2 Re-Architecture Backlog
Date: 2026-03-03
Owner: Modeling Team
Source: `artifacts/reports/rearchitecture_ideal_2026-03-03.md`

## Current Sprint Todos (2026-03-05)
1. [x] Include contract-level missingness indicators in scoreline/corners feature selection.
2. [x] Add walk-forward fold outputs for scoreline/corners/anytime family trainers.
3. [x] Add v2 feature ablation runner for family heads.
4. [x] Add anti-recency XG anchor features into shared dataset derivation and wire to v2 contracts.
5. [x] Add edge-bucket risk monitoring with auto-pause controls in risk assessment policy flow.
6. [x] Run focused validation tests (`tests/v2`, risk-policy tests) and refresh this checklist with outcomes.
7. [x] Execute Anytime Path V3 structural sprint (`tasks_anytime_path_v3.md`) and decide promotion. (rejected)
 8. [x] Recover anytime with richer live-safe snapshot contract and tuned Poisson regularization. (`anytime_rich_snapshot_v1` promoted to `model_artifacts/v2/anytime`)
 9. [x] Pivot betting execution work to a small-bankroll accumulator strategy layer built on the existing walk-forward backtester.
 10. [x] Validate first backtestable small-bankroll campaign and expose current odds-coverage bottleneck (`o15` only).
 11. [x] Repair scoring/runtime pipeline so canonical repo drives WSL schedulers and historical odds-backed coverage expands to `1x2`/`dc`.

## Principles
1. Linchpin markets only.
2. Point-in-time features only (no leakage).
3. Family-specific models, not one global market model.
4. Calibration and deployment gating are mandatory.
5. `predict_only` is valid when promotion gates fail.
6. Betting execution should optimize target-hitting probability only on whitelisted, promoted slices.

## Scope Lock
In-scope families:
1. `1x2` + `dc`
2. Goals totals (`o15`, `u35`)
3. Corners totals + team corners
4. Handicaps (`ah`, `eh` derivations)
5. `1up/2up` (home/away)

Out-of-scope:
1. Combo markets not in linchpins.
2. Legacy broad market training loop.

## Phase 0 - Contracts and Baselines (Week 1)

### Task 0.1 - Define market scope contract
Files:
1. `model_v2/market_scope.yaml` (new)

Acceptance:
1. Contains exact allowed market codes grouped by family.
2. Any market not listed is excluded from v2 training and export.

### Task 0.2 - Define family feature contracts
Files:
1. `model_v2/feature_contracts/scoreline.yaml` (new)
2. `model_v2/feature_contracts/corners.yaml` (new)
3. `model_v2/feature_contracts/anytime.yaml` (new)

Acceptance:
1. Each contract lists required features, optional features, and missingness flags.
2. No corners-only features in scoreline contract, and vice versa.

### Task 0.3 - Baseline registry v2
Files:
1. `model_artifacts/v2/baselines/metrics_baseline_v2.json` (new)
2. `src/modeling/v2/io/baseline_registry.py` (new)

Acceptance:
1. Baseline exists for every in-scope market.
2. Training fails fast when baseline coverage is incomplete.

## Phase 1 - Point-in-Time Data Platform (Weeks 1-2)

### Task 1.1 - Build unified PIT dataset builder
Files:
1. `src/modeling/v2/data/build_pit_dataset.py` (new)
2. `src/modeling/v2/data/sql/pit_dataset.sql` (new)

Acceptance:
1. Every feature row has `feature_time_utc <= prediction_time_utc`.
2. Leakage checks fail pipeline on violations.
3. Output written to `artifacts/v2/datasets/*.parquet`.

### Task 1.2 - Add feature lineage and freshness metadata
Files:
1. `src/modeling/v2/data/feature_lineage.py` (new)
2. `artifacts/v2/feature_lineage/*.json` (generated)

Acceptance:
1. Each feature has source table, extraction method, freshness hours.
2. High-latency features flagged for exclusion if stale.

### Task 1.3 - Missingness indicators and quality flags
Files:
1. `src/modeling/v2/data/build_pit_dataset.py`

Acceptance:
1. For high-impact features, add `_is_missing` indicators.
2. Missingness rates exported per family.

## Phase 2 - Family Models (Weeks 2-4)

### Task 2.1 - Scoreline head
Files:
1. `src/modeling/v2/families/scoreline/train_scoreline.py` (new)
2. `src/modeling/v2/families/scoreline/predict_scoreline.py` (new)
3. `src/modeling/v2/families/scoreline/derive_markets.py` (new)

Outputs:
1. Joint score distribution artifacts.
2. Derived probabilities for `1x2`, `dc`, `ah`, `eh`, `o15`, `u35`.

Acceptance:
1. Market probabilities are coherent (sum/range constraints pass).
2. Derivation tests pass for identity checks.

### Task 2.2 - Corners distribution head
Files:
1. `src/modeling/v2/families/corners/train_corners.py` (new)
2. `src/modeling/v2/families/corners/predict_corners.py` (new)
3. `src/modeling/v2/families/corners/derive_lines.py` (new)

Outputs:
1. Total corners distribution.
2. Home/away corners distributions.
3. Derived line probabilities (`7.5`, `8.5`, `9.5`, `10.5`, and team lines).

Acceptance:
1. No separate disconnected binary model per corners line in v2.
2. Derived probabilities across lines are monotonic and coherent.

### Task 2.3 - Anytime lead head
Files:
1. `src/modeling/v2/families/anytime/train_anytime.py` (new)
2. `src/modeling/v2/families/anytime/predict_anytime.py` (new)

Acceptance:
1. Outputs valid `h_1up`, `a_1up`, `h_2up`, `a_2up`.
2. Path-dependent sanity checks pass.

## Phase 3 - Calibration and Evaluation (Week 4)

### Task 3.1 - Family calibration pipeline
Files:
1. `src/modeling/v2/calibration/run_calibration.py` (new)
2. `src/modeling/v2/calibration/methods.py` (new)

Acceptance:
1. Chooses isotonic/sigmoid/temperature by holdout Brier + ECE.
2. Rejects calibration method when it harms discrimination materially.

### Task 3.2 - Walk-forward evaluation runner
Files:
1. `src/modeling/v2/eval/run_walkforward.py` (new)
2. `src/modeling/v2/eval/metrics.py` (new)

Acceptance:
1. Exports AUC, PR-AUC, Brier, ECE, log loss, fold std per market.
2. Supports per-league drilldown outputs.

### Task 3.3 - Promotion registry v2
Files:
1. `src/modeling/v2/eval/promotion_registry.py` (new)
2. `model_artifacts/v2/promotion_registry.json` (generated)

Acceptance:
1. Promotion reason is explicit (`passed`, `auc_failed`, `brier_failed`, `ece_failed`, `support_failed`).
2. No silent missing baseline behavior.

## Phase 4 - Inference and Decision Layer (Week 5)

### Task 4.1 - V2 inference writer
Files:
1. `src/modeling/v2/inference/predict_v2.py` (new)
2. `src/modeling/v2/inference/upsert_predictions_v2.py` (new)

Acceptance:
1. Writes v2 probabilities with metadata: `model_family`, `calibration_method`, `prediction_time_utc`.
2. Does not include staking decisions.

### Task 4.2 - Decision policy separation
Files:
1. `src/modeling/v2/decision/assess_risk_v2.py` (new)
2. `model_artifacts/v2/risk_policy_v2.json` (new)

Acceptance:
1. Strict separation between probability generation and staking.
2. Markets can be `tradable` or `predict_only` per promotion + coverage gate.

## Phase 5 - Shadow and Cutover (Weeks 6-7)

### Task 5.1 - Shadow run
Files:
1. `jobs/run_v2_shadow.py` (new)
2. `artifacts/v2/shadow/*.json` (generated)

Acceptance:
1. Daily side-by-side comparison vs incumbent by market.
2. Drift and fallback reports generated automatically.

### Task 5.2 - Controlled cutover
Files:
1. `src/modeling/export/export_market_outcomes_fixtures_first.py` (update)
2. `src/modeling/v2/cutover/enable_v2_markets.py` (new)

Acceptance:
1. Only promoted markets cut over.
2. Rollback script available and tested.

## Test Backlog
1. `tests/v2/test_pit_leakage_guard.py` - fails if feature_time > prediction_time.
2. `tests/v2/test_scoreline_derivations.py` - 1X2/DC/AH/OU identity and range checks.
3. `tests/v2/test_corners_line_monotonicity.py` - derived corners line monotonicity.
4. `tests/v2/test_calibration_selector.py` - calibration method chooser behavior.
5. `tests/v2/test_promotion_registry_rules.py` - gate correctness.
6. `tests/v2/test_decision_layer_separation.py` - no staking logic in inference path.

## Deliverables Checklist
1. V2 feature contracts finalized.
2. PIT builder operational with lineage.
3. Scoreline, corners, anytime heads trained.
4. Calibration and evaluation artifacts produced.
5. Promotion registry and risk gating active.
6. Shadow complete with cutover recommendation.

## First Command Set (Kickoff)
1. `python -m pytest -q tests/v2` (after scaffolding)
2. `python src/modeling/v2/data/build_pit_dataset.py --days 3650`
3. `python src/modeling/v2/families/scoreline/train_scoreline.py`
4. `python src/modeling/v2/families/corners/train_corners.py`
5. `python src/modeling/v2/families/anytime/train_anytime.py`
6. `python src/modeling/v2/eval/run_walkforward.py`
7. `python src/modeling/v2/calibration/run_calibration.py`
8. `python src/modeling/v2/eval/promotion_registry.py`
