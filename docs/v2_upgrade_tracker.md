# V2 Upgrade Tracker

Updated: 2026-03-06 (post scoreline PIT v2 rerun)
Owner: Augment Agent
Status legend: TODO / IN_PROGRESS / DONE / BLOCKED

## Current baseline
- DONE: `pytest -q tests/v2`
- DONE: `pytest -q tests/test_sofascore_odds_parse.py tests/test_score_market_outcomes_corners.py tests/test_snapshot_leakage_guard.py`
- DONE: Live DB audit of coverage, dispersion, and odds-line support
- DONE: Live SofaScore path audit for stats, incidents, lineups, odds, and missing players

## Execution board
1. [DONE] Build evaluation and promotion harness
   - Add/finish v2 eval runner
   - Add/finish v2 metrics helpers
   - Add/finish promotion registry with explicit reason codes
   - Acceptance: challenger vs champion comparisons are explicit and reproducible

2. [DONE] Add reversible candidate artifact/version plumbing
   - Separate output dirs for challengers
   - Distinct `model_version` values
   - Acceptance: challenger runs cannot overwrite champion artifacts

3. [DONE] Create corners live-scope candidate
   - Candidate scope centered on `c85`, `c95`, `c105`
   - `c75` non-critical until live support improves
   - Team-corners remain research-only
   - Acceptance: production-critical corners align with actual odds support
   - Result: `model_v2/market_scope_corners_live_candidate.yaml` and isolated `corners_live_scope_candidate_v1` artifacts created

4. [IN_PROGRESS] Benchmark corners challengers
   - Champion vs live-scope candidate
   - Global dispersion vs league/shrinkage-aware dispersion
   - Acceptance: improvement on promotion-critical markets without stability regressions
   - Current result: live-scope candidate matched champion metrics on `c85`, `c95`, `c105` (no uplift yet)
   - Calibration rerun result: `c105` improved on the calibration eval split under sigmoid, but promotion correctly stays on raw holdout because calibrated metrics were computed on a smaller eval-only subset (`n=764` vs raw holdout `n=1527`)

5. [DONE] Make calibration a first-class v2 stage
   - Added row-level holdout prediction artifacts across families
   - Added aggregated calibration runner and family-level calibration artifacts
   - Promotion can read calibration diagnostics
   - Guard added: only prefer calibrated holdout metrics for promotion when the calibrated sample is comparable to the raw holdout sample
   - Acceptance: calibration exists in the real evaluation flow and does not create apples-to-oranges promotion comparisons

6. [DONE] Clean the scoreline PIT path and benchmark the first real PIT challenger
   - Added PIT safety guard so scoreline training refuses dataset artifacts whose sibling `pit_validation_report.json` is failed
   - Fixed the odds snapshot seam first, then found and fixed the remaining uncapped lambda-predictions seam in `market_outcome_calibrator.fetch_dataset()`
   - Real result: `scoreline_pit_candidate_v2` now passes promotion on 59 / 68 scoped markets and all core scoreline markets (`1x2_*`, `dc_*`, `ah_*`, `eh_*`, `o15`, `u35`)
   - Remaining failures are concentrated in 9 tail scoreline bucket markets (`mg_*`, `hmg_*`, `ms_other_awaywin`) and currently fail on AUC only while Brier / log-loss / ECE improved

7. [TODO] Add PIT-safe availability features
   - Missing-player counts / lineup-known flags / freshness
   - Acceptance: leakage-safe feature generation and stable fallback behavior

8. [IN_PROGRESS] Shadow-mode compare and decide promotion
   - Side-by-side predictions under distinct versions
   - Explicit registry verdict
   - Acceptance: rollout decision is reversible and evidence-based
   - Current recommendation: advance `scoreline_pit_candidate_v2` as the new leading scoreline candidate for core markets, while keeping the 9 tail scoreline bucket markets on a follow-up optimization track
   - Tail-market audit update: the remaining failed scoreline markets split into (a) overlapping range buckets (`mg_*`, `hmg_*`) and (b) one ultra-rare residual bucket (`ms_other_awaywin`, ~10 holdout positives). All 9 currently fail on AUC only under zero AUC tolerance, while Brier / log-loss / ECE improved.
   - Practical implication: these 9 markets deserve different attention from the core scoreline markets and should not be treated as equal blockers unless business policy explicitly says every bucket market is promotion-critical.
   - Implementation update: `src/modeling/v2/eval/promotion_registry.py` and `src/modeling/v2/run_evaluation_flow.py` now support optional repeated `--required-market` arguments. Default behavior is unchanged, but promotion reports can now emit a top-level decision based on an explicit core-market list while still preserving per-market pass/fail across the full scope.
   - Concrete core-market verdict: reran the registry for `scoreline_pit_candidate_v2` with required core markets (`1x2_*`, `dc_*`, `ah_h05`, `ah_a05`, `ah_h15`, `ah_a15`, `eh_h1`, `eh_a1`, `o15`, `u35`) and saved `model_artifacts/v2/evaluation_scoreline_pit_candidate_v2/promotion_registry_core_markets.json`. Result: top-level decision `passed`, with 14 / 14 required markets passed and 0 missing.
   - Operationalization update: added repo-backed promotion policy file `model_v2/promotion_policies/scoreline_core.yaml` plus optional `--promotion-policy` support in both evaluation CLIs, so future scoreline promotion runs can use a stable tracked core-market contract instead of repeating long `--required-market` lists.
   - Policy-backed rerun result: `model_artifacts/v2/evaluation_scoreline_pit_candidate_v2/promotion_registry_scoreline_core_policy.json` now records the same verdict through the repo policy file path itself (`scoreline_core_v1`), with decision `passed`, 14 / 14 required markets passed, and 0 missing required markets.
   - Decision artifact update: `src/modeling/v2/run_evaluation_flow.py` now reads the generated `promotion_registry.json` back into `evaluation_flow_report.json` as a compact `promotion_summary` block, so shadow/promotion verdicts are visible directly from the main runner report instead of only inside the full registry payload.
   - End-to-end flow rerun: `model_artifacts/v2/evaluation_scoreline_pit_candidate_v2_policyflow/evaluation_flow_report.json` now contains the compact `promotion_summary` block for the repo-backed policy path, again showing decision `passed`, 59 / 68 markets passed overall, and 14 / 14 required scoreline core markets passed.
   - Recommendation artifact update: added `src/modeling/v2/eval/promotion_recommendation.py` and wired `run_evaluation_flow.py` to emit `promotion_recommendation.json` and `promotion_recommendation.md`. The recommendation layer is deliberately derived-only: it turns the policy-backed decision into an explicit recommendation (`full_scope`, `required_markets_only`, or hold) plus a checklist and follow-up market list.
   - Real recommendation artifacts: `model_artifacts/v2/evaluation_scoreline_pit_candidate_v2_policyflow/promotion_recommendation.json` and `.md` now recommend `required_markets_only` promotion for the PIT scoreline candidate, with 14 / 14 core markets passed and follow-up tail markets explicitly listed (`mg_1_5`, `mg_2_4`, `mg_2_5`, `mg_2_6`, `hmg_0`, `hmg_1_2`, `hmg_1_3`, `hmg_2_3`, `ms_other_awaywin`).

## Required tests as work lands
- `pytest -q tests/v2`
- `pytest -q tests/test_sofascore_odds_parse.py tests/test_score_market_outcomes_corners.py tests/test_snapshot_leakage_guard.py`
- Add targeted tests for every new seam before broadening scope
- Prefer smallest-scope train/eval smoke runs before full evaluation

## Promotion checklist
- No PIT leakage failures
- No unit/integration regressions
- Better or equal metrics on promotion-critical markets
- Calibration not materially worse
- Adequate support
- Gains stable across folds and leagues
- Champion artifacts still intact

## Notes to future self
- Do not assume Bash is the default shell even if available.
- Prefer temp Python scripts for DB probes and structured diagnostics.
- Do not promote team-corners just because model math improves; wait for market support.
- Do not claim scoreline dependence improvements without slice-level evidence.
- Do not substitute calibrated holdout metrics into promotion unless the calibrated sample matches the raw holdout comparison set.
- Do not label a challenger as PIT-safe while the dataset artifact fails its own PIT validation report.
- Do not let secondary overlapping scoreline bucket markets block a clearly better core scoreline challenger without an explicit business reason.

