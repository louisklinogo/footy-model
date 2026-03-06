# V2 Upgrade Tracker

Updated: 2026-03-06
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

4. [TODO] Benchmark corners challengers
   - Champion vs live-scope candidate
   - Global dispersion vs league/shrinkage-aware dispersion
   - Acceptance: improvement on promotion-critical markets without stability regressions
   - Current result: live-scope candidate matched champion metrics on `c85`, `c95`, `c105` (no uplift yet)

5. [TODO] Create scoreline low-score challenger
   - Start with low-score correction, not full replacement
   - Acceptance: better draw/low-score calibration without broad degradation

6. [TODO] Add PIT-safe availability features
   - Missing-player counts / lineup-known flags / freshness
   - Acceptance: leakage-safe feature generation and stable fallback behavior

7. [TODO] Shadow-mode compare and decide promotion
   - Side-by-side predictions under distinct versions
   - Explicit registry verdict
   - Acceptance: rollout decision is reversible and evidence-based

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

