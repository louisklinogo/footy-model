## 1. Overview

Harden the v2 evaluation baseline flow so stale baseline files fail early, before a promotion run can proceed on incomplete coverage.

### Goals and success criteria
- Fail fast when the frozen baseline is missing any scoped market rows needed for promotion.
- Fail fast when effective `required_markets` (CLI + policy) are outside scope or absent from the baseline.
- Preserve current absolute-gate semantics in `src/modeling/v2/eval/promotion_registry.py`: explicit baseline metric values of `null` remain allowed and must not be treated as missing.
- Keep edits minimal and localized to the existing validation seam, evaluation runner, tests, and docs.

### Scope boundaries
Included:
- `src/modeling/v2/io/baseline_registry.py`
- `src/modeling/v2/run_evaluation_flow.py`
- targeted tests under `tests/v2/`
- `docs/v2_evaluation_workflow.md`

Excluded:
- metric-comparison logic in `promotion_registry.py`
- baseline rebuild algorithm changes
- schema, DB, scheduler, or artifact-format changes

## 2. Prerequisites
- No new dependencies, migrations, or data changes.
- Reuse existing helpers:
  - `load_scope_markets()` / `load_baseline_metrics()` from `baseline_registry.py`
  - `resolve_required_markets()` from `src/modeling/v2/eval/promotion_registry.py`
- Keep current baseline JSON shape and current `null` metric handling untouched.

## 3. Implementation Steps

### Step 1: Add presence-only baseline validation helpers
- Modify `src/modeling/v2/io/baseline_registry.py`.
- Add a small reusable helper for validating baseline row presence for an arbitrary market list, with clear error text that names the market set (`scope` vs `required`).
- Keep `validate_baseline_coverage()` backward-compatible by either delegating to the new helper or remaining as the scope-specific wrapper.
- Add/retain a separate scope-membership check for required markets if that is easiest to reuse from the runner.
- Important: presence means “row exists after current market-expansion logic”; a row with `auc/brier/log_loss/ece/n = null` still counts as present.
- Testing: cover both missing-row failures and explicit-null-row success.

### Step 2: Add evaluation-runner preflight validation
- Modify `src/modeling/v2/run_evaluation_flow.py`.
- Add a small preflight helper used when promotion is enabled:
  1. resolve effective required markets via `resolve_required_markets()` so policy-backed runs are validated up front;
  2. load scope markets and ensure required markets are in scope;
  3. load baseline rows and validate frozen-baseline coverage.
- Fail before training/evaluation when `--skip-promotion` is false, `--rebuild-baseline` is false, and the supplied baseline is stale.
- Use actionable errors: missing scope markets, missing required markets, or required markets outside scope.
- Testing: unit-test the helper directly instead of shelling out.

### Step 3: Preserve `--rebuild-baseline` workflow safely
- Modify `src/modeling/v2/run_evaluation_flow.py` only as much as needed.
- Do not fail the run against the old baseline before a requested rebuild.
- Replace the current list-comprehension step runner in `main()` with an explicit loop so the same validation can run immediately after `baseline.rebuild` and before `eval.promotion_registry`.
- This keeps intentional baseline refreshes working while still blocking promotion if the rebuilt baseline is still incomplete.
- Testing: add a focused test for the “defer until after rebuild” behavior at helper/runner level.

### Step 4: Add regression tests for null-metric absolute gates
- Modify `tests/v2/test_baseline_registry.py` and `tests/v2/test_evaluation_harness.py`.
- Extend baseline-registry tests to assert explicit-null rows count as covered.
- Keep or slightly expand the existing harness test `test_build_promotion_registry_allows_absolute_gate_when_baseline_metrics_are_null` so the promotion semantics stay unchanged.
- The hardening patch should validate row presence only; it must not require numeric baseline metrics for every market.

### Step 5: Document the hardened behavior
- Modify `docs/v2_evaluation_workflow.md`.
- Add a short note in the evaluation-flow section that:
  - frozen baseline files are now preflight-validated for scope/required-market coverage;
  - `--rebuild-baseline` intentionally defers that check until after rebuild;
  - explicit `null` baseline metrics are still allowed for markets that use absolute gates.
- Add remediation guidance: refresh the baseline or fix the required-market/policy list when validation fails.

## 4. File Changes Summary

### Created
- None for the implementation itself.

### Modified
- `src/modeling/v2/io/baseline_registry.py`
- `src/modeling/v2/run_evaluation_flow.py`
- `tests/v2/test_baseline_registry.py`
- `tests/v2/test_evaluation_flow_runner.py`
- `tests/v2/test_evaluation_harness.py`
- `docs/v2_evaluation_workflow.md`

### Deleted
- None.

## 5. Testing Strategy
- Unit tests:
  - baseline presence validator raises on missing scoped/required rows;
  - required markets outside scope raise clearly;
  - explicit-null baseline rows are accepted as present;
  - rebuild path defers validation until after rebuild.
- Regression tests:
  - existing `promotion_registry` null-metric absolute-gate behavior still passes.
- Suggested commands:
  - `pytest -q tests/v2/test_baseline_registry.py tests/v2/test_evaluation_flow_runner.py tests/v2/test_evaluation_harness.py`
  - then `pytest -q tests/v2` if the targeted tests pass.

## 6. Rollback Plan
- Revert the runner preflight helper and post-rebuild validation hook in `run_evaluation_flow.py`.
- Revert the new generic validation helper(s) in `baseline_registry.py`.
- Revert the added tests and doc note.
- No DB rollback or artifact migration is required because the hardening change is validation-only.

## 7. Estimated Effort
- Rough estimate: 1-3 hours including tests and docs.
- Complexity: low to medium.
- Main risk: over-validating and accidentally treating intentional `null` baseline metrics as missing; mitigate by keeping validation strictly presence-based and retaining the existing harness regression test.

