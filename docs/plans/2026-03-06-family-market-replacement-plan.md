# Family Market Replacement Plan

Date: 2026-03-06  
Owner: Augment Agent / Senior DS loop  
Status: active working plan  

## Purpose

Make the v2 family architecture the production-grade successor to `market_outcome_gbm / fixtures_first_prematch_v1`.

This plan is canonical for the replacement program strategy. It is not the live runtime source of truth; for that see `docs/current_state.md`.

## Canonical source-of-truth hierarchy

1. Active code paths and live DB behavior.
2. `docs/current_state.md` for current runtime truth.
3. This file for replacement-program strategy, phases, gates, and artifact locations.
4. `docs/v2_upgrade_tracker.md` for execution-board status only.
5. `artifacts/v2/family_replacement/` for reproducible reports and comparisons.

## Decision

Treat the family architecture as the successor path. Do not spend major strategic effort patching the legacy per-market market-outcome stack except where needed for safe benchmarking or temporary operational hygiene.

## Replacement standard

The family stack is replacement-ready only when all are true:
- beats or ties current served behavior on matched markets that it owns,
- is not materially worse on calibration (`brier`, `log_loss`, `ece`),
- preserves internal market coherence and derivation correctness,
- has reproducible artifacts and reversible promotion paths,
- survives a short shadow period cleanly,
- is stable by league/slice, not only in pooled aggregate.

## Owned market families

- **Scoreline family (shared artifact, split governance):**
  - `scoreline_directional`: `1x2_*`, `dc_*`, `ah_*`, `eh_*`
  - `scoreline_totals_core`: `o15`, `u35`
  - `scoreline_multigoals`: `mg_*`, `hmg_*`, `amg_*`
  - `scoreline_multiscore`: `ms_*`
- **Corners family:** `c75`, `c85`, `c95`, `c105`, `hc*`, `ac*`
- **Anytime family:** `h_1up`, `a_1up`, `h_2up`, `a_2up`

Non-family markets remain out of scope until explicitly assigned.

## Phase plan

### Phase 0 — Governance and canonical evaluation contract

Objective:
- lock one plan, one output location, one replacement gate.

Required outputs:
- this plan file,
- unified artifact/report root: `artifacts/v2/family_replacement/`,
- explicit comparison segments: full overlap, family overlap, core markets, by league.

Exit gate:
- no new strategic docs created outside `docs/plans/` for this program.

### Phase 1 — Unified replacement harness

Objective:
- turn the ad hoc head-to-head work into a reusable, repeatable replacement harness.

Work:
- standardize live-vs-family matched-cohort evaluation,
- separate true live-model rows from fallback-driven rows,
- report pooled + by-family + by-league metrics,
- persist comparison outputs under `artifacts/v2/family_replacement/<run_id>/`.

Exit gate:
- one command/report can answer whether a family is replacement-ready.

### Phase 2 — Corners productionization

Objective:
- promote corners from "strong candidate" to first production-grade family pillar.

Work:
- validate totals + team-corners by league,
- verify calibration/monotonicity under matched live comparisons,
- run short shadow validation,
- prepare reversible promotion path.

Exit gate:
- corners clears replacement gate on owned markets and has no operational blockers.

### Phase 3 — Scoreline v2.1 structural upgrade

Objective:
- keep the family approach, improve the score process until scoreline-owned markets are net positive.

Work:
- diagnose failures on `o15`, `1x2_h`, `dc_x2`, `ah_h05`, `eh_h1`,
- upgrade beyond the current independent-Poisson assumption,
- tighten exact market derivations,
- add derived-market calibration where justified.

Exit gate:
- scoreline clears the replacement gate by subfamily ownership, not just aggregate overlap.

### Phase 4 — Anytime path-aware redesign

Objective:
- redesign anytime as a true path model rather than a constant-rate approximation.

Work:
- move from `constant` path logic to `phase_split` or equivalent,
- add timing/state-aware signals,
- test structural model plus optional residual correction layer,
- re-evaluate on matched live cohorts.

Exit gate:
- anytime is at least neutral-to-positive on owned markets with acceptable calibration.

### Phase 5 — Shadow, cutover, and retirement plan

Objective:
- promote families only when individually ready; retire legacy ownership family-by-family.

Work:
- shadow ready families,
- codify family ownership and rollback rules,
- cut over in reversible steps,
- keep legacy benchmark only for audit/backstop purposes.

Exit gate:
- family-owned production scope is explicit and reversible.

## Immediate execution order

1. Build the unified replacement harness.
2. Productionize corners.
3. Run scoreline root-cause diagnostics and ship v2.1.
4. Redesign anytime.
5. Run shadow and decide family-by-family cutover.

## File placement rules for this program

- Strategy/design docs: `docs/plans/`
- Backlog/execution status: `docs/v2_upgrade_tracker.md` + task list
- Reproducible run outputs: `artifacts/v2/family_replacement/`
- Candidate artifacts: `model_artifacts/v2/<candidate_name>/`
- No standalone root-level notes/tasks for this program.