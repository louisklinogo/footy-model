## Session handoff: scoreline family replacement + live runtime audit (2026-03-07)

### Why this handoff exists

This branch now contains several linked changes across scoreline family governance, live-vs-family benchmarking, and live runtime inference hygiene. The next session should not restart discovery from scratch.

### What was completed in this session

1. Split scoreline governance/reporting into four subfamilies while keeping one shared scoreline artifact source:
   - `scoreline_directional`
   - `scoreline_totals_core`
   - `scoreline_multigoals`
   - `scoreline_multiscore`
2. Added canonical family mapping helper:
   - `src/modeling/v2/market_family_registry.py`
3. Updated matched replacement harness to report subfamily segments and load scoreline subfamilies from the shared `scoreline` artifact dir:
   - `src/modeling/v2/eval/live_replacement_compare.py`
4. Updated scope/docs/tests for the split:
   - `model_v2/market_scope.yaml`
   - `tests/v2/test_live_replacement_compare.py`
   - `docs/v2_upgrade_tracker.md`
   - `docs/plans/2026-03-06-family-market-replacement-plan.md`
5. Audited directional fallback root cause in the active live path.
6. Implemented a minimal runtime inference fix for stale binary artifacts:
   - `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py`
   - `tests/test_market_prediction_fallback.py`

### Root-cause findings

The earlier pooled directional weakness was not mainly a scoreline-model issue.

- `1x2_h`, `1x2_a`, `dc_1x`, `dc_x2` had live artifacts on disk, but they were trained on an older feature schema and failed at runtime with `predict_error:ValueError`.
- `ah_*` and `eh_*` still have no active live artifacts and therefore fallback with `missing_artifact`.
- `1x2_d` and `dc_12` avoided fallback because runtime routed them through the multiclass path.

Minimal safe fix implemented:

- if a binary model exposes `feature_names_in_`, align the inference row to that trained feature subset before `predict_proba`.

### Validation already run

Focused tests:

- `pytest -q tests/v2/test_live_replacement_compare.py tests/v2/test_baseline_registry.py`
- `pytest -q tests/test_market_prediction_fallback.py`

End-to-end benchmark reruns:

- `python -m src.modeling.v2.eval.live_replacement_compare --run-name live_replacement_20260307_scoreline_subfamilies`
- `python -m src.modeling.v2.eval.live_replacement_compare --run-name live_replacement_20260307_schema_aligned`

### Canonical artifact paths from this session

- Pre-runtime-fix scoreline-subfamily rerun:
  - `artifacts/v2/family_replacement/live_replacement_20260307_scoreline_subfamilies/summary.json`
- Post-runtime-fix rerun (treat as latest benchmark truth):
  - `artifacts/v2/family_replacement/live_replacement_20260307_schema_aligned/summary.json`
  - `artifacts/v2/family_replacement/live_replacement_20260307_schema_aligned/per_market_comparison.csv`
  - `artifacts/v2/family_replacement/live_replacement_20260307_schema_aligned/matched_prediction_rows.csv`

### Current verdict after the runtime fix

Important: the runtime fix improved directional fairness, but it also changed the overall benchmark picture.

#### Scoreline directional

- `scoreline_directional_overlap` is now slightly positive:
  - fallback rate ~ `48.2%`
  - `delta_auc` ~ `+0.00014`
  - `delta_brier` ~ `-0.00024`
  - `delta_log_loss` ~ `-0.00050`
- `scoreline_directional_overlap_nonfallback` remains positive.
- Remaining directional fallback is now only unsupported `ah_*` / `eh_*` markets.

#### Scoreline totals core

- Still blocked by `o15`.
- `scoreline_totals_core_overlap` remains negative.

#### Scoreline multigoals / multiscore

- No live-overlap basis yet in the benchmark cohort.

#### Big updated conclusion

Once stale-schema fallbacks were removed, the overall family challenger stack no longer clearly beat live on matched overlap.

- `full_overlap` is now slightly negative.
- `full_overlap_nonfallback` is also slightly negative.
- `corners_overlap` and `anytime_overlap` were also negative in the post-fix rerun.

So the old broad conclusion (for example, that corners was clearly ready) should not be trusted without re-reading the benchmark in the post-fix world.

### Recommended next-session TODOs

1. Re-read the post-fix benchmark artifact-by-artifact, not family-by-family in the old sense.
2. Update `docs/v2_upgrade_tracker.md` to reflect the new post-fix truth:
   - directional runtime issue fixed
   - overall matched overlap now slightly favors live
   - old pre-fix benchmark interpretations are partially confounded
3. Decide explicit governance for unsupported handicap directional markets:
   - either regenerate live artifacts for `ah_*` / `eh_*`
   - or mark them intentionally unsupported in live replacement policy
4. Reassess corners with the post-fix benchmark before any productionization decision.
5. Keep `scoreline_totals_core` on live for now because `o15` remains the blocker.

### Recommended next-session order

1. Inspect `live_replacement_20260307_schema_aligned/summary.json` and `per_market_comparison.csv`
2. Write tracker update with corrected interpretation
3. Decide whether to audit the rest of the legacy binary stack for similar stale-artifact behavior
4. Only after that, revisit promotion decisions

### Paste-in starter prompt for the next session

Continue from `docs/plans/2026-03-07-session-handoff-scoreline-replacement.md`.

Use `artifacts/v2/family_replacement/live_replacement_20260307_schema_aligned/summary.json` as the latest benchmark truth. The key nuance is that a runtime schema-alignment fix in `src/modeling/evaluation/predict_market_outcomes_fixtures_first.py` removed stale binary artifact predict-time failures for `1x2_h`, `1x2_a`, `dc_1x`, and `dc_x2`. That made `scoreline_directional` slightly positive, but it also made the overall matched overlap slightly negative for the family challenger vs live. Start by re-reading the post-fix artifacts and updating the tracker/doc conclusions before proposing any promotion decision.