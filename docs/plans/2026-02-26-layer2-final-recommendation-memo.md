# Layer 2 Final Recommendation Memo

Date: 2026-02-26
Scope: Layer 2 situational residual model governance after timing repair and controlled reruns

## Executive Decision
- Recommended production candidate now: `baseline_full_features` (`model_artifacts/situational_model/`).
- Do not promote `candidate_keyabs_only` right now.
- Keep the Layer 2 reconciliation freeze active for broad rollout decisions until calibration sample and serving-readiness gates are satisfied.

## Why This Is the Recommendation
1. Aggregate lift is positive for the full model versus Layer 1 baseline.
   - Home RMSE: `1.1963 -> 1.1910` (+0.44%)
   - Away RMSE: `1.0854 -> 1.0783` (+0.66%)
2. The keyabs-only variant is a tradeoff, not a clear win.
   - Home RMSE worsens: `1.1910 -> 1.1924`
   - Away RMSE improves: `1.0783 -> 1.0762`
   - Enabled leagues narrow from `E1, RO1` to `RO1` only.
3. Historical availability timing leakage is repaired.
   - FT availability late rows: `0`
   - This removes the previous hard blocker for injury-feature historical validity.

## What Is Still Not Fully Resolved
1. Feature governance quality is still mixed.
   - Audit snapshot: 36 features, 1 implemented well, 35 with issues (mostly mixed incremental value and drop-candidate evidence).
2. Upcoming availability coverage remains unpopulated in the latest 14-day monitor window (`0/630`).
   - This is a prediction-time readiness gap, not a historical training leakage gap.
3. Calibration diagnostics are sample-limited.
   - Latest run had too few scored rows per market for decision-grade calibration conclusions.

## Policy Position
1. Keep current deployed Layer 2 artifact family as the full-feature model.
2. Keep derby-driven features in deferred/conditional status until prevalence support improves.
3. Treat injury-derived signals as conditional and monitor their incremental value in controlled reruns.
4. Keep league-level gates strict; do not broaden enabled leagues without passing gates.

## Next Decision Gates To Close Freeze
1. Calibration gate:
   - collect enough scored market rows to run stable ECE/reliability bins.
2. Serving-readiness gate:
   - populate upcoming pre-kickoff availability coverage before trusting injury features for scheduled fixtures.
3. Governance gate:
   - run one explicit sign-off pass for keep/fix/drop/defer with current post-repair evidence.

## Evidence Artifacts
- `artifacts/reports/layer2_reconciliation/layer2_source_timing_checks.json`
- `artifacts/reports/layer2_reconciliation/layer2_feature_intent_implementation_audit.json`
- `artifacts/reports/layer2_reconciliation/layer2_postrepair_full_vs_keyabs_comparison.md`
- `model_artifacts/situational_model/layer2_deployment_policy.json`
