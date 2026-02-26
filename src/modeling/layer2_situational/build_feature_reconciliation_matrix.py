from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build feature reconciliation matrix from Layer 2 feature ledger"
    )
    parser.add_argument(
        "--ledger-json",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation/layer2_feature_ledger.json"),
        help="Input feature ledger JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation"),
        help="Output directory for reconciliation matrix",
    )
    return parser.parse_args()


def _action_from_decision(decision: str) -> str:
    if decision == "fix_lineage_before_judgment":
        return "repair_upstream_lineage_then_retest"
    if decision == "keep_candidate":
        return "retain_in_candidate_feature_set"
    if decision == "drop_candidate":
        return "drop_in_next_controlled_experiment"
    if decision == "defer_sparse":
        return "defer_until_coverage_improves"
    return "segment_and_retest_before_final_decision"


def main() -> None:
    args = parse_args()
    if not args.ledger_json.exists():
        raise FileNotFoundError(f"Missing ledger JSON: {args.ledger_json}")

    payload = json.loads(args.ledger_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    generated_at = datetime.now(UTC).isoformat()

    matrix_rows: list[dict[str, object]] = []
    for row in rows:
        decision = str(row.get("provisional_decision", "conditional_candidate"))
        matrix_rows.append(
            {
                "feature": row.get("feature"),
                "source": row.get("source"),
                "source_risk": row.get("source_risk"),
                "intended_behavior": row.get("hypothesis"),
                "implemented_in_train": True,
                "implemented_in_predict": True,
                "observed_train_coverage_pct": row.get("train_coverage_pct"),
                "observed_test_coverage_pct": row.get("test_coverage_pct"),
                "observed_single_lift_home": row.get("single_lift_home"),
                "observed_single_lift_away": row.get("single_lift_away"),
                "observed_drop_delta_home_rmse": row.get("drop_delta_home_rmse"),
                "observed_drop_delta_away_rmse": row.get("drop_delta_away_rmse"),
                "decision": decision,
                "action": _action_from_decision(decision),
                "decision_rationale": row.get("rationale"),
            }
        )

    summary = {
        "generated_at": generated_at,
        "train_n": payload.get("train_n"),
        "test_n": payload.get("test_n"),
        "full_home_rmse": payload.get("full_home_rmse"),
        "full_away_rmse": payload.get("full_away_rmse"),
        "baseline_home_rmse": payload.get("baseline_home_rmse"),
        "baseline_away_rmse": payload.get("baseline_away_rmse"),
        "decision_counts": {},
    }

    for row in matrix_rows:
        key = str(row["decision"])
        summary["decision_counts"][key] = int(summary["decision_counts"].get(key, 0)) + 1

    out_payload = {"summary": summary, "rows": matrix_rows}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "layer2_feature_reconciliation_matrix.json"
    md_path = args.output_dir / "layer2_feature_reconciliation_matrix.md"

    json_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Layer 2 Feature Reconciliation Matrix")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("## Summary")
    lines.append(
        f"- baseline RMSE home/away: {summary['baseline_home_rmse']:.4f} / {summary['baseline_away_rmse']:.4f}"
    )
    lines.append(
        f"- full-model RMSE home/away: {summary['full_home_rmse']:.4f} / {summary['full_away_rmse']:.4f}"
    )
    lines.append(
        f"- decision counts: {json.dumps(summary['decision_counts'], ensure_ascii=True)}"
    )
    lines.append("")
    lines.append("| Feature | Source | Risk | Train Cov % | Test Cov % | Decision | Action |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
    for row in matrix_rows:
        lines.append(
            f"| {row['feature']} | {row['source']} | {row['source_risk']} | "
            f"{float(row['observed_train_coverage_pct']):.1f} | "
            f"{float(row['observed_test_coverage_pct']):.1f} | "
            f"{row['decision']} | {row['action']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `implemented_in_train` and `implemented_in_predict` are true because the deployed model payload carries the feature contract.")
    lines.append("- Final keep/drop calls for lineage-blocked sources remain deferred until timestamp repair is in place.")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
