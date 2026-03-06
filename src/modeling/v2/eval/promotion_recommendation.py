from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in value:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def build_promotion_recommendation(
    flow_report: dict[str, Any],
    promotion_registry: dict[str, Any] | None,
) -> dict[str, Any] | None:
    promotion_summary = flow_report.get("promotion_summary")
    if not isinstance(promotion_summary, dict):
        return None
    decision = promotion_summary.get("decision") if isinstance(promotion_summary.get("decision"), dict) else {}
    summary = promotion_summary.get("summary") if isinstance(promotion_summary.get("summary"), dict) else {}
    rules = promotion_summary.get("rules") if isinstance(promotion_summary.get("rules"), dict) else {}
    registry_summary = promotion_registry.get("summary") if isinstance((promotion_registry or {}).get("summary"), dict) else {}

    decision_status = str(decision.get("status") or "unknown")
    decision_basis = str(decision.get("basis") or "unknown")
    overall_failed = int(summary.get("markets_failed") or 0)
    overall_passed = int(summary.get("markets_passed") or 0)
    overall_total = int(summary.get("markets_total") or (overall_passed + overall_failed))
    required_total = int(summary.get("required_markets_total") or 0)
    required_passed = int(summary.get("required_markets_passed") or 0)
    required_failed = int(summary.get("required_markets_failed") or 0)
    missing_required = _string_list(summary.get("required_markets_missing") or decision.get("missing_required_markets"))
    failed_required_markets = _string_list(decision.get("failed_markets"))
    failed_scoped_markets = _string_list(registry_summary.get("failed_markets"))

    recommendation_status = "hold"
    recommended_scope = "none"
    headline = "Hold promotion until the recorded gate passes."
    rationale: list[str] = []
    if decision_status == "passed" and decision_basis == "required_markets" and overall_failed > 0:
        recommendation_status = "promote"
        recommended_scope = "required_markets_only"
        headline = "Promote for required/core markets only; keep tail markets on a follow-up track."
        rationale = [
            f"Required promotion gate passed ({required_passed}/{required_total}).",
            f"Overall scoped evaluation still has {overall_failed} non-required failed markets.",
        ]
    elif decision_status == "passed":
        recommendation_status = "promote"
        recommended_scope = "full_scope"
        headline = "Promote for the full scoped market set."
        rationale = [
            f"Top-level promotion decision passed on basis={decision_basis}.",
            f"Scoped evaluation reports {overall_passed}/{overall_total} markets passed.",
        ]
    else:
        rationale = [
            f"Top-level promotion decision status is {decision_status}.",
            f"Required gate has {required_failed} failed and {len(missing_required)} missing required markets.",
        ]

    checklist = [
        {
            "item": "promotion_registry_generated",
            "ok": True,
            "detail": str(promotion_summary.get("promotion_registry_path") or ""),
        },
        {
            "item": "top_level_decision_passed",
            "ok": decision_status == "passed",
            "detail": decision_status,
        },
        {
            "item": "required_markets_complete",
            "ok": len(missing_required) == 0,
            "detail": f"missing={len(missing_required)}",
        },
        {
            "item": "required_markets_passed",
            "ok": required_failed == 0 and len(failed_required_markets) == 0,
            "detail": f"passed={required_passed}/{required_total}",
        },
        {
            "item": "full_scope_markets_passed",
            "ok": overall_failed == 0,
            "detail": f"passed={overall_passed}/{overall_total}",
        },
    ]
    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "evaluation_flow_report_path": flow_report.get("evaluation_flow_report_path"),
        "evaluation_dir": flow_report.get("evaluation_dir"),
        "promotion_registry_path": promotion_summary.get("promotion_registry_path"),
        "promotion_policy_name": rules.get("promotion_policy_name"),
        "promotion_policy_path": rules.get("promotion_policy_path"),
        "decision": decision,
        "summary": summary,
        "recommendation_status": recommendation_status,
        "recommended_scope": recommended_scope,
        "headline": headline,
        "rationale": rationale,
        "follow_up_markets": failed_scoped_markets,
        "checklist": checklist,
    }


def render_promotion_recommendation_markdown(recommendation: dict[str, Any]) -> str:
    checklist = recommendation.get("checklist") if isinstance(recommendation.get("checklist"), list) else []
    decision = recommendation.get("decision") if isinstance(recommendation.get("decision"), dict) else {}
    summary = recommendation.get("summary") if isinstance(recommendation.get("summary"), dict) else {}
    follow_up_markets = _string_list(recommendation.get("follow_up_markets"))
    lines = [
        "# Promotion Recommendation",
        "",
        f"- Generated at: {recommendation.get('generated_at_utc')}",
        f"- Recommendation: {recommendation.get('recommendation_status')}",
        f"- Recommended scope: {recommendation.get('recommended_scope')}",
        f"- Headline: {recommendation.get('headline')}",
        f"- Policy: {recommendation.get('promotion_policy_name') or 'none'}",
        f"- Decision basis: {decision.get('basis')}",
        "",
        "## Evidence",
        "",
        f"- Top-level decision: {decision.get('status')}",
        f"- Scoped markets passed: {summary.get('markets_passed')} / {summary.get('markets_total')}",
        f"- Required markets passed: {summary.get('required_markets_passed')} / {summary.get('required_markets_total')}",
        f"- Required markets missing: {summary.get('required_markets_missing')}",
        "",
        "## Checklist",
        "",
    ]
    for item in checklist:
        if not isinstance(item, dict):
            continue
        mark = "[x]" if bool(item.get("ok")) else "[ ]"
        lines.append(f"- {mark} {item.get('item')}: {item.get('detail')}")
    lines.extend(["", "## Rationale", ""])
    for reason in recommendation.get("rationale") if isinstance(recommendation.get("rationale"), list) else []:
        lines.append(f"- {reason}")
    lines.extend(["", "## Follow-up markets", ""])
    if follow_up_markets:
        for market in follow_up_markets:
            lines.append(f"- {market}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def write_promotion_recommendation_artifacts(
    *,
    flow_report: dict[str, Any],
    evaluation_dir: Path,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]] | None:
    promotion_summary = flow_report.get("promotion_summary")
    if not isinstance(promotion_summary, dict):
        return None
    registry_path = Path(str(promotion_summary.get("promotion_registry_path") or "")).resolve()
    promotion_registry = _read_json_object(registry_path)
    recommendation = build_promotion_recommendation(flow_report, promotion_registry)
    if recommendation is None:
        return None
    json_path = output_json or (evaluation_dir / "promotion_recommendation.json")
    md_path = output_md or (evaluation_dir / "promotion_recommendation.md")
    json_path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    md_path.write_text(render_promotion_recommendation_markdown(recommendation), encoding="utf-8")
    return json_path, md_path, recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a promotion recommendation artifact from an evaluation flow report.")
    parser.add_argument("--evaluation-flow-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow_report = _read_json_object(args.evaluation_flow_report)
    if flow_report is None:
        raise RuntimeError(f"Could not read evaluation flow report: {args.evaluation_flow_report}")
    flow_report.setdefault("evaluation_flow_report_path", str(args.evaluation_flow_report))
    result = write_promotion_recommendation_artifacts(
        flow_report=flow_report,
        evaluation_dir=args.evaluation_flow_report.parent,
        output_json=args.output_json,
        output_md=args.output_markdown,
    )
    if result is None:
        raise RuntimeError("No promotion_summary present in evaluation flow report; nothing to recommend.")
    json_path, md_path, recommendation = result
    print(f"Saved recommendation JSON: {json_path}")
    print(f"Saved recommendation Markdown: {md_path}")
    print(
        f"recommendation_status={recommendation['recommendation_status']} scope={recommendation['recommended_scope']}"
    )


if __name__ == "__main__":
    main()