from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from src.modeling.v2.io.artifact_identity import load_artifact_metadata


FAMILY_DEFAULTS = {
    "scoreline": {"model_name": "scoreline_v2", "model_version": "poisson_head_v1"},
    "corners": {"model_name": "corners_v2", "model_version": "distribution_head_v1"},
    "anytime": {"model_name": "anytime_v2", "model_version": "markov_head_v1"},
}


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolved_model_version(metadata: dict[str, Any], defaults: dict[str, Any], artifact_dir: Path | None) -> str | None:
    return (
        metadata.get("model_version")
        or defaults.get("model_version")
        or (artifact_dir.name if artifact_dir is not None else None)
    )


def _identity_from_artifact_dir(path_value: str | None, family: str) -> dict[str, Any]:
    defaults = FAMILY_DEFAULTS.get(family, {})
    artifact_dir = Path(path_value) if path_value else None
    metadata = load_artifact_metadata(artifact_dir) if artifact_dir is not None else {}
    return {
        "family": family,
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        "model_name": metadata.get("model_name") or defaults.get("model_name") or family,
        "model_version": _resolved_model_version(metadata, defaults, artifact_dir),
        "artifact_created_at": metadata.get("created_at"),
    }


def _identity_from_holdout_path(path_value: str | None, family: str) -> dict[str, Any]:
    defaults = FAMILY_DEFAULTS.get(family, {})
    holdout_path = Path(path_value) if path_value else None
    artifact_dir = holdout_path.parent if holdout_path is not None else None
    metadata = load_artifact_metadata(artifact_dir) if artifact_dir is not None else {}
    return {
        "family": family,
        "holdout_path": str(holdout_path) if holdout_path is not None else None,
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        "model_name": metadata.get("model_name") or defaults.get("model_name") or family,
        "model_version": _resolved_model_version(metadata, defaults, artifact_dir),
        "artifact_created_at": metadata.get("created_at"),
    }


def _baseline_holdout_source(baseline_sources: dict[str, Any], family: str) -> str | None:
    path_value = baseline_sources.get(f"{family}_holdout_path")
    if path_value:
        return str(path_value)
    fallback_value = baseline_sources.get(f"{family}_holdout")
    return str(fallback_value) if fallback_value else None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def build_shadow_comparison_report(
    *,
    flow_report: dict[str, Any],
    baseline: dict[str, Any],
    promotion_registry: dict[str, Any],
) -> dict[str, Any]:
    candidate_artifacts = {
        family: _identity_from_artifact_dir(flow_report.get(f"{family}_dir"), family)
        for family in FAMILY_DEFAULTS
    }
    baseline_sources = baseline.get("sources") if isinstance(baseline.get("sources"), dict) else {}
    baseline_artifacts = {
        family: _identity_from_holdout_path(_baseline_holdout_source(baseline_sources, family), family)
        for family in FAMILY_DEFAULTS
    }

    market_rows: list[dict[str, Any]] = []
    improvement_summary = {"auc": 0, "brier": 0, "log_loss": 0, "ece": 0}
    for row in list(promotion_registry.get("markets") or []):
        if not isinstance(row, dict):
            continue
        family = str(row.get("family") or "")
        baseline_metrics = row.get("baseline") if isinstance(row.get("baseline"), dict) else {}
        challenger_metrics = row.get("effective_holdout") if isinstance(row.get("effective_holdout"), dict) else {}
        delta_auc = None
        delta_brier = None
        delta_log_loss = None
        delta_ece = None
        auc = _safe_float(challenger_metrics.get("auc"))
        baseline_auc = _safe_float(baseline_metrics.get("auc"))
        if auc is not None and baseline_auc is not None:
            delta_auc = auc - baseline_auc
            if delta_auc > 0:
                improvement_summary["auc"] += 1
        brier = _safe_float(challenger_metrics.get("brier"))
        baseline_brier = _safe_float(baseline_metrics.get("brier"))
        if brier is not None and baseline_brier is not None:
            delta_brier = brier - baseline_brier
            if delta_brier < 0:
                improvement_summary["brier"] += 1
        log_loss = _safe_float(challenger_metrics.get("log_loss"))
        baseline_log_loss = _safe_float(baseline_metrics.get("log_loss"))
        if log_loss is not None and baseline_log_loss is not None:
            delta_log_loss = log_loss - baseline_log_loss
            if delta_log_loss < 0:
                improvement_summary["log_loss"] += 1
        ece = _safe_float(challenger_metrics.get("ece"))
        baseline_ece = _safe_float(baseline_metrics.get("ece"))
        if ece is not None and baseline_ece is not None:
            delta_ece = ece - baseline_ece
            if delta_ece < 0:
                improvement_summary["ece"] += 1

        market_rows.append(
            {
                "market": row.get("market"),
                "family": family,
                "status": row.get("status"),
                "reasons": list(row.get("reasons") or []),
                "baseline_artifact": baseline_artifacts.get(family),
                "candidate_artifact": candidate_artifacts.get(family),
                "baseline": baseline_metrics,
                "challenger": challenger_metrics,
                "deltas": {
                    "auc": delta_auc,
                    "brier": delta_brier,
                    "log_loss": delta_log_loss,
                    "ece": delta_ece,
                },
            }
        )

    summary = promotion_registry.get("summary") if isinstance(promotion_registry.get("summary"), dict) else {}
    decision = promotion_registry.get("decision") if isinstance(promotion_registry.get("decision"), dict) else {}
    recommendation = flow_report.get("promotion_recommendation") if isinstance(flow_report.get("promotion_recommendation"), dict) else {}
    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "evaluation_dir": flow_report.get("evaluation_dir"),
        "baseline_path": flow_report.get("baseline_path"),
        "promotion_registry_path": (flow_report.get("promotion_summary") or {}).get("promotion_registry_path"),
        "decision": decision,
        "promotion_recommendation": recommendation,
        "summary": summary,
        "candidate_artifacts": candidate_artifacts,
        "baseline_artifacts": baseline_artifacts,
        "improvement_summary": improvement_summary,
        "markets": market_rows,
    }


def _build_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision") if isinstance(report.get("decision"), dict) else {}
    recommendation = report.get("promotion_recommendation") if isinstance(report.get("promotion_recommendation"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Shadow comparison report",
        "",
        f"- Generated at: {report.get('generated_at_utc')}",
        f"- Decision: {decision.get('status', 'unknown')}",
        f"- Recommended scope: {recommendation.get('scope', 'n/a')}",
        f"- Required markets passed: {summary.get('required_markets_passed', 0)}/{summary.get('required_markets_total', 0)}",
        "",
        "## Artifact versions",
        "",
    ]
    for family, candidate in sorted((report.get("candidate_artifacts") or {}).items()):
        baseline = (report.get("baseline_artifacts") or {}).get(family) or {}
        lines.append(
            f"- {family}: baseline `{baseline.get('model_version', '-')}` vs challenger `{candidate.get('model_version', '-')}`"
        )
    lines.extend(
        [
            "",
            "## Market deltas",
            "",
            "| Market | Family | Status | ΔAUC | ΔBrier | ΔLogLoss | ΔECE |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in list(report.get("markets") or []):
        deltas = row.get("deltas") if isinstance(row.get("deltas"), dict) else {}
        lines.append(
            "| {market} | {family} | {status} | {auc} | {brier} | {log_loss} | {ece} |".format(
                market=row.get("market") or "-",
                family=row.get("family") or "-",
                status=row.get("status") or "-",
                auc=_format_metric(_safe_float(deltas.get("auc"))),
                brier=_format_metric(_safe_float(deltas.get("brier"))),
                log_loss=_format_metric(_safe_float(deltas.get("log_loss"))),
                ece=_format_metric(_safe_float(deltas.get("ece"))),
            )
        )
    return "\n".join(lines) + "\n"


def write_shadow_comparison_artifacts(
    *,
    flow_report: dict[str, Any],
    evaluation_dir: Path,
) -> dict[str, Any] | None:
    baseline_path = Path(str(flow_report.get("baseline_path") or ""))
    promotion_registry_path = Path(evaluation_dir) / "promotion_registry.json"
    baseline = _read_json_object(baseline_path)
    promotion_registry = _read_json_object(promotion_registry_path)
    if baseline is None or promotion_registry is None:
        return None

    report = build_shadow_comparison_report(
        flow_report=flow_report,
        baseline=baseline,
        promotion_registry=promotion_registry,
    )
    json_path = Path(evaluation_dir) / "shadow_comparison.json"
    md_path = Path(evaluation_dir) / "shadow_comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "market_count": len(list(report.get("markets") or [])),
    }