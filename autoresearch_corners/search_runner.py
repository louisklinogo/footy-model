from __future__ import annotations

import argparse
from datetime import UTC, datetime
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from autoresearch_corners.interaction_library import (
    BLOCKS_BY_NAME,
    get_block,
    list_blocks,
    managed_feature_names,
    selected_feature_names,
)
from autoresearch_corners.run_experiment import (
    _resolve_repo_path,
    build_run_plan,
    resolve_experiment,
    run_plan,
    write_run_summary,
)
from src.modeling.v2.io.contracts import load_feature_contract
DEFAULT_CONFIG = Path(__file__).with_name("search.local.json")
EXAMPLE_CONFIG = Path(__file__).with_name("search.example.json")
STATUS_ORDER = {"keep": 0, "promising_but_unstable": 1, "reject": 2, "failed": 3, "dry_run": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded corners autoresearch over named interaction blocks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_search_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        if path == DEFAULT_CONFIG:
            raise FileNotFoundError(
                f"Missing {path}. Copy {EXAMPLE_CONFIG.name} to {DEFAULT_CONFIG.name} and edit it first."
            )
        raise FileNotFoundError(f"Missing search config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Search config must be a JSON object.")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def resolve_search(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = ("candidate_prefix", "dataset_path", "base_contract_path", "path_version")
    missing = [key for key in required if not str(payload.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Search config missing required keys: {', '.join(missing)}")
    configured_blocks = payload.get("block_names")
    if configured_blocks is None:
        block_names = [block.name for block in list_blocks()]
    else:
        block_names = [str(name).strip() for name in configured_blocks if str(name).strip()]
    if not block_names:
        raise ValueError("Search config must include at least one block name.")
    configured_required_blocks = payload.get("required_block_names")
    if configured_required_blocks is None:
        required_block_names: list[str] = []
    else:
        required_block_names = [
            str(name).strip() for name in configured_required_blocks if str(name).strip()
        ]
    unknown = sorted(name for name in block_names if name not in BLOCKS_BY_NAME)
    if unknown:
        raise ValueError(f"Unknown block names: {', '.join(unknown)}")
    unknown_required = sorted(name for name in required_block_names if name not in BLOCKS_BY_NAME)
    if unknown_required:
        raise ValueError(f"Unknown required block names: {', '.join(unknown_required)}")
    overlap = sorted(set(block_names).intersection(required_block_names))
    if overlap:
        raise ValueError(
            f"Block names cannot also be required blocks: {', '.join(overlap)}"
        )
    max_blocks = int(payload.get("max_blocks_per_candidate", 1))
    if max_blocks <= 0:
        raise ValueError("max_blocks_per_candidate must be positive.")
    return {
        "candidate_prefix": str(payload["candidate_prefix"]).strip(),
        "dataset_path": _resolve_repo_path(str(payload["dataset_path"])),
        "base_contract_path": _resolve_repo_path(str(payload["base_contract_path"])),
        "scope_path": _resolve_repo_path(str(payload.get("scope_path", "model_v2/market_scope.yaml"))),
        "artifact_root": _resolve_repo_path(str(payload.get("artifact_root", "model_artifacts/v2"))),
        "comparison_root": _resolve_repo_path(
            str(payload.get("comparison_root", "artifacts/v2/family_replacement"))
        ),
        "results_dir": _resolve_repo_path(str(payload.get("results_dir", "autoresearch_corners/results"))),
        "scoreline_dir": _resolve_repo_path(str(payload.get("scoreline_dir", "model_artifacts/v2/scoreline"))),
        "anytime_dir": _resolve_repo_path(str(payload.get("anytime_dir", "model_artifacts/v2/anytime"))),
        "path_version": str(payload["path_version"]),
        "model_type": str(payload.get("model_type", "auto")),
        "max_rows": payload.get("max_rows"),
        "folds": payload.get("folds"),
        "min_fold_test_n": payload.get("min_fold_test_n"),
        "days": int(payload.get("days", 3)),
        "backfill_days": int(payload.get("backfill_days", 60)),
        "league": payload.get("league"),
        "limit": payload.get("limit"),
        "apply_calibrators": bool(payload.get("apply_calibrators", True)),
        "include_baseline": bool(payload.get("include_baseline", True)),
        "max_blocks_per_candidate": max_blocks,
        "required_block_names": required_block_names,
        "block_names": block_names,
        "notes": payload.get("notes"),
    }


def candidate_block_combinations(
    block_names: list[str],
    *,
    max_blocks_per_candidate: int,
    include_baseline: bool,
) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = [tuple()] if include_baseline else []
    max_size = min(max_blocks_per_candidate, len(block_names))
    for size in range(1, max_size + 1):
        combos.extend(tuple(combo) for combo in itertools.combinations(block_names, size))
    return combos


def candidate_tag(prefix: str, block_names: tuple[str, ...]) -> str:
    if not block_names:
        return f"{prefix}__baseline"
    aliases = [get_block(name).short_name for name in block_names]
    return f"{prefix}__{'__'.join(aliases)}"


def _write_contract(
    path: Path,
    *,
    family: str,
    required_features: Iterable[str],
    optional_features: Iterable[str],
    missingness_indicators: Iterable[str],
    forbidden_cross_family_features: Iterable[str],
    disabled_features: Iterable[str],
) -> None:
    def _append_list(lines_out: list[str], key: str, values: Iterable[str]) -> None:
        rows = list(values)
        if not rows:
            lines_out.append(f"{key}: []")
            return
        lines_out.append(f"{key}:")
        for feature in rows:
            lines_out.append(f"  - {feature}")

    lines = ["version: 1", f"family: {family}"]
    _append_list(lines, "required_features", required_features)
    _append_list(lines, "optional_features", optional_features)
    _append_list(lines, "missingness_indicators", missingness_indicators)
    _append_list(lines, "forbidden_cross_family_features", forbidden_cross_family_features)
    _append_list(lines, "disabled_features", disabled_features)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_candidate_payload(
    *,
    search: Mapping[str, Any],
    block_combo: tuple[str, ...],
    generated_dir: Path,
) -> dict[str, Any]:
    contract = load_feature_contract(Path(search["base_contract_path"]))
    managed = set(managed_feature_names())
    required_blocks = tuple(search.get("required_block_names", ()))
    active_blocks = (*required_blocks, *block_combo)
    candidate_name = candidate_tag(str(search["candidate_prefix"]), active_blocks)
    base_optional = [feature for feature in contract.optional_features if feature not in managed]
    selected_optional = list(base_optional)
    for feature in selected_feature_names(active_blocks):
        if feature not in selected_optional:
            selected_optional.append(feature)
    contract_path = generated_dir / "contracts" / f"{candidate_name}.yaml"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    _write_contract(
        contract_path,
        family=contract.family,
        required_features=contract.required_features,
        optional_features=selected_optional,
        missingness_indicators=contract.missingness_indicators,
        forbidden_cross_family_features=contract.forbidden_cross_family_features,
        disabled_features=contract.disabled_features,
    )
    artifact_dir = Path(search["artifact_root"]) / candidate_name
    comparison_dir = Path(search["comparison_root"]) / f"live_replacement_{candidate_name}"
    payload = {
        "candidate_tag": candidate_name,
        "model_version": candidate_name,
        "dataset_path": str(search["dataset_path"]),
        "contract_path": str(contract_path),
        "path_version": str(search["path_version"]),
        "scope_path": str(search["scope_path"]),
        "artifact_dir": str(artifact_dir),
        "comparison_dir": str(comparison_dir),
        "scoreline_dir": str(search["scoreline_dir"]),
        "anytime_dir": str(search["anytime_dir"]),
        "results_dir": str(Path(search["results_dir"])),
        "model_type": str(search["model_type"]),
        "days": int(search["days"]),
        "backfill_days": int(search["backfill_days"]),
        "apply_calibrators": bool(search["apply_calibrators"]),
        "notes": f"blocks={','.join(active_blocks) if active_blocks else 'baseline'}",
    }
    for key in ("max_rows", "folds", "min_fold_test_n", "league", "limit"):
        value = search.get(key)
        if value is not None:
            payload[key] = value
    config_path = generated_dir / "configs" / f"{candidate_name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["_generated_contract_path"] = str(contract_path)
    payload["_generated_config_path"] = str(config_path)
    payload["_selected_blocks"] = list(active_blocks)
    return payload


def summarize_holdout_metrics(path: Path) -> dict[str, float | int] | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        return None
    out: dict[str, float | int] = {"markets": int(len(rows))}
    for key in ("auc", "brier", "log_loss", "ece"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            out[f"{key}_mean"] = float(sum(values) / len(values))
    return out


def parse_live_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segment_summaries")
    if not isinstance(segments, list):
        return None
    preferred = ("corners_overlap_nonfallback", "corners_overlap")
    by_name = {
        str(segment.get("segment")): segment
        for segment in segments
        if isinstance(segment, dict) and segment.get("segment")
    }
    for segment_name in preferred:
        if segment_name in by_name:
            segment = by_name[segment_name]
            delta = segment.get("delta") if isinstance(segment.get("delta"), dict) else {}
            return {
                "segment": segment_name,
                "rows": int(segment.get("rows", 0)),
                "delta_auc": float(delta.get("auc", 0.0)),
                "delta_brier": float(delta.get("brier", 0.0)),
                "delta_log_loss": float(delta.get("log_loss", 0.0)),
                "delta_ece": float(delta.get("ece", 0.0)),
                "market_win_counts": segment.get("market_win_counts"),
            }
    return None


def classify_candidate(live_summary: dict[str, Any] | None) -> str:
    if live_summary is None:
        return "failed"
    auc = float(live_summary.get("delta_auc", 0.0))
    brier = float(live_summary.get("delta_brier", 0.0))
    log_loss = float(live_summary.get("delta_log_loss", 0.0))
    if auc > 0.0 and brier < 0.0 and log_loss < 0.0:
        return "keep"
    if brier < 0.0 and log_loss < 0.0:
        return "promising_but_unstable"
    return "reject"


def build_leaderboard_markdown(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "# Corners Autoresearch Leaderboard",
        "",
        "| Candidate | Blocks | Status | Segment | ΔAUC | ΔBrier | ΔLogLoss | Holdout AUC | Holdout Brier |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        live = row.get("live_summary") or {}
        holdout = row.get("holdout_summary") or {}
        lines.append(
            f"| {row['candidate_tag']} | {', '.join(row['blocks']) or 'baseline'} | {row['status']} | {live.get('segment', '')} | "
            f"{live.get('delta_auc', '')} | {live.get('delta_brier', '')} | {live.get('delta_log_loss', '')} | "
            f"{holdout.get('auc_mean', '')} | {holdout.get('brier_mean', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_search(
    *,
    search: Mapping[str, Any],
    python_bin: str,
    dry_run: bool,
) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    search_run_dir = Path(search["results_dir"]) / f"{search['candidate_prefix']}_{timestamp}"
    generated_dir = search_run_dir / "generated"
    combos = candidate_block_combinations(
        list(search["block_names"]),
        max_blocks_per_candidate=int(search["max_blocks_per_candidate"]),
        include_baseline=bool(search["include_baseline"]),
    )
    candidate_rows: list[dict[str, Any]] = []
    for combo in combos:
        payload = build_candidate_payload(search=search, block_combo=combo, generated_dir=generated_dir)
        experiment = resolve_experiment(payload)
        plan = build_run_plan(experiment, python_bin=python_bin)
        candidate_run_dir = search_run_dir / experiment["candidate_tag"]
        results = run_plan(plan, run_dir=candidate_run_dir, dry_run=dry_run)
        write_run_summary(
            config_path=Path(payload["_generated_config_path"]),
            experiment=experiment,
            plan=plan,
            results=results,
            run_dir=candidate_run_dir,
            dry_run=dry_run,
        )
        live_summary = None
        holdout_summary = None
        status = "dry_run" if dry_run else "failed"
        if not dry_run and all(int(row["returncode"]) == 0 for row in results):
            live_summary = parse_live_summary(Path(experiment["comparison_dir"]) / "summary.json")
            holdout_summary = summarize_holdout_metrics(Path(experiment["artifact_dir"]) / "metrics_holdout.json")
            status = classify_candidate(live_summary)
        candidate_rows.append(
            {
                "candidate_tag": experiment["candidate_tag"],
                "blocks": list(combo),
                "block_descriptions": [get_block(name).description for name in combo],
                "generated_contract_path": payload["_generated_contract_path"],
                "generated_config_path": payload["_generated_config_path"],
                "artifact_dir": str(experiment["artifact_dir"]),
                "comparison_dir": str(experiment["comparison_dir"]),
                "status": status,
                "live_summary": live_summary,
                "holdout_summary": holdout_summary,
                "results": results,
            }
        )

    candidate_rows.sort(
        key=lambda row: (
            STATUS_ORDER.get(str(row["status"]), 99),
            -(float((row.get("live_summary") or {}).get("delta_auc", -999.0))),
            float((row.get("live_summary") or {}).get("delta_brier", 999.0)),
            float((row.get("live_summary") or {}).get("delta_log_loss", 999.0)),
        )
    )
    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "search": _json_safe(dict(search)),
        "candidate_count": len(candidate_rows),
        "candidates": _json_safe(candidate_rows),
    }
    search_run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = search_run_dir / "search_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    leaderboard_path = search_run_dir / "leaderboard.md"
    leaderboard_path.write_text(build_leaderboard_markdown(candidate_rows), encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    payload = load_search_config(Path(args.config))
    search = resolve_search(payload)
    summary_path = run_search(
        search=search,
        python_bin=str(args.python_bin),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps({"summary_path": str(summary_path)}, indent=2))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if bool(args.dry_run):
        return 0
    statuses = [str(row.get("status", "failed")) for row in summary.get("candidates", [])]
    if any(status == "keep" for status in statuses):
        return 0
    return 1 if any(status == "failed" for status in statuses) else 0


if __name__ == "__main__":
    raise SystemExit(main())
