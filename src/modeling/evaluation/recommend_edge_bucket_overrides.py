from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_POLICY_PATH = ROOT_DIR / "model_artifacts" / "market_models" / "risk_policy.json"
DEFAULT_REPORT_DIR = ROOT_DIR / "artifacts" / "reports" / "risk_edge_buckets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recommend per-market edge-bucket gate overrides from edge-bucket reports."
        )
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Risk policy JSON path.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Edge bucket report JSON path. Defaults to latest file in --report-dir.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory used when --report-path is omitted.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output recommendation JSON path.",
    )
    parser.add_argument(
        "--roi-floor",
        type=float,
        default=0.0,
        help="Buckets below this mean ROI trigger tighter precision gates.",
    )
    parser.add_argument(
        "--precision-uplift",
        type=float,
        default=0.02,
        help="Min-precision uplift above observed precision for negative-ROI buckets.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Optional override for min bucket samples used in recommendations.",
    )
    parser.add_argument(
        "--max-min-precision",
        type=float,
        default=0.9,
        help="Upper cap for recommended min_precision.",
    )
    parser.add_argument(
        "--apply-policy",
        action="store_true",
        help="Write recommendations directly into --policy-path.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _latest_report_path(report_dir: Path) -> Path | None:
    if not report_dir.exists() or not report_dir.is_dir():
        return None
    files = sorted(report_dir.glob("edge_bucket_report_*.json"))
    if not files:
        return None
    return files[-1]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN guard
        return None
    return out


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _existing_override(
    policy: dict[str, Any], market_code: str, edge_bucket: str
) -> dict[str, Any]:
    overrides = policy.get("edge_bucket_overrides")
    if not isinstance(overrides, dict):
        return {}
    market_overrides = overrides.get(market_code)
    if not isinstance(market_overrides, dict):
        return {}
    bucket_override = market_overrides.get(edge_bucket)
    if not isinstance(bucket_override, dict):
        return {}
    return bucket_override


def _is_existing_equal_or_stricter(
    existing: dict[str, Any], proposed: dict[str, Any]
) -> bool:
    if not existing:
        return False

    existing_min_precision = _coerce_float(existing.get("min_precision"))
    proposed_min_precision = _coerce_float(proposed.get("min_precision"))
    if (
        proposed_min_precision is not None
        and existing_min_precision is not None
        and existing_min_precision + 1e-9 < proposed_min_precision
    ):
        return False

    existing_loss_pause = _coerce_int(existing.get("loss_streak_pause"))
    proposed_loss_pause = _coerce_int(proposed.get("loss_streak_pause"))
    if (
        proposed_loss_pause is not None
        and existing_loss_pause is not None
        and existing_loss_pause > proposed_loss_pause
    ):
        return False

    existing_min_samples = _coerce_int(existing.get("min_samples"))
    proposed_min_samples = _coerce_int(proposed.get("min_samples"))
    if (
        proposed_min_samples is not None
        and existing_min_samples is not None
        and existing_min_samples > proposed_min_samples
    ):
        return False

    if (
        bool(proposed.get("hard_precision_gate", True))
        and not bool(existing.get("hard_precision_gate", False))
    ):
        return False

    if (
        bool(proposed.get("hard_loss_streak_pause", True))
        and not bool(existing.get("hard_loss_streak_pause", False))
    ):
        return False

    return True


def recommend_overrides(
    *,
    policy: dict[str, Any],
    report: dict[str, Any],
    roi_floor: float,
    precision_uplift: float,
    min_samples_override: int | None,
    max_min_precision: float,
) -> list[dict[str, Any]]:
    base_min_samples = (
        int(min_samples_override)
        if min_samples_override is not None
        else int(policy.get("edge_bucket_min_samples", 60))
    )
    base_min_precision = float(policy.get("edge_bucket_min_precision", 0.54))
    base_loss_pause = int(policy.get("edge_bucket_loss_streak_pause", 3))
    hard_precision_gate = bool(policy.get("hard_edge_bucket_precision_gate", True))
    hard_loss_gate = bool(policy.get("hard_edge_bucket_loss_streak_pause", True))

    rows = report.get("edge_bucket_by_market")
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        market_code = str(row.get("market_code") or "").strip()
        edge_bucket = str(row.get("edge_bucket") or "").strip()
        if not market_code or not edge_bucket:
            continue

        n = _coerce_int(row.get("n"))
        precision = _coerce_float(row.get("precision"))
        loss_streak = _coerce_int(row.get("loss_streak"))
        mean_roi = _coerce_float(row.get("mean_roi_unit"))
        if n is None or n < base_min_samples:
            continue

        proposed: dict[str, Any] = {
            "min_samples": base_min_samples,
            "min_precision": base_min_precision,
            "loss_streak_pause": base_loss_pause,
            "hard_precision_gate": hard_precision_gate,
            "hard_loss_streak_pause": hard_loss_gate,
        }
        reasons: list[str] = []

        if mean_roi is not None and mean_roi < roi_floor and precision is not None:
            target_precision = min(max_min_precision, precision + precision_uplift)
            target_precision = round(target_precision, 4)
            if target_precision > proposed["min_precision"] + 1e-9:
                proposed["min_precision"] = target_precision
                reasons.append(
                    (
                        "negative_roi"
                        f"(roi={mean_roi:.4f}, precision={precision:.4f}, "
                        f"min_precision={target_precision:.4f})"
                    )
                )

        if loss_streak is not None and loss_streak >= base_loss_pause and base_loss_pause > 1:
            tighter_pause = max(1, base_loss_pause - 1)
            if tighter_pause < proposed["loss_streak_pause"]:
                proposed["loss_streak_pause"] = tighter_pause
                reasons.append(
                    (
                        "loss_streak"
                        f"(loss_streak={loss_streak}, "
                        f"loss_streak_pause={tighter_pause})"
                    )
                )

        if not reasons:
            continue

        existing = _existing_override(policy, market_code, edge_bucket)
        if _is_existing_equal_or_stricter(existing, proposed):
            continue

        out.append(
            {
                "market_code": market_code,
                "edge_bucket": edge_bucket,
                "n": n,
                "precision": precision,
                "loss_streak": loss_streak,
                "mean_roi_unit": mean_roi,
                "existing_override": existing,
                "proposed_override": proposed,
                "reasons": reasons,
            }
        )

    out.sort(key=lambda row: (row["market_code"], row["edge_bucket"]))
    return out


def apply_recommendations(
    policy: dict[str, Any], recommendations: list[dict[str, Any]]
) -> dict[str, Any]:
    updated = deepcopy(policy)
    overrides = updated.get("edge_bucket_overrides")
    if not isinstance(overrides, dict):
        overrides = {}
        updated["edge_bucket_overrides"] = overrides

    for rec in recommendations:
        market_code = str(rec["market_code"])
        edge_bucket = str(rec["edge_bucket"])
        market_overrides = overrides.get(market_code)
        if not isinstance(market_overrides, dict):
            market_overrides = {}
            overrides[market_code] = market_overrides
        market_overrides[edge_bucket] = dict(rec["proposed_override"])

    return updated


def main() -> None:
    args = parse_args()
    if not args.policy_path.exists():
        raise FileNotFoundError(f"Policy path not found: {args.policy_path}")

    report_path = args.report_path
    if report_path is None:
        report_path = _latest_report_path(args.report_dir)
        if report_path is None:
            raise FileNotFoundError(
                f"No edge bucket report found in directory: {args.report_dir}"
            )
    if not report_path.exists():
        raise FileNotFoundError(f"Report path not found: {report_path}")

    policy = _load_json(args.policy_path)
    report = _load_json(report_path)
    recommendations = recommend_overrides(
        policy=policy,
        report=report,
        roi_floor=float(args.roi_floor),
        precision_uplift=float(args.precision_uplift),
        min_samples_override=args.min_samples,
        max_min_precision=float(args.max_min_precision),
    )

    output = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "policy_path": str(args.policy_path),
        "report_path": str(report_path),
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
    }

    output_path = args.output_path
    if output_path is None:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = args.report_dir / f"edge_bucket_override_recommendations_{stamp}.json"
    _write_json(output_path, output)

    if args.apply_policy and recommendations:
        updated = apply_recommendations(policy, recommendations)
        _write_json(args.policy_path, updated)
        print(f"Applied {len(recommendations)} recommendations to {args.policy_path}")
    elif args.apply_policy:
        print("No recommendations to apply.")

    print(f"Report: {report_path}")
    print(f"Saved recommendations: {output_path}")
    print(f"recommendations={len(recommendations)}")


if __name__ == "__main__":
    main()
