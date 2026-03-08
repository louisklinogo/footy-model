from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.v2.io.baseline_registry import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_SCOPE_PATH,
    expand_baseline_market_rows,
    load_scope_markets,
)


DEFAULT_SCORELINE = ROOT_DIR / "model_artifacts" / "v2" / "scoreline" / "metrics_holdout.json"
DEFAULT_CORNERS = ROOT_DIR / "model_artifacts" / "v2" / "corners" / "metrics_holdout.json"
DEFAULT_ANYTIME = ROOT_DIR / "model_artifacts" / "v2" / "anytime" / "metrics_holdout.json"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _to_float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild v2 baseline metrics from latest family holdout artifacts."
    )
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE_PATH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--scoreline", type=Path, default=DEFAULT_SCORELINE)
    parser.add_argument("--corners", type=Path, default=DEFAULT_CORNERS)
    parser.add_argument("--anytime", type=Path, default=DEFAULT_ANYTIME)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow writing nulls when a scoped market is missing from holdout artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scope_markets = load_scope_markets(args.scope)
    if not scope_markets:
        raise RuntimeError(f"No scope markets found in {args.scope}")

    combined_rows = []
    combined_rows.extend(_read_rows(args.scoreline))
    combined_rows.extend(_read_rows(args.corners))
    combined_rows.extend(_read_rows(args.anytime))

    by_market = expand_baseline_market_rows(combined_rows)

    missing = [market for market in scope_markets if market not in by_market]
    if missing and not args.allow_missing:
        raise RuntimeError(
            "Missing scoped markets in holdout artifacts: "
            + ", ".join(sorted(missing))
            + ". Re-run family trainers or pass --allow-missing."
        )

    metrics = []
    for market in scope_markets:
        row = by_market.get(market, {})
        metric_row = {
            "market_code": market,
            "auc": _to_float_or_none(row.get("auc")),
            "pr_auc": _to_float_or_none(row.get("pr_auc")),
            "accuracy": _to_float_or_none(row.get("accuracy")),
            "brier": _to_float_or_none(row.get("brier")),
            "log_loss": _to_float_or_none(row.get("log_loss")),
            "ece": _to_float_or_none(row.get("ece")),
            "n": _to_int_or_none(row.get("n")),
        }
        source_market = str(row.get("source_market_code") or "").strip()
        mapping_note = str(row.get("mapping_note") or "").strip()
        if source_market and source_market != market:
            metric_row["source_market_code"] = source_market
        if mapping_note and mapping_note != "source":
            metric_row["mapping_note"] = mapping_note
        metrics.append(metric_row)

    payload = {
        "version": 2,
        "description": "Baseline comparator registry for v2 market promotion.",
        "updated_at_utc": datetime.now(tz=UTC).isoformat(),
        "sources": {
            "scoreline_holdout": str(args.scoreline),
            "scoreline_holdout_path": str(args.scoreline),
            "corners_holdout": str(args.corners),
            "corners_holdout_path": str(args.corners),
            "anytime_holdout": str(args.anytime),
            "anytime_holdout_path": str(args.anytime),
            "scope": str(args.scope),
        },
        "metrics": metrics,
    }

    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    args.baseline.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved baseline metrics: {args.baseline}")
    print(f"Scoped markets: {len(scope_markets)}")
    print(f"Missing in artifacts: {len(missing)}")


if __name__ == "__main__":
    main()
