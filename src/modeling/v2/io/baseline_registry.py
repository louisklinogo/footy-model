from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_SCOPE_PATH = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_BASELINE_PATH = (
    ROOT_DIR / "model_artifacts" / "v2" / "baselines" / "metrics_baseline_v2.json"
)


@dataclass(frozen=True)
class BaselineMetric:
    market_code: str
    auc: float | None
    brier: float | None


def _to_float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_scope_markets_fallback(scope_text: str) -> list[str]:
    # Fallback parser for the specific market_scope.yaml layout in this repo.
    markets: list[str] = []
    in_markets = False
    for raw in scope_text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "markets:":
            in_markets = True
            continue
        if in_markets and not line.startswith(" "):
            break
        if in_markets and stripped.startswith("- "):
            code = stripped[2:].strip().strip("'").strip('"')
            if code:
                markets.append(code)
    return markets


def load_scope_markets(scope_path: Path = DEFAULT_SCOPE_PATH) -> list[str]:
    text = scope_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if isinstance(payload, dict):
            markets = payload.get("markets")
            if isinstance(markets, list):
                out = [str(item).strip() for item in markets if str(item).strip()]
                if out:
                    return out
    except Exception:
        pass
    return _extract_scope_markets_fallback(text)


def load_baseline_metrics(
    baseline_path: Path = DEFAULT_BASELINE_PATH,
) -> dict[str, BaselineMetric]:
    if not baseline_path.exists():
        return {}
    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    rows: list[dict[str, Any]]
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict):
        metrics = payload.get("metrics")
        rows = [row for row in metrics if isinstance(row, dict)] if isinstance(metrics, list) else []
    else:
        return {}

    out: dict[str, BaselineMetric] = {}
    for row in rows:
        market_code = str(row.get("market_code") or row.get("market") or "").strip()
        if not market_code:
            continue
        out[market_code] = BaselineMetric(
            market_code=market_code,
            auc=_to_float_or_none(row.get("auc")),
            brier=_to_float_or_none(row.get("brier")),
        )
    return out


def missing_baseline_markets(
    *,
    scope_markets: list[str],
    baseline_metrics: dict[str, BaselineMetric],
) -> list[str]:
    return sorted({market for market in scope_markets if market not in baseline_metrics})


def validate_baseline_coverage(
    *,
    scope_markets: list[str],
    baseline_metrics: dict[str, BaselineMetric],
    strict: bool,
    baseline_path: Path,
) -> list[str]:
    missing = missing_baseline_markets(
        scope_markets=scope_markets, baseline_metrics=baseline_metrics
    )
    if missing and strict:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            "Baseline registry missing markets: "
            f"{missing_csv}. path={baseline_path}. "
            "Add missing rows or run with --allow-missing-baseline."
        )
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate v2 baseline registry coverage against market scope."
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=DEFAULT_SCOPE_PATH,
        help="Path to model_v2 market scope yaml.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Path to baseline metrics JSON.",
    )
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="Do not fail when scope markets are missing in baseline file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scope_markets = load_scope_markets(args.scope)
    if not scope_markets:
        raise RuntimeError(f"No markets found in scope file: {args.scope}")

    baselines = load_baseline_metrics(args.baseline)
    missing = validate_baseline_coverage(
        scope_markets=scope_markets,
        baseline_metrics=baselines,
        strict=not bool(args.allow_missing_baseline),
        baseline_path=args.baseline,
    )
    if missing:
        print("Missing baseline markets (non-strict mode):")
        for market in missing:
            print(f"- {market}")
    else:
        print("Baseline coverage valid for all scoped markets.")


if __name__ == "__main__":
    main()

