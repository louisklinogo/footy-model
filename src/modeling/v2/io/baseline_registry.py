from __future__ import annotations

import argparse
from functools import lru_cache
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_SCOPE_PATH = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_HANDICAP_CONTRACT_PATH = ROOT_DIR / "model_v2" / "handicap_contract.yaml"
DEFAULT_BASELINE_PATH = (
    ROOT_DIR / "model_artifacts" / "v2" / "baselines" / "metrics_baseline_v2.json"
)

_FALLBACK_LEGACY_CANONICAL_SELECTIONS = {
    "ah_h05": "ah2_home_m05",
    "ah_a05": "ah2_away_m05",
    "ah_h15": "ah2_home_m15",
    "ah_a15": "ah2_away_m15",
}
_FALLBACK_LEGACY_CANONICAL_PROXIES = {
    "eh_h1": "eh3_0_1_home",
    "eh_a1": "eh3_1_0_away",
}
_EVENT_EQUIVALENT_MARKETS = {
    "dc_x2": ("eh3_0_1_away",),
    "dc_1x": ("eh3_1_0_home",),
    "ah_h15": ("eh3_0_1_home",),
    "ah_a15": ("eh3_1_0_away",),
}
_EXPANSION_PRIORITY = {
    "source": 0,
    "legacy_canonical_selection": 1,
    "legacy_canonical_proxy": 1,
    "derived_ah_complement": 2,
    "event_equivalent_market": 3,
}


@dataclass(frozen=True)
class BaselineMetric:
    market_code: str
    auc: float | None
    brier: float | None
    pr_auc: float | None = None
    accuracy: float | None = None
    log_loss: float | None = None
    ece: float | None = None
    n: int | None = None


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


def _read_yaml_payload(path: Path) -> dict[str, Any] | None:
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


@lru_cache(maxsize=1)
def _load_handicap_compatibility_maps() -> tuple[dict[str, str], dict[str, str]]:
    canonical_selections = dict(_FALLBACK_LEGACY_CANONICAL_SELECTIONS)
    canonical_proxies = dict(_FALLBACK_LEGACY_CANONICAL_PROXIES)

    payload = _read_yaml_payload(DEFAULT_HANDICAP_CONTRACT_PATH)
    runtime_scope = (
        payload.get("legacy_repo_compatibility", {}).get("current_runtime_scope", {})
        if isinstance(payload, dict)
        else {}
    )
    if isinstance(runtime_scope, dict):
        for legacy_market, metadata in runtime_scope.items():
            if not isinstance(metadata, dict):
                continue
            canonical_selection = str(metadata.get("canonical_selection") or "").strip()
            if canonical_selection:
                canonical_selections[str(legacy_market).strip()] = canonical_selection
            canonical_proxy = str(metadata.get("canonical_proxy_for") or "").strip()
            if canonical_proxy:
                canonical_proxies[str(legacy_market).strip()] = canonical_proxy

    return canonical_selections, canonical_proxies


def _ah2_complement_market(market_code: str) -> str | None:
    parts = market_code.split("_")
    if len(parts) != 3 or parts[0] != "ah2":
        return None
    side = {"home": "away", "away": "home"}.get(parts[1])
    signed_line = parts[2]
    if side is None or len(signed_line) < 2 or signed_line[0] not in {"m", "p"}:
        return None
    magnitude = signed_line[1:]
    if not magnitude.isdigit():
        return None
    flipped = ("p" if signed_line[0] == "m" else "m") + magnitude
    return f"ah2_{side}_{flipped}"


def expand_baseline_market_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    canonical_selections, canonical_proxies = _load_handicap_compatibility_maps()
    expanded: dict[str, dict[str, Any]] = {}
    priorities: dict[str, int] = {}

    def register_market(
        *,
        target_market: str,
        source_market: str,
        note: str,
        row: dict[str, Any],
    ) -> None:
        priority = _EXPANSION_PRIORITY[note]
        existing_priority = priorities.get(target_market)
        if existing_priority is not None and existing_priority <= priority:
            return
        mapped = dict(row)
        mapped["market"] = target_market
        mapped["market_code"] = target_market
        mapped["source_market_code"] = source_market
        mapped["mapping_note"] = note
        expanded[target_market] = mapped
        priorities[target_market] = priority

    for row in rows:
        market = str(row.get("market") or row.get("market_code") or "").strip()
        if not market:
            continue

        register_market(target_market=market, source_market=market, note="source", row=row)

        canonical_selection = canonical_selections.get(market)
        if canonical_selection:
            register_market(
                target_market=canonical_selection,
                source_market=market,
                note="legacy_canonical_selection",
                row=row,
            )
            complement_market = _ah2_complement_market(canonical_selection)
            if complement_market:
                register_market(
                    target_market=complement_market,
                    source_market=market,
                    note="derived_ah_complement",
                    row=row,
                )

        canonical_proxy = canonical_proxies.get(market)
        if canonical_proxy:
            register_market(
                target_market=canonical_proxy,
                source_market=market,
                note="legacy_canonical_proxy",
                row=row,
            )

        for equivalent_market in _EVENT_EQUIVALENT_MARKETS.get(market, ()): 
            register_market(
                target_market=equivalent_market,
                source_market=market,
                note="event_equivalent_market",
                row=row,
            )

    return expanded


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

    expanded_rows = expand_baseline_market_rows(rows)

    out: dict[str, BaselineMetric] = {}
    for market_code, row in expanded_rows.items():
        if not market_code:
            continue
        out[market_code] = BaselineMetric(
            market_code=market_code,
            auc=_to_float_or_none(row.get("auc")),
            brier=_to_float_or_none(row.get("brier")),
            pr_auc=_to_float_or_none(row.get("pr_auc")),
            accuracy=_to_float_or_none(row.get("accuracy")),
            log_loss=_to_float_or_none(row.get("log_loss")),
            ece=_to_float_or_none(row.get("ece")),
            n=_to_int_or_none(row.get("n")),
        )
    return out


def missing_baseline_markets(
    *,
    scope_markets: list[str],
    baseline_metrics: dict[str, BaselineMetric],
) -> list[str]:
    return sorted({market for market in scope_markets if market not in baseline_metrics})


def missing_scope_markets(
    *,
    selected_markets: list[str],
    scope_markets: list[str],
) -> list[str]:
    scope_set = set(scope_markets)
    return sorted({market for market in selected_markets if market not in scope_set})


def validate_market_presence(
    *,
    market_list: list[str],
    baseline_metrics: dict[str, BaselineMetric],
    strict: bool,
    baseline_path: Path,
    market_set_name: str,
) -> list[str]:
    missing = sorted({market for market in market_list if market not in baseline_metrics})
    if missing and strict:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            f"Baseline registry missing {market_set_name} markets: "
            f"{missing_csv}. path={baseline_path}. "
            "Refresh the frozen baseline or fix the promotion policy / required-market list."
        )
    return missing


def validate_markets_in_scope(
    *,
    selected_markets: list[str],
    scope_markets: list[str],
    strict: bool,
    scope_path: Path,
    market_set_name: str,
) -> list[str]:
    missing = missing_scope_markets(
        selected_markets=selected_markets,
        scope_markets=scope_markets,
    )
    if missing and strict:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            f"{market_set_name.capitalize()} markets are not present in scope: "
            f"{missing_csv}. path={scope_path}. "
            "Fix the promotion policy / required-market list or update the scoped market contract first."
        )
    return missing


def validate_baseline_coverage(
    *,
    scope_markets: list[str],
    baseline_metrics: dict[str, BaselineMetric],
    strict: bool,
    baseline_path: Path,
) -> list[str]:
    missing = validate_market_presence(
        market_list=scope_markets,
        baseline_metrics=baseline_metrics,
        strict=strict,
        baseline_path=baseline_path,
        market_set_name="scope",
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

