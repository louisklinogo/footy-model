from __future__ import annotations

import json
from pathlib import Path
from typing import cast


ROOT_DIR = Path(__file__).resolve().parents[2]
LINCHPIN_MARKETS_PATH = (
    ROOT_DIR / "model_artifacts" / "market_models" / "linchpin_markets.json"
)


def load_linchpin_markets(path: Path | None = None) -> list[dict[str, object]]:
    contract_path = path or LINCHPIN_MARKETS_PATH
    with contract_path.open("r", encoding="utf-8") as handle:
        payload_obj = cast(object, json.load(handle))

    if not isinstance(payload_obj, list):
        raise ValueError("linchpin market contract must be a JSON list")
    payload = cast(list[object], payload_obj)

    seen_codes: set[str] = set()
    rows: list[dict[str, object]] = []
    for row_raw in payload:
        row_obj = row_raw
        if not isinstance(row_obj, dict):
            raise ValueError("each linchpin market row must be a JSON object")
        row = cast(dict[str, object], row_obj)
        code = row.get("market_code")
        if not isinstance(code, str) or not code:
            raise ValueError("each linchpin market row must include market_code")
        if code in seen_codes:
            raise ValueError(f"duplicate market_code in linchpin contract: {code}")
        seen_codes.add(code)
        rows.append(row)
    return rows


def get_linchpin_market_map(path: Path | None = None) -> dict[str, dict[str, object]]:
    rows = load_linchpin_markets(path=path)
    return {str(row["market_code"]): row for row in rows}


def get_linchpin_market_codes(path: Path | None = None) -> tuple[str, ...]:
    rows = load_linchpin_markets(path=path)
    return tuple(str(row["market_code"]) for row in rows)
