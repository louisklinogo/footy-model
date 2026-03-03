from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class OddsResolution:
    odds_used: float | None
    provider: str | None
    snapshot_type: str | None
    snapshot_time_utc: datetime | None
    line_num: float | None
    odds_field: str | None
    fallback_used: bool | str


@dataclass(frozen=True)
class _Selection:
    market_codes: tuple[str, ...]
    side: str
    line_num: float | None


@dataclass(frozen=True)
class _OddsRow:
    provider: str | None
    snapshot_type: str | None
    snapshot_time_utc: datetime | None
    market_code: str | None
    line_num: float | None
    prices: dict[str, Any]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _extract_numeric(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is not None:
        return parsed
    if isinstance(value, dict):
        for key in ("odds", "price", "value", "decimal"):
            maybe = _safe_float(value.get(key))
            if maybe is not None:
                return maybe
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_SIDE_KEYS: dict[str, tuple[str, ...]] = {
    "home": ("home", "h", "1", "Home"),
    "draw": ("draw", "d", "x", "X", "Draw"),
    "away": ("away", "a", "2", "Away"),
    "home_draw": ("home_draw", "1x", "1X", "homeOrDraw"),
    "draw_away": ("draw_away", "x2", "X2", "drawOrAway"),
    "home_away": ("home_away", "12", "homeOrAway"),
    "over": ("over", "o", "Over"),
    "under": ("under", "u", "Under"),
    "yes": ("yes", "y", "Yes"),
    "no": ("no", "n", "No"),
}


_MARKET_MAP: dict[str, tuple[tuple[_Selection, ...], tuple[_Selection, ...]]] = {
    "1x2_h": ((_Selection(("1x2",), "home", None),), ()),
    "1x2_d": ((_Selection(("1x2",), "draw", None),), ()),
    "1x2_a": ((_Selection(("1x2",), "away", None),), ()),
    "dc_1x": ((_Selection(("dc",), "home_draw", None),), ()),
    "dc_x2": ((_Selection(("dc",), "draw_away", None),), ()),
    "dc_12": ((_Selection(("dc",), "home_away", None),), ()),
    "btts": ((_Selection(("btts",), "yes", None),), ()),
    "o15": ((_Selection(("ou",), "over", 1.5),), ()),
    "u15": ((_Selection(("ou",), "under", 1.5),), ()),
    "o25": ((_Selection(("ou",), "over", 2.5),), ()),
    "u25": ((_Selection(("ou",), "under", 2.5),), ()),
    "o35": ((_Selection(("ou",), "over", 3.5),), ()),
    "u35": ((_Selection(("ou",), "under", 3.5),), ()),
    "o45": ((_Selection(("ou",), "over", 4.5),), ()),
    "u45": ((_Selection(("ou",), "under", 4.5),), ()),
    "ho15": ((_Selection(("home_ou",), "over", 1.5),), ()),
    "ao15": ((_Selection(("away_ou",), "over", 1.5),), ()),
    "c75": ((_Selection(("corners_ou",), "over", 7.5),), ()),
    "c85": ((_Selection(("corners_ou",), "over", 8.5),), ()),
    "c95": ((_Selection(("corners_ou",), "over", 9.5),), ()),
    "c105": ((_Selection(("corners_ou",), "over", 10.5),), ()),
    "hc25": (
        (
            _Selection(
                ("home_corners_ou", "corners_home_ou", "team_corners_home_ou"),
                "over",
                2.5,
            ),
        ),
        (),
    ),
    "hc35": (
        (
            _Selection(
                ("home_corners_ou", "corners_home_ou", "team_corners_home_ou"),
                "over",
                3.5,
            ),
        ),
        (),
    ),
    "hc45": (
        (
            _Selection(
                ("home_corners_ou", "corners_home_ou", "team_corners_home_ou"),
                "over",
                4.5,
            ),
        ),
        (),
    ),
    "hc55": (
        (
            _Selection(
                ("home_corners_ou", "corners_home_ou", "team_corners_home_ou"),
                "over",
                5.5,
            ),
        ),
        (),
    ),
    "ac25": (
        (
            _Selection(
                ("away_corners_ou", "corners_away_ou", "team_corners_away_ou"),
                "over",
                2.5,
            ),
        ),
        (),
    ),
    "ac35": (
        (
            _Selection(
                ("away_corners_ou", "corners_away_ou", "team_corners_away_ou"),
                "over",
                3.5,
            ),
        ),
        (),
    ),
    "ac45": (
        (
            _Selection(
                ("away_corners_ou", "corners_away_ou", "team_corners_away_ou"),
                "over",
                4.5,
            ),
        ),
        (),
    ),
    "ac55": (
        (
            _Selection(
                ("away_corners_ou", "corners_away_ou", "team_corners_away_ou"),
                "over",
                5.5,
            ),
        ),
        (),
    ),
    "ah_h05": (
        (_Selection(("ah",), "home", -0.5),),
        (_Selection(("ah",), "home", 0.5),),
    ),
    "ah_a05": (
        (_Selection(("ah",), "away", 0.5),),
        (_Selection(("ah",), "away", -0.5),),
    ),
    "ah_h15": (
        (_Selection(("ah",), "home", -1.5),),
        (_Selection(("ah",), "home", 1.5),),
    ),
    "ah_a15": (
        (_Selection(("ah",), "away", 1.5),),
        (_Selection(("ah",), "away", -1.5),),
    ),
    "eh_h1": (
        (_Selection(("eh",), "home", -1.0),),
        (_Selection(("ah",), "home", -1.5), _Selection(("ah",), "home", 1.5)),
    ),
    "eh_a1": (
        (_Selection(("eh",), "away", 1.0),),
        (_Selection(("ah",), "away", 1.5), _Selection(("ah",), "away", -1.5)),
    ),
}


def _normalize_row(row: dict[str, Any]) -> _OddsRow:
    payload = _as_dict(row.get("odds_json"))
    if not payload:
        payload = _as_dict(row)
    prices = _as_dict(payload.get("prices_latest"))
    if not prices:
        prices = _as_dict(row.get("prices_latest"))
    provider = row.get("provider") or payload.get("provider")
    snapshot_type = row.get("snapshot_type") or payload.get("snapshot_type")
    snapshot_time_utc = _parse_datetime(
        row.get("snapshot_time_utc") or payload.get("snapshot_time_utc")
    )
    market_code = row.get("market_code") or payload.get("market_code")
    line_num = _safe_float(row.get("line_num"))
    if line_num is None:
        line_num = _safe_float(payload.get("line_num"))
    return _OddsRow(
        provider=str(provider) if provider is not None else None,
        snapshot_type=str(snapshot_type) if snapshot_type is not None else None,
        snapshot_time_utc=snapshot_time_utc,
        market_code=str(market_code) if market_code is not None else None,
        line_num=line_num,
        prices=prices,
    )


def _line_matches(row_line: float | None, target_line: float | None) -> bool:
    if target_line is None:
        return True
    if row_line is None:
        return False
    return abs(row_line - target_line) <= 0.01


def _extract_price(
    prices: dict[str, Any], side: str
) -> tuple[float | None, str | None]:
    candidates = _SIDE_KEYS.get(side, (side,))
    for key in candidates:
        if key not in prices:
            continue
        value = _extract_numeric(prices.get(key))
        if value is not None and value > 1.0:
            return value, key
    return None, None


def _pick_row(
    rows: list[_OddsRow],
    selection: _Selection,
    snapshot_type: str,
) -> tuple[_OddsRow | None, float | None, str | None]:
    candidates: list[tuple[_OddsRow, float, str]] = []
    market_targets = set(selection.market_codes)
    for row in rows:
        if row.snapshot_type != snapshot_type:
            continue
        if row.market_code not in market_targets:
            continue
        if not _line_matches(row.line_num, selection.line_num):
            continue
        price, field = _extract_price(row.prices, selection.side)
        if price is None or field is None:
            continue
        candidates.append((row, price, field))
    if not candidates:
        return None, None, None
    candidates.sort(
        key=lambda item: (
            item[0].snapshot_time_utc or datetime.min.replace(tzinfo=timezone.utc)
        ),
        reverse=True,
    )
    picked_row, picked_price, picked_field = candidates[0]
    return picked_row, picked_price, picked_field


def resolve_odds_for_market(
    fixture_kickoff_utc: datetime | str,
    odds_rows: list[dict[str, Any]],
    market_code: str,
) -> OddsResolution:
    normalized_code = market_code[2:] if market_code.startswith("p_") else market_code
    config = _MARKET_MAP.get(normalized_code)
    kickoff_dt = _parse_datetime(fixture_kickoff_utc)
    if config is None or kickoff_dt is None:
        return OddsResolution(None, None, None, None, None, None, False)

    normalized_rows = [_normalize_row(row) for row in odds_rows]
    eligible = [
        row
        for row in normalized_rows
        if row.snapshot_time_utc is not None and row.snapshot_time_utc <= kickoff_dt
    ]
    if not eligible:
        return OddsResolution(None, None, None, None, None, None, False)

    primary, fallback = config
    snapshot_types = ["latest_pre_match", "closing"]
    additional = sorted(
        {
            row.snapshot_type
            for row in eligible
            if row.snapshot_type
            and row.snapshot_type not in {"latest_pre_match", "closing"}
        }
    )
    snapshot_types.extend(additional)

    for snapshot_type in snapshot_types:
        for selection in primary:
            row, price, field = _pick_row(eligible, selection, snapshot_type)
            if row is None or price is None:
                continue
            fallback_trace: bool | str = False
            if snapshot_type != "latest_pre_match":
                fallback_trace = f"snapshot_type:{snapshot_type}"
            return OddsResolution(
                odds_used=price,
                provider=row.provider,
                snapshot_type=row.snapshot_type,
                snapshot_time_utc=row.snapshot_time_utc,
                line_num=row.line_num,
                odds_field=field,
                fallback_used=fallback_trace,
            )

        for selection in fallback:
            row, price, field = _pick_row(eligible, selection, snapshot_type)
            if row is None or price is None:
                continue
            traces = [f"selection_fallback:{normalized_code}"]
            if snapshot_type != "latest_pre_match":
                traces.append(f"snapshot_type:{snapshot_type}")
            return OddsResolution(
                odds_used=price,
                provider=row.provider,
                snapshot_type=row.snapshot_type,
                snapshot_time_utc=row.snapshot_time_utc,
                line_num=row.line_num,
                odds_field=field,
                fallback_used=";".join(traces),
            )

    return OddsResolution(None, None, None, None, None, None, False)
