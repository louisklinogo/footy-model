from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_RULE_LAYER_CONFIG: dict[str, Any] = {
    "config_version": "v2",
    "enabled": True,
    "max_down_pct": 0.15,
    "max_up_pct": 0.10,
    "upcoming_tier_2_pct": -0.05,
    "upcoming_tier_3_pct": -0.08,
    "key_absent_pct": -0.08,
    "congestion_threshold_games_14d": 3,
    "congestion_pct": -0.04,
    "rest_disadvantage_threshold_days": -2.5,
    "rest_advantage_threshold_days": 2.5,
    "rest_disadvantage_pct": -0.05,
    "rest_advantage_pct": 0.03,
    # Cross-league safety mode (for leagues with layer2 policy disabled).
    "safety_enabled_for_disabled_leagues": False,
    "safety_max_down_pct": 0.06,
    "safety_max_up_pct": 0.02,
    "safety_upcoming_tier_2_pct": -0.015,
    "safety_upcoming_tier_3_pct": -0.03,
    "safety_key_absent_pct": -0.04,
    "safety_congestion_threshold_games_14d": 3,
    "safety_congestion_pct": -0.015,
    "safety_rest_disadvantage_threshold_days": -2.5,
    "safety_rest_disadvantage_pct": -0.015,
    "safety_top4_pressure_pct": -0.012,
    "safety_table_adjacent_position_gap_max": 2,
    "safety_table_adjacent_points_gap_max": 6,
    "safety_table_adjacent_pct": -0.01,
    "safety_table_adjacent_only_if_weaker": True,
    "safety_require_odds_confirmation": True,
    "safety_allow_without_odds": False,
    "safety_odds_min_abs_gap": 0.0,
}


def load_rule_layer_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return dict(DEFAULT_RULE_LAYER_CONFIG)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_RULE_LAYER_CONFIG)
    if not isinstance(payload, dict):
        return dict(DEFAULT_RULE_LAYER_CONFIG)
    cfg = dict(DEFAULT_RULE_LAYER_CONFIG)
    cfg.update(payload)
    return cfg


def save_rule_layer_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        try:
            return row.get(key, default)
        except Exception:
            pass
    return getattr(row, key, default)


def _signed(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _safe_pct(value: Any) -> float:
    return _to_float(value, 0.0)


def _add_component(
    *,
    rules: list[str],
    components: list[dict[str, float]],
    rule_name: str,
    pct_value: Any,
) -> float:
    pct = _safe_pct(pct_value)
    if abs(pct) <= 1e-12:
        return 0.0
    rules.append(rule_name)
    components.append({"rule": rule_name, "pct": float(pct)})
    return pct


def _compute_caps(config: dict[str, Any], *, scope: str) -> tuple[float, float]:
    if scope == "disabled_league_safety":
        max_down = abs(_to_float(config.get("safety_max_down_pct"), 0.06))
        max_up = abs(_to_float(config.get("safety_max_up_pct"), 0.02))
        return max_down, max_up
    max_down = abs(_to_float(config.get("max_down_pct"), 0.15))
    max_up = abs(_to_float(config.get("max_up_pct"), 0.10))
    return max_down, max_up


def _compute_odds_gate(
    row: Any,
    *,
    side: str,
    pct_value: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    require_odds = bool(config.get("safety_require_odds_confirmation", True))
    allow_without_odds = bool(config.get("safety_allow_without_odds", False))
    min_abs_gap = abs(_to_float(config.get("safety_odds_min_abs_gap"), 0.0))
    odds_gap = _to_float(_row_get(row, f"odds_model_gap_{side}", float("nan")), float("nan"))
    has_gap = not math.isnan(odds_gap) and abs(odds_gap) > min_abs_gap

    odds_missing = not has_gap
    odds_conflict = False
    odds_confirmed = False
    odds_gate_blocked = False

    if abs(pct_value) <= 1e-12:
        return {
            "odds_gap": float(odds_gap) if not math.isnan(odds_gap) else None,
            "odds_has_gap": bool(has_gap),
            "odds_missing": bool(odds_missing),
            "odds_conflict": False,
            "odds_confirmed": bool(has_gap),
            "odds_gate_blocked": False,
        }

    if require_odds:
        if odds_missing and not allow_without_odds:
            odds_gate_blocked = True
        elif has_gap:
            odds_conflict = _signed(pct_value) != _signed(odds_gap)
            odds_confirmed = not odds_conflict
            if odds_conflict:
                odds_gate_blocked = True
    elif has_gap:
        odds_confirmed = _signed(pct_value) == _signed(odds_gap)
        odds_conflict = not odds_confirmed

    return {
        "odds_gap": float(odds_gap) if not math.isnan(odds_gap) else None,
        "odds_has_gap": bool(has_gap),
        "odds_missing": bool(odds_missing),
        "odds_conflict": bool(odds_conflict),
        "odds_confirmed": bool(odds_confirmed),
        "odds_gate_blocked": bool(odds_gate_blocked),
    }


def _global_rule_adjustment_for_side(row: Any, side: str, config: dict[str, Any]) -> dict[str, Any]:
    rules: list[str] = []
    components: list[dict[str, float]] = []
    raw_pct = 0.0

    tier = _to_int(_row_get(row, f"{side}_upcoming_tier", 0), 0)
    if tier >= 3:
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="upcoming_tier_3",
            pct_value=config.get("upcoming_tier_3_pct", -0.08),
        )
    elif tier >= 2:
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="upcoming_tier_2",
            pct_value=config.get("upcoming_tier_2_pct", -0.05),
        )

    key_absent = _to_int(_row_get(row, f"{side}_key_absent", 0), 0)
    if key_absent == 1:
        key_absent_pct = config.get(f"key_absent_pct_{side}", config.get("key_absent_pct"))
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="key_absent",
            pct_value=key_absent_pct if key_absent_pct is not None else -0.08,
        )

    side_recent = _to_int(_row_get(row, f"{side}_recent", 0), 0)
    cong_threshold = _to_int(config.get("congestion_threshold_games_14d"), 3)
    if side_recent >= cong_threshold:
        congestion_pct = config.get(f"congestion_pct_{side}", config.get("congestion_pct"))
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="congestion",
            pct_value=congestion_pct if congestion_pct is not None else -0.04,
        )

    rest_delta = _to_float(_row_get(row, "rest_delta", 0.0), 0.0)
    side_rest_delta = rest_delta if side == "home" else -rest_delta
    rest_down_thr = _to_float(config.get("rest_disadvantage_threshold_days"), -2.5)
    rest_up_thr = _to_float(config.get("rest_advantage_threshold_days"), 2.5)
    if side_rest_delta <= rest_down_thr:
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="rest_disadvantage",
            pct_value=config.get("rest_disadvantage_pct", -0.05),
        )
    elif side_rest_delta >= rest_up_thr:
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="rest_advantage",
            pct_value=config.get("rest_advantage_pct", 0.03),
        )

    max_down, max_up = _compute_caps(config, scope="enabled_league")
    capped_pct = max(-max_down, min(max_up, raw_pct))
    applied = bool(abs(capped_pct) > 1e-12)

    return {
        "mode": "global",
        "scope": "enabled_league",
        "rules": rules,
        "components": components,
        "pct_raw": float(raw_pct),
        "pct_capped": float(capped_pct),
        "pct_capped_pre_gate": float(capped_pct),
        "applied": applied,
        "cap_down": float(max_down),
        "cap_up": float(max_up),
        "odds_gap": None,
        "odds_has_gap": False,
        "odds_missing": False,
        "odds_confirmed": False,
        "odds_conflict": False,
        "odds_gate_blocked": False,
    }


def _is_weaker_side(row: Any, side: str) -> bool:
    points_gap = _to_float(_row_get(row, "points_gap", 0.0), 0.0)
    position_gap = _to_float(_row_get(row, "position_gap", 0.0), 0.0)
    # points_gap: home - away. position_gap: away_position - home_position.
    if side == "home":
        return points_gap < 0.0 or position_gap < 0.0
    return points_gap > 0.0 or position_gap > 0.0


def _safety_rule_adjustment_for_side(row: Any, side: str, config: dict[str, Any]) -> dict[str, Any]:
    rules: list[str] = []
    components: list[dict[str, float]] = []
    raw_pct = 0.0

    tier = _to_int(_row_get(row, f"{side}_upcoming_tier", 0), 0)
    if tier >= 3:
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="upcoming_tier_3",
            pct_value=config.get("safety_upcoming_tier_3_pct", -0.03),
        )
    elif tier >= 2:
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="upcoming_tier_2",
            pct_value=config.get("safety_upcoming_tier_2_pct", -0.015),
        )

    key_absent = _to_int(_row_get(row, f"{side}_key_absent", 0), 0)
    if key_absent == 1:
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="key_absent",
            pct_value=config.get("safety_key_absent_pct", -0.04),
        )

    side_recent = _to_int(_row_get(row, f"{side}_recent", 0), 0)
    cong_threshold = _to_int(config.get("safety_congestion_threshold_games_14d"), 3)
    if side_recent >= cong_threshold:
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="congestion",
            pct_value=config.get("safety_congestion_pct", -0.015),
        )

    rest_delta = _to_float(_row_get(row, "rest_delta", 0.0), 0.0)
    side_rest_delta = rest_delta if side == "home" else -rest_delta
    rest_down_thr = _to_float(config.get("safety_rest_disadvantage_threshold_days"), -2.5)
    if side_rest_delta <= rest_down_thr:
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="rest_disadvantage",
            pct_value=config.get("safety_rest_disadvantage_pct", -0.015),
        )

    top4_pressure = _to_int(_row_get(row, f"{side}_playing_top4", 0), 0)
    if top4_pressure == 1:
        raw_pct += _add_component(
            rules=rules,
            components=components,
            rule_name="top4_pressure",
            pct_value=config.get("safety_top4_pressure_pct", -0.012),
        )

    pos_gap_max = abs(_to_float(config.get("safety_table_adjacent_position_gap_max"), 2.0))
    pts_gap_max = abs(_to_float(config.get("safety_table_adjacent_points_gap_max"), 6.0))
    pos_gap_abs = abs(_to_float(_row_get(row, "position_gap", 0.0), 0.0))
    pts_gap_abs = abs(_to_float(_row_get(row, "points_gap", 0.0), 0.0))
    near_table_band = pos_gap_abs <= pos_gap_max and pts_gap_abs <= pts_gap_max
    if near_table_band:
        only_if_weaker = bool(config.get("safety_table_adjacent_only_if_weaker", True))
        if (not only_if_weaker) or _is_weaker_side(row, side):
            raw_pct += _add_component(
                rules=rules,
                components=components,
                rule_name="table_adjacent_volatility",
                pct_value=config.get("safety_table_adjacent_pct", -0.01),
            )

    max_down, max_up = _compute_caps(config, scope="disabled_league_safety")
    capped_pre_gate = max(-max_down, min(max_up, raw_pct))
    odds_gate = _compute_odds_gate(
        row=row,
        side=side,
        pct_value=capped_pre_gate,
        config=config,
    )
    capped_pct = 0.0 if odds_gate["odds_gate_blocked"] else capped_pre_gate
    applied = bool(abs(capped_pct) > 1e-12)

    return {
        "mode": "safety",
        "scope": "disabled_league_safety",
        "rules": rules,
        "components": components,
        "pct_raw": float(raw_pct),
        "pct_capped": float(capped_pct),
        "pct_capped_pre_gate": float(capped_pre_gate),
        "applied": applied,
        "cap_down": float(max_down),
        "cap_up": float(max_up),
        "odds_gap": odds_gate["odds_gap"],
        "odds_has_gap": bool(odds_gate["odds_has_gap"]),
        "odds_missing": bool(odds_gate["odds_missing"]),
        "odds_confirmed": bool(odds_gate["odds_confirmed"]),
        "odds_conflict": bool(odds_gate["odds_conflict"]),
        "odds_gate_blocked": bool(odds_gate["odds_gate_blocked"]),
    }


def rule_adjustment_for_side(
    row: Any,
    side: str,
    config: dict[str, Any],
    scope: str = "enabled_league",
) -> dict[str, Any]:
    if side not in {"home", "away"}:
        raise ValueError(f"Invalid side: {side}")

    normalized_scope = (scope or "enabled_league").strip().lower()
    if normalized_scope in {"enabled_league", "global", "default"}:
        return _global_rule_adjustment_for_side(row=row, side=side, config=config)
    if normalized_scope in {"disabled_league_safety", "safety"}:
        return _safety_rule_adjustment_for_side(row=row, side=side, config=config)
    raise ValueError(f"Invalid rule-layer scope: {scope}")


def apply_rule_adjustment(
    base_lambda: float,
    row: Any,
    side: str,
    config: dict[str, Any],
    scope: str = "enabled_league",
) -> dict[str, Any]:
    adj = rule_adjustment_for_side(row=row, side=side, config=config, scope=scope)
    adjusted = max(0.01, float(base_lambda) * (1.0 + float(adj["pct_capped"])))
    return {
        **adj,
        "lambda_before": float(base_lambda),
        "lambda_after": float(adjusted),
    }
