from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RULE_LAYER_CONFIG: dict[str, Any] = {
    "config_version": "v1",
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


def rule_adjustment_for_side(row: Any, side: str, config: dict[str, Any]) -> dict[str, Any]:
    if side not in {"home", "away"}:
        raise ValueError(f"Invalid side: {side}")

    rules: list[str] = []
    raw_pct = 0.0

    tier = _to_int(_row_get(row, f"{side}_upcoming_tier", 0), 0)
    if tier >= 3:
        raw_pct += _to_float(config.get("upcoming_tier_3_pct"), -0.08)
        rules.append("upcoming_tier_3")
    elif tier >= 2:
        raw_pct += _to_float(config.get("upcoming_tier_2_pct"), -0.05)
        rules.append("upcoming_tier_2")

    key_absent = _to_int(_row_get(row, f"{side}_key_absent", 0), 0)
    if key_absent == 1:
        key_absent_pct = config.get(f"key_absent_pct_{side}", config.get("key_absent_pct"))
        raw_pct += _to_float(key_absent_pct, -0.08)
        rules.append("key_absent")

    side_recent = _to_int(_row_get(row, f"{side}_recent", 0), 0)
    cong_threshold = _to_int(config.get("congestion_threshold_games_14d"), 3)
    if side_recent >= cong_threshold:
        congestion_pct = config.get(f"congestion_pct_{side}", config.get("congestion_pct"))
        raw_pct += _to_float(congestion_pct, -0.04)
        rules.append("congestion")

    rest_delta = _to_float(_row_get(row, "rest_delta", 0.0), 0.0)
    side_rest_delta = rest_delta if side == "home" else -rest_delta
    rest_down_thr = _to_float(config.get("rest_disadvantage_threshold_days"), -2.5)
    rest_up_thr = _to_float(config.get("rest_advantage_threshold_days"), 2.5)
    if side_rest_delta <= rest_down_thr:
        raw_pct += _to_float(config.get("rest_disadvantage_pct"), -0.05)
        rules.append("rest_disadvantage")
    elif side_rest_delta >= rest_up_thr:
        raw_pct += _to_float(config.get("rest_advantage_pct"), 0.03)
        rules.append("rest_advantage")

    max_down = abs(_to_float(config.get("max_down_pct"), 0.15))
    max_up = abs(_to_float(config.get("max_up_pct"), 0.10))
    capped_pct = max(-max_down, min(max_up, raw_pct))

    applied = bool(abs(capped_pct) > 1e-12)
    return {
        "rules": rules,
        "pct_raw": float(raw_pct),
        "pct_capped": float(capped_pct),
        "applied": applied,
        "cap_down": float(max_down),
        "cap_up": float(max_up),
    }


def apply_rule_adjustment(
    base_lambda: float, row: Any, side: str, config: dict[str, Any]
) -> dict[str, Any]:
    adj = rule_adjustment_for_side(row, side, config)
    adjusted = max(0.01, float(base_lambda) * (1.0 + float(adj["pct_capped"])))
    return {
        **adj,
        "lambda_before": float(base_lambda),
        "lambda_after": float(adjusted),
    }
