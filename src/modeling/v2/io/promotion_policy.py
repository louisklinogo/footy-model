from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotionPolicy:
    name: str
    description: str
    required_markets: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    out: dict[str, Any] = {}
    current_key: str | None = None
    current_list: list[str] | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped and not stripped.startswith("- "):
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            current_key = key
            if value:
                out[key] = value
                current_list = None
            else:
                out[key] = []
                current_list = out[key]
            continue
        if stripped.startswith("- ") and current_list is not None and current_key:
            current_list.append(stripped[2:].strip().strip("'").strip('"'))
    return out


def _normalize_markets(raw_markets: object) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_markets if isinstance(raw_markets, list) else []:
        market = str(raw).strip()
        if not market or market in seen:
            continue
        out.append(market)
        seen.add(market)
    return out


def load_promotion_policy(path: Path) -> PromotionPolicy:
    payload = _load_yaml(path)
    name = str(payload.get("name") or path.stem).strip()
    description = str(payload.get("description") or "").strip()
    required_markets = _normalize_markets(payload.get("required_markets"))
    if not required_markets:
        raise RuntimeError(f"Promotion policy must define at least one required_markets entry: {path}")
    return PromotionPolicy(
        name=name,
        description=description,
        required_markets=required_markets,
    )