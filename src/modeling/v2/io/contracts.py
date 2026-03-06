from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureContract:
    family: str
    required_features: list[str]
    optional_features: list[str]
    missingness_indicators: list[str]
    forbidden_cross_family_features: list[str]
    disabled_features: list[str]


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


def load_feature_contract(path: Path) -> FeatureContract:
    payload = _load_yaml(path)
    family = str(payload.get("family") or path.stem).strip()
    required = [str(x).strip() for x in payload.get("required_features", []) if str(x).strip()]
    optional = [str(x).strip() for x in payload.get("optional_features", []) if str(x).strip()]
    missing = [str(x).strip() for x in payload.get("missingness_indicators", []) if str(x).strip()]
    forbidden = [
        str(x).strip()
        for x in payload.get("forbidden_cross_family_features", [])
        if str(x).strip()
    ]
    disabled = [
        str(x).strip() for x in payload.get("disabled_features", []) if str(x).strip()
    ]
    return FeatureContract(
        family=family,
        required_features=required,
        optional_features=optional,
        missingness_indicators=missing,
        forbidden_cross_family_features=forbidden,
        disabled_features=disabled,
    )
