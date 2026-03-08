from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ARTIFACT_METADATA_FILE = "artifact_metadata.json"


def build_artifact_metadata(
    *,
    family: str,
    model_name: str,
    model_version: str,
    artifact_dir: Path,
    trained_at_utc: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "family": str(family),
        "model_name": str(model_name),
        "model_version": str(model_version).strip(),
        "artifact_dir": str(artifact_dir),
    }
    if trained_at_utc:
        payload["trained_at_utc"] = str(trained_at_utc)
    if extra:
        payload.update(extra)
    return payload


def write_artifact_metadata(artifact_dir: Path, payload: dict[str, Any]) -> Path:
    path = Path(artifact_dir) / ARTIFACT_METADATA_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_artifact_metadata(artifact_dir: Path) -> dict[str, Any]:
    path = Path(artifact_dir) / ARTIFACT_METADATA_FILE
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_model_identity(
    artifact_dir: Path,
    *,
    default_model_name: str,
    default_model_version: str,
    override_model_name: str | None = None,
    override_model_version: str | None = None,
) -> tuple[str, str]:
    metadata = load_artifact_metadata(artifact_dir)
    name_value = override_model_name or metadata.get("model_name") or default_model_name
    model_name = str(name_value).strip() or str(default_model_name)
    version_value = override_model_version or metadata.get("model_version") or default_model_version
    model_version = str(version_value).strip() or str(default_model_version)
    return model_name, model_version