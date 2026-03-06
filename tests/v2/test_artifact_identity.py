from __future__ import annotations

from pathlib import Path
import tempfile

from src.modeling.v2.io.artifact_identity import (
    build_artifact_metadata,
    load_artifact_metadata,
    resolve_model_identity,
    write_artifact_metadata,
)


def test_artifact_metadata_round_trip_and_identity_resolution() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_artifact_identity_") as td:
        artifact_dir = Path(td) / "scoreline_candidate"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = build_artifact_metadata(
            family="scoreline",
            model_name="scoreline_v2",
            model_version="scoreline_dc_candidate_v1",
            artifact_dir=artifact_dir,
            trained_at_utc="2026-03-06T00:00:00+00:00",
            extra={"model_type_selected": "histgb_poisson"},
        )
        write_artifact_metadata(artifact_dir, payload)

        stored = load_artifact_metadata(artifact_dir)
        assert stored["family"] == "scoreline"
        assert stored["model_version"] == "scoreline_dc_candidate_v1"
        assert stored["artifact_dir"] == str(artifact_dir)

        model_name, model_version = resolve_model_identity(
            artifact_dir,
            default_model_name="scoreline_v2",
            default_model_version="poisson_head_v1",
        )
        assert model_name == "scoreline_v2"
        assert model_version == "scoreline_dc_candidate_v1"


def test_resolve_model_identity_falls_back_and_override_wins() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_artifact_identity_fallback_") as td:
        artifact_dir = Path(td) / "corners_candidate"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_name, model_version = resolve_model_identity(
            artifact_dir,
            default_model_name="corners_v2",
            default_model_version="distribution_head_v1",
        )
        assert model_name == "corners_v2"
        assert model_version == "distribution_head_v1"

        write_artifact_metadata(
            artifact_dir,
            build_artifact_metadata(
                family="corners",
                model_name="corners_v2",
                model_version="corners_live_scope_candidate_v1",
                artifact_dir=artifact_dir,
            ),
        )
        _, overridden = resolve_model_identity(
            artifact_dir,
            default_model_name="corners_v2",
            default_model_version="distribution_head_v1",
            override_model_version="corners_live_scope_candidate_shadow_v2",
        )
        assert overridden == "corners_live_scope_candidate_shadow_v2"