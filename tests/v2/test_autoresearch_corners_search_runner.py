from __future__ import annotations

import json
from pathlib import Path
import tempfile

from autoresearch_corners.interaction_library import managed_feature_names
from autoresearch_corners.search_runner import (
    build_candidate_payload,
    candidate_block_combinations,
    classify_candidate,
    parse_live_summary,
    resolve_search,
    run_search,
)
from src.modeling.v2.io.contracts import load_feature_contract


def _base_contract_text() -> str:
    managed = list(managed_feature_names())
    return "\n".join(
        [
            "family: corners",
            "required_features:",
            "  - home_rolling_corners",
            "  - away_rolling_corners",
            "optional_features:",
            "  - style_matchup_balanced_press_measured__balanced_press_measured",
            f"  - {managed[0]}",
            "missingness_indicators:",
            "  - home_recent_xg_mean_5_is_missing",
            "forbidden_cross_family_features:",
            "  - odds_over_15",
            "disabled_features: []",
            "",
        ]
    )


def _search_payload(tmp_path: Path) -> dict[str, object]:
    contract_path = tmp_path / "corners.yaml"
    contract_path.write_text(_base_contract_text(), encoding="utf-8")
    return {
        "candidate_prefix": "corners_interaction_search_v1",
        "dataset_path": "artifacts/v2/datasets/final_corners_feature_pass_20260310/pit_dataset_20260310T085757Z.csv",
        "base_contract_path": str(contract_path),
        "path_version": "totals_first",
        "scoreline_dir": "model_artifacts/v2/scoreline_v21_total_intensity_snap_20260308",
        "anytime_dir": "model_artifacts/v2/anytime_direct_monotone_v1_candidate_20260308",
        "artifact_root": str(tmp_path / "artifacts"),
        "comparison_root": str(tmp_path / "compare"),
        "results_dir": str(tmp_path / "results"),
        "block_names": ["style_matchup_possession", "style_matchup_box_touches"],
        "max_blocks_per_candidate": 2,
        "include_baseline": True,
    }


def test_candidate_block_combinations_includes_baseline_and_pairs() -> None:
    combos = candidate_block_combinations(
        ["style_matchup_possession", "style_matchup_box_touches"],
        max_blocks_per_candidate=2,
        include_baseline=True,
    )
    assert combos == [
        tuple(),
        ("style_matchup_possession",),
        ("style_matchup_box_touches",),
        ("style_matchup_possession", "style_matchup_box_touches"),
    ]


def test_build_candidate_payload_filters_managed_features_and_adds_selected() -> None:
    with tempfile.TemporaryDirectory(prefix="corners_search_payload_") as td:
        tmp_path = Path(td)
        search = resolve_search(_search_payload(tmp_path))
        payload = build_candidate_payload(
            search=search,
            block_combo=("style_matchup_box_touches",),
            generated_dir=tmp_path / "generated",
        )
        contract = load_feature_contract(Path(payload["_generated_contract_path"]))
        optional = list(contract.optional_features)
        assert "style_matchup_balanced_press_measured__balanced_press_measured" in optional
        assert not any(
            feature.endswith("__x__home_rolling_possession")
            for feature in optional
        )
        assert any(
            feature.endswith("__x__home_rolling_box_touches")
            for feature in optional
        )
        assert not any(
            feature.endswith("__x__home_rolling_sot_against")
            for feature in optional
        )


def test_build_candidate_payload_adds_new_away_tail_blocks() -> None:
    with tempfile.TemporaryDirectory(prefix="corners_search_payload_") as td:
        tmp_path = Path(td)
        search = resolve_search(_search_payload(tmp_path))
        payload = build_candidate_payload(
            search=search,
            block_combo=("style_matchup_sot_against",),
            generated_dir=tmp_path / "generated",
        )
        contract = load_feature_contract(Path(payload["_generated_contract_path"]))
        optional = list(contract.optional_features)
        assert any(
            feature.endswith("__x__home_rolling_sot_against")
            for feature in optional
        )
        assert any(
            feature.endswith("__x__away_rolling_sot_against")
            for feature in optional
        )


def test_build_candidate_payload_adds_total_tail_blocks() -> None:
    with tempfile.TemporaryDirectory(prefix="corners_search_payload_") as td:
        tmp_path = Path(td)
        search = resolve_search(_search_payload(tmp_path))
        payload = build_candidate_payload(
            search=search,
            block_combo=("style_matchup_crosses",),
            generated_dir=tmp_path / "generated",
        )
        contract = load_feature_contract(Path(payload["_generated_contract_path"]))
        optional = list(contract.optional_features)
        assert any(
            feature.endswith("__x__home_rolling_crosses")
            for feature in optional
        )
        assert any(
            feature.endswith("__x__away_rolling_crosses")
            for feature in optional
        )


def test_build_candidate_payload_keeps_required_anchor_blocks() -> None:
    with tempfile.TemporaryDirectory(prefix="corners_search_payload_") as td:
        tmp_path = Path(td)
        payload_config = _search_payload(tmp_path)
        payload_config["required_block_names"] = ["style_matchup_possession"]
        payload_config["block_names"] = ["style_matchup_box_touches"]
        search = resolve_search(payload_config)
        payload = build_candidate_payload(
            search=search,
            block_combo=("style_matchup_box_touches",),
            generated_dir=tmp_path / "generated",
        )
        contract = load_feature_contract(Path(payload["_generated_contract_path"]))
        optional = list(contract.optional_features)
        assert any(
            feature.endswith("__x__home_rolling_possession")
            for feature in optional
        )
        assert any(
            feature.endswith("__x__home_rolling_box_touches")
            for feature in optional
        )
        assert payload["candidate_tag"].endswith("__sm_poss__sm_bt")


def test_parse_live_summary_and_classify_candidate() -> None:
    with tempfile.TemporaryDirectory(prefix="corners_search_live_summary_") as td:
        path = Path(td) / "summary.json"
        path.write_text(
            json.dumps(
                {
                    "segment_summaries": [
                        {
                            "segment": "corners_overlap_nonfallback",
                            "rows": 123,
                            "delta": {
                                "auc": 0.01,
                                "brier": -0.002,
                                "log_loss": -0.003,
                                "ece": -0.001,
                            },
                            "market_win_counts": {"auc": 8, "brier": 9, "log_loss": 9},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        summary = parse_live_summary(path)
        assert summary is not None
        assert summary["segment"] == "corners_overlap_nonfallback"
        assert classify_candidate(summary) == "keep"


def test_run_search_dry_run_writes_search_summary() -> None:
    with tempfile.TemporaryDirectory(prefix="corners_search_dry_run_") as td:
        tmp_path = Path(td)
        search = resolve_search(_search_payload(tmp_path))
        summary_path = run_search(
            search=search,
            python_bin="python",
            dry_run=True,
        )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["dry_run"] is True
        assert payload["candidate_count"] == 4
        assert payload["candidates"][0]["candidate_tag"].startswith("corners_interaction_search_v1")
