from __future__ import annotations

from pathlib import Path
import tempfile

from src.modeling.v2.eval.promotion_registry import resolve_required_markets
from src.modeling.v2.io.promotion_policy import load_promotion_policy


def test_load_promotion_policy_reads_required_markets_and_dedupes() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_promotion_policy_") as td:
        path = Path(td) / "scoreline_core.yaml"
        path.write_text(
            "\n".join(
                [
                    "name: scoreline_core_v1",
                    "required_markets:",
                    "  - 1x2_h",
                    "  - 1x2_d",
                    "  - 1x2_h",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        policy = load_promotion_policy(path)
        assert policy.name == "scoreline_core_v1"
        assert policy.required_markets == ["1x2_h", "1x2_d"]


def test_resolve_required_markets_merges_cli_and_policy_markets() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_promotion_policy_merge_") as td:
        path = Path(td) / "scoreline_core.yaml"
        path.write_text(
            "\n".join(
                [
                    "name: scoreline_core_v1",
                    "required_markets:",
                    "  - 1x2_h",
                    "  - 1x2_d",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        markets, metadata = resolve_required_markets(
            required_markets=["o15", "1x2_h"],
            promotion_policy_path=path,
        )
        assert markets == ["o15", "1x2_h", "1x2_d"]
        assert metadata["promotion_policy_path"] == str(path)
        assert metadata["promotion_policy_name"] == "scoreline_core_v1"