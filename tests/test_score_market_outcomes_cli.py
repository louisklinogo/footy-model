from __future__ import annotations

import sys

from src.modeling.evaluation.score_market_outcomes_fixtures_first import parse_args


def test_parse_args_accepts_explicit_model_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_market_outcomes_fixtures_first.py",
            "--model",
            "market_outcome_v2",
            "--version",
            "hybrid_v1",
            "--league",
            "E0",
            "--since-days",
            "14",
            "--limit",
            "50",
        ],
    )
    args = parse_args()
    assert args.model == "market_outcome_v2"
    assert args.version == "hybrid_v1"
    assert args.league == "E0"
    assert args.since_days == 14
    assert args.limit == 50
