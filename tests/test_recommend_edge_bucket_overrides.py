from __future__ import annotations

from src.modeling.evaluation.recommend_edge_bucket_overrides import (
    apply_recommendations,
    recommend_overrides,
)


def test_recommend_overrides_for_negative_roi_bucket() -> None:
    policy = {
        "edge_bucket_min_samples": 60,
        "edge_bucket_min_precision": 0.54,
        "edge_bucket_loss_streak_pause": 3,
        "hard_edge_bucket_precision_gate": True,
        "hard_edge_bucket_loss_streak_pause": True,
    }
    report = {
        "edge_bucket_by_market": [
            {
                "market_code": "o15",
                "edge_bucket": "4-6%",
                "n": 60,
                "precision": 0.70,
                "loss_streak": 1,
                "mean_roi_unit": -0.0457,
            }
        ]
    }

    recs = recommend_overrides(
        policy=policy,
        report=report,
        roi_floor=0.0,
        precision_uplift=0.02,
        min_samples_override=None,
        max_min_precision=0.9,
    )

    assert len(recs) == 1
    rec = recs[0]
    assert rec["market_code"] == "o15"
    assert rec["edge_bucket"] == "4-6%"
    assert float(rec["proposed_override"]["min_precision"]) == 0.72
    assert int(rec["proposed_override"]["loss_streak_pause"]) == 3


def test_recommend_overrides_skips_when_existing_is_stricter() -> None:
    policy = {
        "edge_bucket_min_samples": 60,
        "edge_bucket_min_precision": 0.54,
        "edge_bucket_loss_streak_pause": 3,
        "hard_edge_bucket_precision_gate": True,
        "hard_edge_bucket_loss_streak_pause": True,
        "edge_bucket_overrides": {
            "o15": {
                "4-6%": {
                    "min_samples": 60,
                    "min_precision": 0.75,
                    "loss_streak_pause": 2,
                    "hard_precision_gate": True,
                    "hard_loss_streak_pause": True,
                }
            }
        },
    }
    report = {
        "edge_bucket_by_market": [
            {
                "market_code": "o15",
                "edge_bucket": "4-6%",
                "n": 60,
                "precision": 0.70,
                "loss_streak": 1,
                "mean_roi_unit": -0.02,
            }
        ]
    }

    recs = recommend_overrides(
        policy=policy,
        report=report,
        roi_floor=0.0,
        precision_uplift=0.02,
        min_samples_override=None,
        max_min_precision=0.9,
    )

    assert recs == []


def test_apply_recommendations_merges_policy_structure() -> None:
    policy = {
        "edge_bucket_min_samples": 60,
        "edge_bucket_min_precision": 0.54,
        "edge_bucket_loss_streak_pause": 3,
    }
    recs = [
        {
            "market_code": "o15",
            "edge_bucket": "4-6%",
            "proposed_override": {
                "min_samples": 60,
                "min_precision": 0.72,
                "loss_streak_pause": 2,
                "hard_precision_gate": True,
                "hard_loss_streak_pause": True,
            },
        }
    ]

    updated = apply_recommendations(policy, recs)
    overrides = updated["edge_bucket_overrides"]
    assert float(overrides["o15"]["4-6%"]["min_precision"]) == 0.72
    assert int(overrides["o15"]["4-6%"]["loss_streak_pause"]) == 2
