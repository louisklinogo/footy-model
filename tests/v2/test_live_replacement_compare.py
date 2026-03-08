from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd

from src.modeling.v2.calibration.methods import fit_binary_calibrator
from src.modeling.v2.eval.live_replacement_compare import (
    build_live_replacement_report,
    load_family_holdout_predictions,
    parse_args,
    write_live_replacement_artifacts,
)


def test_parse_args_applies_calibrators_by_default_and_supports_opt_out(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["live_replacement_compare.py"])
    assert parse_args().apply_calibrators is True

    monkeypatch.setattr(
        sys,
        "argv",
        ["live_replacement_compare.py", "--no-apply-calibrators"],
    )
    assert parse_args().apply_calibrators is False


def test_load_family_holdout_predictions_applies_calibrators_and_records_metadata(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "scoreline_candidate"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"market": "o15", "fixture_id": 1, "p_model": 0.25, "y_true": 1, "league_code": "L1", "match_datetime_utc": "2026-03-01T12:00:00Z"},
            {"market": "o15", "fixture_id": 2, "p_model": 0.75, "y_true": 0, "league_code": "L1", "match_datetime_utc": "2026-03-02T12:00:00Z"},
        ]
    ).to_csv(artifact_dir / "holdout_predictions.csv", index=False)
    calibrator = fit_binary_calibrator(
        "sigmoid",
        p_model=np.array([0.2, 0.4, 0.6, 0.8]),
        y_true=np.array([0, 0, 1, 1]),
    )
    joblib.dump({"o15": calibrator}, artifact_dir / "calibrators.joblib")
    (artifact_dir / "artifact_metadata.json").write_text(
        json.dumps({"model_name": "scoreline_v2", "model_version": "candidate_v1"}),
        encoding="utf-8",
    )

    frame = load_family_holdout_predictions(
        {"scoreline": artifact_dir},
        family_markets={"mini_scoreline": {"o15"}},
        apply_calibrators=True,
        artifact_sources={"mini_scoreline": "scoreline"},
    )

    assert not np.allclose(frame["p_challenger"], frame["p_challenger_raw"])
    summary = frame.attrs["artifact_summary"]
    assert summary["challenger_apply_calibrators"] is True
    assert summary["families"]["mini_scoreline"]["resolved_artifact_dir"] == str(artifact_dir)
    assert summary["families"]["mini_scoreline"]["artifact_metadata"]["model_version"] == "candidate_v1"


def test_build_live_replacement_report_writes_fallback_segments_and_metadata(tmp_path) -> None:
    challenger = pd.DataFrame(
        [
            {"family": "mini_scoreline", "market_code": "o15", "fixture_id": 1, "p_challenger": 0.8, "p_challenger_raw": 0.75, "challenger_calibrated": True, "challenger_calibration_method": "sigmoid", "holdout_y_true": 1, "league_code": "L1", "match_datetime_utc": "2026-03-01T12:00:00Z"},
            {"family": "mini_scoreline", "market_code": "o15", "fixture_id": 2, "p_challenger": 0.2, "p_challenger_raw": 0.25, "challenger_calibrated": True, "challenger_calibration_method": "sigmoid", "holdout_y_true": 0, "league_code": "L1", "match_datetime_utc": "2026-03-02T12:00:00Z"},
        ]
    )
    challenger.attrs["artifact_summary"] = {
        "challenger_apply_calibrators": True,
        "families": {"mini_scoreline": {"resolved_artifact_dir": "artifact_dir"}},
    }
    live = pd.DataFrame(
        [
            {"fixture_id": 1, "market_code": "o15", "p_live": 0.7, "fallback_used": False, "fallback_reason": None, "prediction_model_family": "binary"},
            {"fixture_id": 2, "market_code": "o15", "p_live": 0.3, "fallback_used": True, "fallback_reason": "missing_artifact", "prediction_model_family": "binary"},
        ]
    )
    actuals = pd.DataFrame(
        [
            {"fixture_id": 1, "match_datetime_utc": "2026-03-01T12:00:00Z", "home_goals": 2, "away_goals": 0},
            {"fixture_id": 2, "match_datetime_utc": "2026-03-02T12:00:00Z", "home_goals": 0, "away_goals": 0},
        ]
    )

    report, per_market, matched_rows, league_family = build_live_replacement_report(
        challenger=challenger,
        live=live,
        actuals=actuals,
        family_markets={"mini_scoreline": {"o15"}},
    )

    assert report["artifact_summary"]["challenger_apply_calibrators"] is True
    assert report["cohort_summary"]["date_min"] == "2026-03-01 12:00:00+00:00"
    assert report["cohort_summary"]["date_max"] == "2026-03-02 12:00:00+00:00"
    fallback_segment = next(
        row for row in report["segment_summaries"] if row["segment"] == "full_overlap_fallback_only"
    )
    assert fallback_segment["rows"] == 1
    assert set(matched_rows["live_row_source"]) == {"live_model", "fallback"}

    artifacts = write_live_replacement_artifacts(
        output_dir=tmp_path / "out",
        report=report,
        per_market=per_market,
        matched_rows=matched_rows,
        league_family=league_family,
    )
    written = pd.read_csv(artifacts["matched_rows_path"])
    assert {"match_datetime_utc", "live_row_source"}.issubset(set(written.columns))