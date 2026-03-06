from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd

from src.modeling.v2.calibration.run_calibration import build_calibration_report


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_market_predictions(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for idx in range(1, 241):
        rows.append(
            {
                "market": "o15",
                "fixture_id": idx,
                "match_datetime_utc": f"2026-01-{(idx % 28) + 1:02d}T00:00:00+00:00",
                "y_true": 0 if idx % 4 in {0, 1} else 1,
                "p_model": 0.4 if idx % 4 in {0, 1} else 0.6,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_calibration_report_writes_summary_and_family_artifacts() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_calibration_runner_") as td:
        root = Path(td)
        scoreline_dir = root / "scoreline"
        anytime_dir = root / "anytime"
        corners_dir = root / "corners"
        scoreline_dir.mkdir(parents=True, exist_ok=True)
        anytime_dir.mkdir(parents=True, exist_ok=True)
        corners_dir.mkdir(parents=True, exist_ok=True)
        _write_market_predictions(scoreline_dir / "holdout_predictions.csv")
        pd.DataFrame(columns=["market", "fixture_id", "y_true", "p_model"]).to_csv(
            anytime_dir / "holdout_predictions.csv", index=False
        )
        evaluation_report_path = root / "evaluation_report.json"
        _write_json(
            evaluation_report_path,
            {
                "families": {
                    "scoreline": {"artifact_dir": str(scoreline_dir)},
                    "anytime": {"artifact_dir": str(anytime_dir)},
                    "corners": {"artifact_dir": str(corners_dir)},
                }
            },
        )

        report = build_calibration_report(
            evaluation_report_path=evaluation_report_path,
            output_dir=root / "evaluation",
            fit_fraction=0.5,
            min_fit_rows=80,
            min_eval_rows=80,
            max_auc_drop=0.01,
        )

        assert report["families"]["scoreline"]["attempted_markets"] == 1
        assert report["families"]["scoreline"]["calibrated_markets"] == 1
        assert Path(report["families"]["scoreline"]["calibration_report_path"]).exists()
        assert Path(report["families"]["scoreline"]["calibrators_path"]).exists()
        assert report["promotion_holdout_by_market"]["o15"]["calibration_method"] in {"sigmoid", "isotonic"}