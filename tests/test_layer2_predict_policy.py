from __future__ import annotations

import json
import sys
from pathlib import Path
import shutil
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import pytest

from src.modeling.layer2_situational.deployment_policy import POLICY_FILENAME
from src.modeling.layer2_situational import predict_situational_residual


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.value, dtype=float)


def _workspace_tmp_dir(prefix: str) -> Path:
    base = Path(".pytest_tmp")
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"{prefix}{uuid4().hex}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def test_predict_main_applies_league_policy_alpha(monkeypatch) -> None:
    tmp_dir = _workspace_tmp_dir("predict_policy_")
    model_dir = tmp_dir / "situational_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_blob = {
        "features": ["f1"],
        "home_model": _ConstantModel(0.4),
        "away_model": _ConstantModel(-0.2),
    }
    joblib.dump(model_blob, model_dir / "situational_model.pkl")
    (model_dir / POLICY_FILENAME).write_text(
        json.dumps(
            {
                "defaults": {
                    "enabled": False,
                    "alpha": 0.0,
                    "reason": "default_disabled",
                },
                "leagues": {
                    "E0": {"enabled": True, "alpha": 0.5, "reason": "enabled"},
                    "E1": {"enabled": False, "alpha": 0.0, "reason": "cv_gate_failed"},
                },
            }
        ),
        encoding="utf-8",
    )

    fixture_df = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "league_code": "E0",
                "f1": 1.0,
                "lambda_home": 1.2,
                "lambda_away": 0.9,
                "home_played": 8,
                "away_played": 8,
            },
            {
                "fixture_id": 2,
                "league_code": "E1",
                "f1": 1.0,
                "lambda_home": 1.1,
                "lambda_away": 1.3,
                "home_played": 8,
                "away_played": 8,
            },
        ]
    )

    captured: list[dict] = []

    monkeypatch.setattr(predict_situational_residual, "MODEL_DIR", model_dir)
    monkeypatch.setattr(
        predict_situational_residual,
        "load_prediction_data",
        lambda days, league: fixture_df.copy(),
    )
    monkeypatch.setattr(
        predict_situational_residual,
        "save_residuals",
        lambda rows: captured.extend(rows),
    )
    monkeypatch.setattr(sys, "argv", ["predict_situational_residual.py"])

    try:
        predict_situational_residual.main()

        assert len(captured) == 8

        by_key = {(r["fixture_id"], r["market_code"]): r["meta"] for r in captured}

        e0_adj_home = by_key[(1, "adj_lambda_home")]
        assert e0_adj_home["lambda"] == pytest.approx(1.4)
        assert e0_adj_home["layer2_enabled"] is True
        assert e0_adj_home["layer2_alpha"] == 0.5
        assert e0_adj_home["residual_raw"] == 0.4
        assert e0_adj_home["residual_applied"] == pytest.approx(0.2)

        e1_adj_home = by_key[(2, "adj_lambda_home")]
        assert e1_adj_home["lambda"] == pytest.approx(1.1)
        assert e1_adj_home["layer2_enabled"] is False
        assert e1_adj_home["layer2_alpha"] == 0.0
        assert e1_adj_home["residual_raw"] == 0.4
        assert e1_adj_home["residual_applied"] == 0.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
