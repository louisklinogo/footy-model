from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.v2.calibration.methods import apply_binary_calibrator
from src.modeling.evaluation.predict_market_outcomes_fixtures_first import (
    add_derived_features,
    fetch_candidate_fixtures,
)
from src.modeling.v2.families.corners.derive_lines import (
    AWAY_MARKETS,
    derive_and_validate_corners,
    HOME_MARKETS,
    over_probability,
    reconcile_team_means_from_total_home_mu,
    reconcile_team_means_from_total_share,
    TOTAL_MARKETS,
)
from src.modeling.v2.families.corners.features import (
    add_corners_context_features,
    apply_league_regime,
)
from src.modeling.v2.families.corners.neural_residual import (
    NEURAL_TOTAL_RESIDUAL_BUNDLE_FILENAME,
    NEURAL_TOTAL_SHARE_RESIDUAL_PATH_VERSION,
    NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME,
    NEURAL_SHARE_RESIDUAL_PATH_VERSION,
    load_neural_total_residual_bundle,
    load_neural_share_residual_bundle,
    predict_neural_total_residual_delta,
    predict_neural_share_residual_delta,
)
from src.modeling.v2.families.corners.neural_total_ladder import (
    NEURAL_TOTAL_MARKET_LADDER_BUNDLE_FILENAME,
    NEURAL_TOTAL_MARKET_LADDER_PATH_VERSION,
    load_neural_total_market_ladder_bundle,
    predict_neural_total_market_ladder_probs,
)
from src.modeling.v2.io.artifact_identity import resolve_model_identity
from src.modeling.v2.io.baseline_registry import load_scope_markets


DEFAULT_ARTIFACT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "corners"
DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "artifacts" / "v2" / "predictions"
MODEL_NAME = "corners_v2"
MODEL_VERSION = "distribution_head_v1"
TOTAL_SURFACE_PATHS = {"totals_surface_calibrated"}
NEURAL_TOTAL_LADDER_PATHS = {NEURAL_TOTAL_MARKET_LADDER_PATH_VERSION}
PMF_SURFACE_PATHS = {"pmf_surface_blended", "pmf_surface_blended_v2"}
SHARE_PRIOR_PATHS = {"totals_first_league_share_residual"}
NEURAL_SHARE_RESIDUAL_PATHS = {NEURAL_SHARE_RESIDUAL_PATH_VERSION}
NEURAL_TOTAL_SHARE_RESIDUAL_PATHS = {NEURAL_TOTAL_SHARE_RESIDUAL_PATH_VERSION}
CALIBRATED_TEAM_HEAD_PATHS = {"totals_first_team_market_calibrated"}
MARKET_HEAD_PATHS = {"totals_first_market_heads", *CALIBRATED_TEAM_HEAD_PATHS}
TOTALS_FIRST_PATHS = {
    "totals_first",
    "totals_first_residual",
    *SHARE_PRIOR_PATHS,
    *NEURAL_SHARE_RESIDUAL_PATHS,
    *NEURAL_TOTAL_SHARE_RESIDUAL_PATHS,
    *MARKET_HEAD_PATHS,
    *TOTAL_SURFACE_PATHS,
    *NEURAL_TOTAL_LADDER_PATHS,
    *PMF_SURFACE_PATHS,
}
TOTAL_HEAD_MARKETS = tuple(TOTAL_MARKETS.keys())
TEAM_HEAD_MARKETS = tuple(HOME_MARKETS.keys()) + tuple(AWAY_MARKETS.keys())
PMF_MAX_COUNT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict v2 corners family markets.")
    parser.add_argument("--league", type=str, default=None, help="Optional league filter.")
    parser.add_argument("--days", type=int, default=3, help="Horizon in days.")
    parser.add_argument("--limit", type=int, default=None, help="Max fixtures.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing trained corners artifacts.",
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=DEFAULT_SCOPE,
        help="Market scope file path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Prediction artifact output directory.",
    )
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Upsert predictions into predictions table.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional override for model_name. Use to bundle multiple v2 families under one runtime identity.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Optional override for model_version. Defaults to artifact metadata when present.",
    )
    return parser.parse_args()


def load_artifacts(
    artifact_dir: Path,
) -> tuple[dict[str, object], list[str], dict[str, float], dict[str, float | None], str, dict[str, Any]]:
    model_config_path = artifact_dir / "model_config.json"
    path_version = "sum_heads"
    model_config: dict[str, Any] = {}
    if model_config_path.exists():
        model_config = json.loads(model_config_path.read_text(encoding="utf-8")) or {}
        path_version = str(model_config.get("path_version") or "sum_heads")
    elif (artifact_dir / "total_corners_model.pkl").exists() and (artifact_dir / "home_share_model.pkl").exists():
        path_version = (
            "pmf_surface_blended"
            if (artifact_dir / "home_count_distribution_model.pkl").exists()
            else (
            NEURAL_TOTAL_MARKET_LADDER_PATH_VERSION
            if (artifact_dir / NEURAL_TOTAL_MARKET_LADDER_BUNDLE_FILENAME).exists()
            else (
            "totals_surface_calibrated"
            if (artifact_dir / "total_market_heads.pkl").exists()
            else (
            "totals_first_league_share_residual"
            if (artifact_dir / "home_share_residual_model.pkl").exists()
            else (
            "totals_first_team_market_calibrated"
            if (artifact_dir / "team_market_head_calibrators.joblib").exists()
            else (
            "totals_first_market_heads"
            if (artifact_dir / "team_market_heads.pkl").exists()
            else (
            NEURAL_TOTAL_SHARE_RESIDUAL_PATH_VERSION
            if (artifact_dir / NEURAL_TOTAL_RESIDUAL_BUNDLE_FILENAME).exists()
            and (artifact_dir / NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME).exists()
            else (
            NEURAL_SHARE_RESIDUAL_PATH_VERSION
            if (artifact_dir / NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME).exists()
            else (
            "totals_first_residual"
            if (artifact_dir / "home_delta_model.pkl").exists()
            else "totals_first"
            )
            )
            )
            )
            )
            )
            )
            )
        )

    if path_version in TOTALS_FIRST_PATHS:
        models = {
            "total": joblib.load(artifact_dir / "total_corners_model.pkl"),
            "home_share": joblib.load(artifact_dir / "home_share_model.pkl"),
        }
        if path_version == "totals_first_residual":
            models["home_delta"] = joblib.load(artifact_dir / "home_delta_model.pkl")
        if path_version in NEURAL_TOTAL_SHARE_RESIDUAL_PATHS:
            models["neural_total_residual"] = load_neural_total_residual_bundle(
                artifact_dir
                / str(model_config.get("neural_total_residual_sidecar") or NEURAL_TOTAL_RESIDUAL_BUNDLE_FILENAME)
            )
        if path_version in NEURAL_SHARE_RESIDUAL_PATHS:
            models["neural_share_residual"] = load_neural_share_residual_bundle(
                artifact_dir / str(model_config.get("neural_residual_sidecar") or NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME)
            )
        if path_version in NEURAL_TOTAL_SHARE_RESIDUAL_PATHS:
            models["neural_share_residual"] = load_neural_share_residual_bundle(
                artifact_dir
                / str(model_config.get("neural_share_residual_sidecar") or NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME)
            )
        if path_version in SHARE_PRIOR_PATHS:
            models["home_share_residual"] = joblib.load(artifact_dir / "home_share_residual_model.pkl")
            models["home_share_prior_blend"] = float(model_config.get("home_share_prior_blend") or 1.0)
        if path_version in PMF_SURFACE_PATHS:
            models["pmf_surface_models"] = {
                "home": joblib.load(artifact_dir / "home_count_distribution_model.pkl"),
                "away": joblib.load(artifact_dir / "away_count_distribution_model.pkl"),
                "max_count": int(model_config.get("pmf_max_count") or PMF_MAX_COUNT),
            }
            calibrator_path = artifact_dir / "pmf_market_calibrators.joblib"
            models["pmf_market_calibrators"] = (
                joblib.load(calibrator_path) if calibrator_path.exists() else {}
            )
            models["pmf_market_blend"] = model_config.get("pmf_market_blend") or {}
        if path_version in TOTAL_SURFACE_PATHS:
            models["total_market_heads"] = joblib.load(artifact_dir / "total_market_heads.pkl")
            calibrator_path = artifact_dir / "total_market_head_calibrators.joblib"
            models["total_market_head_calibrators"] = (
                joblib.load(calibrator_path) if calibrator_path.exists() else {}
            )
            models["total_market_head_blend"] = model_config.get("total_market_head_blend") or {}
        if path_version in NEURAL_TOTAL_LADDER_PATHS:
            models["neural_total_market_ladder"] = load_neural_total_market_ladder_bundle(
                artifact_dir
                / str(model_config.get("neural_total_market_ladder_sidecar") or NEURAL_TOTAL_MARKET_LADDER_BUNDLE_FILENAME)
            )
            calibrator_path = artifact_dir / "total_market_ladder_calibrators.joblib"
            models["total_market_ladder_calibrators"] = (
                joblib.load(calibrator_path) if calibrator_path.exists() else {}
            )
            models["total_market_ladder_blend"] = model_config.get("total_market_ladder_blend") or {}
        if path_version in MARKET_HEAD_PATHS:
            models["team_market_heads"] = joblib.load(artifact_dir / "team_market_heads.pkl")
            if path_version in CALIBRATED_TEAM_HEAD_PATHS:
                calibrator_path = artifact_dir / "team_market_head_calibrators.joblib"
                models["team_market_head_calibrators"] = (
                    joblib.load(calibrator_path) if calibrator_path.exists() else {}
                )
                models["team_market_head_blend"] = model_config.get("team_market_head_blend") or {}
    else:
        models = {
            "home": joblib.load(artifact_dir / "home_corners_model.pkl"),
            "away": joblib.load(artifact_dir / "away_corners_model.pkl"),
        }
    features = json.loads((artifact_dir / "features.json").read_text(encoding="utf-8"))
    imputation_payload = json.loads(
        (artifact_dir / "imputation.json").read_text(encoding="utf-8")
    )
    medians = {
        str(k): float(v)
        for k, v in (imputation_payload.get("global_medians") or {}).items()
        if isinstance(v, (int, float))
    }
    dispersion = json.loads((artifact_dir / "dispersion.json").read_text(encoding="utf-8"))
    league_regime_path = artifact_dir / "league_regime.json"
    league_regime = (
        json.loads(league_regime_path.read_text(encoding="utf-8"))
        if league_regime_path.exists()
        else {}
    )
    return models, list(features), medians, dispersion, path_version, league_regime


def _apply_imputation(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for feat, med in medians.items():
        out[feat] = out[feat].fillna(float(med))
    return out


def _positive_class_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    if isinstance(model, (int, float, np.integer, np.floating)):
        return np.full(len(x), float(np.clip(float(model), 0.001, 0.999)), dtype=float)
    probs = model.predict_proba(x)
    classes = [int(c) for c in getattr(model, "classes_", [0, 1])]
    if probs.ndim != 2:
        return np.full(len(x), 0.5, dtype=float)
    if probs.shape[1] == 1:
        value = 1.0 if classes and classes[0] == 1 else 0.0
        return np.full(len(x), value, dtype=float)
    idx = classes.index(1) if 1 in classes else probs.shape[1] - 1
    return np.clip(probs[:, idx], 0.001, 0.999)


def _full_count_probability_matrix(model: Any, x: pd.DataFrame, max_count: int = PMF_MAX_COUNT) -> np.ndarray:
    probs = model.predict_proba(x)
    classes = [int(c) for c in getattr(model, "classes_", [])]
    out = np.zeros((len(x), max_count + 1), dtype=float)
    if probs.ndim != 2 or probs.shape[0] != len(x):
        out[:, 0] = 1.0
        return out
    for idx, cls in enumerate(classes):
        out[:, min(int(cls), max_count)] += probs[:, idx]
    row_sums = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out / row_sums


def _pmf_expectation(pmf: np.ndarray) -> np.ndarray:
    support = np.arange(pmf.shape[1], dtype=float)
    return pmf @ support


def _derive_distribution_surface_probs(
    *,
    home_pmf: np.ndarray,
    away_pmf: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    total_pmf = np.zeros((home_pmf.shape[0], home_pmf.shape[1] + away_pmf.shape[1] - 1), dtype=float)
    for idx in range(home_pmf.shape[0]):
        total_pmf[idx] = np.convolve(home_pmf[idx], away_pmf[idx])
    total_probs: dict[str, np.ndarray] = {}
    for market, line in TOTAL_MARKETS.items():
        threshold = int(np.floor(float(line))) + 1
        total_probs[market] = total_pmf[:, threshold:].sum(axis=1)
    team_probs: dict[str, np.ndarray] = {}
    for market, line in HOME_MARKETS.items():
        threshold = int(np.floor(float(line))) + 1
        team_probs[market] = home_pmf[:, threshold:].sum(axis=1)
    for market, line in AWAY_MARKETS.items():
        threshold = int(np.floor(float(line))) + 1
        team_probs[market] = away_pmf[:, threshold:].sum(axis=1)
    return (
        _project_total_head_prob_arrays(total_probs),
        _project_team_head_prob_arrays(team_probs),
        _pmf_expectation(home_pmf),
        _pmf_expectation(away_pmf),
        _pmf_expectation(total_pmf),
    )


def _derive_full_market_prior_probs(
    *,
    total_mu: np.ndarray,
    home_mu: np.ndarray,
    away_mu: np.ndarray,
    total_r: float | None,
    home_r: float | None,
    away_r: float | None,
) -> dict[str, np.ndarray]:
    prior: dict[str, np.ndarray] = {}
    prior.update(_derive_total_market_prior_probs(total_mu=total_mu, total_r=total_r))
    prior.update(
        _derive_team_market_prior_probs(
            home_mu=home_mu,
            away_mu=away_mu,
            home_r=home_r,
            away_r=away_r,
        )
    )
    return prior


def _build_pmf_head_frame(
    x: pd.DataFrame,
    *,
    total_mu: np.ndarray,
    home_mu: np.ndarray,
    away_mu: np.ndarray,
    home_share: np.ndarray,
    prior_probs: dict[str, np.ndarray],
    path_version: str,
) -> pd.DataFrame:
    if path_version != "pmf_surface_blended_v2":
        return x
    prior_frame = pd.DataFrame(
        {
            "prior_total_corners_mu": np.asarray(total_mu, dtype=float),
            "prior_home_corners_mu": np.asarray(home_mu, dtype=float),
            "prior_away_corners_mu": np.asarray(away_mu, dtype=float),
            "prior_home_share": np.asarray(home_share, dtype=float),
            **{
                f"prior_{market}": np.asarray(values, dtype=float)
                for market, values in prior_probs.items()
            },
        }
    )
    return pd.concat([x.reset_index(drop=True), prior_frame.reset_index(drop=True)], axis=1)


def _league_home_share_prior(frame: pd.DataFrame) -> np.ndarray:
    if "league_home_share_mean" not in frame.columns:
        return np.full(len(frame), 0.5, dtype=float)
    values = pd.to_numeric(frame["league_home_share_mean"], errors="coerce").to_numpy(dtype=float)
    return np.clip(np.nan_to_num(values, nan=0.5, posinf=0.95, neginf=0.05), 0.05, 0.95)


def _predict_share_with_prior(
    *,
    residual_model: Any,
    x: pd.DataFrame,
    blend_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior_share = _league_home_share_prior(x)
    raw_share = np.clip(prior_share + residual_model.predict(x), 0.05, 0.95)
    home_share = np.clip(prior_share + float(blend_alpha) * (raw_share - prior_share), 0.05, 0.95)
    return home_share, prior_share, raw_share


def _predict_team_market_head_probs(
    *,
    models: dict[str, object],
    head_x: pd.DataFrame,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for market in TEAM_HEAD_MARKETS:
        out[market] = _positive_class_probability(models[market], head_x)
    for markets in (tuple(HOME_MARKETS.keys()), tuple(AWAY_MARKETS.keys())):
        matrix = np.column_stack([out[market] for market in markets])
        matrix = np.minimum.accumulate(np.clip(matrix, 0.001, 0.999), axis=1)
        for idx, market in enumerate(markets):
            out[market] = matrix[:, idx]
    return out


def _project_total_head_prob_arrays(total_probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {market: np.asarray(values, dtype=float).copy() for market, values in total_probs.items()}
    matrix = np.column_stack([out[market] for market in TOTAL_HEAD_MARKETS])
    matrix = np.minimum.accumulate(np.clip(matrix, 0.001, 0.999), axis=1)
    for idx, market in enumerate(TOTAL_HEAD_MARKETS):
        out[market] = matrix[:, idx]
    return out


def _derive_total_market_prior_probs(
    *,
    total_mu: np.ndarray,
    total_r: float | None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    total_arr = np.asarray(total_mu, dtype=float)
    for market, line in TOTAL_MARKETS.items():
        out[market] = np.asarray(
            [over_probability(float(mu), line, total_r) for mu in total_arr],
            dtype=float,
        )
    return _project_total_head_prob_arrays(out)


def _build_total_head_frame(
    x: pd.DataFrame,
    prior_probs: dict[str, np.ndarray],
) -> pd.DataFrame:
    prior_frame = pd.DataFrame(
        {f"prior_{market}": np.asarray(values, dtype=float) for market, values in prior_probs.items()}
    )
    return pd.concat([x.reset_index(drop=True), prior_frame.reset_index(drop=True)], axis=1)


def _derive_team_market_prior_probs(
    *,
    home_mu: np.ndarray,
    away_mu: np.ndarray,
    home_r: float | None,
    away_r: float | None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    home_arr = np.asarray(home_mu, dtype=float)
    away_arr = np.asarray(away_mu, dtype=float)
    for market, line in HOME_MARKETS.items():
        out[market] = np.asarray(
            [over_probability(float(mu), line, home_r) for mu in home_arr],
            dtype=float,
        )
    for market, line in AWAY_MARKETS.items():
        out[market] = np.asarray(
            [over_probability(float(mu), line, away_r) for mu in away_arr],
            dtype=float,
        )
    return _project_team_head_prob_arrays(out)


def _project_team_head_prob_arrays(team_probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {market: np.asarray(values, dtype=float).copy() for market, values in team_probs.items()}
    for markets in (tuple(HOME_MARKETS.keys()), tuple(AWAY_MARKETS.keys())):
        matrix = np.column_stack([out[market] for market in markets])
        matrix = np.minimum.accumulate(np.clip(matrix, 0.001, 0.999), axis=1)
        for idx, market in enumerate(markets):
            out[market] = matrix[:, idx]
    return out


def _build_team_head_frame(
    x: pd.DataFrame,
    prior_probs: dict[str, np.ndarray],
) -> pd.DataFrame:
    prior_frame = pd.DataFrame(
        {f"prior_{market}": np.asarray(values, dtype=float) for market, values in prior_probs.items()}
    )
    return pd.concat([x.reset_index(drop=True), prior_frame.reset_index(drop=True)], axis=1)


def _predict_corner_rates(
    *,
    models: dict[str, object],
    x: pd.DataFrame,
    path_version: str,
    total_r: float | None = None,
    home_r: float | None = None,
    away_r: float | None = None,
) -> dict[str, Any]:
    if path_version in TOTALS_FIRST_PATHS:
        total_mu = np.clip(models["total"].predict(x), 0.1, 30.0)
        total_mu_base = np.asarray(total_mu, dtype=float)
        home_share_prior = None
        home_share_raw = None
        total_raw = np.asarray(total_mu, dtype=float)
        total_delta = None
        neural_share_delta = None
        if path_version in NEURAL_TOTAL_SHARE_RESIDUAL_PATHS:
            base_share_for_total = np.clip(models["home_share"].predict(x), 0.05, 0.95)
            total_delta = np.asarray(
                predict_neural_total_residual_delta(
                    models["neural_total_residual"],
                    x,
                    base_total_mu=total_mu_base,
                    base_home_share=base_share_for_total,
                ),
                dtype=float,
            )
            total_raw = total_mu_base + total_delta
            total_mu = np.clip(total_raw, 0.1, 30.0)
        if path_version in SHARE_PRIOR_PATHS:
            home_share, home_share_prior, home_share_raw = _predict_share_with_prior(
                residual_model=models["home_share_residual"],
                x=x,
                blend_alpha=float(models.get("home_share_prior_blend", 1.0)),
            )
        else:
            home_share = np.clip(models["home_share"].predict(x), 0.05, 0.95)
        base_share = np.asarray(home_share, dtype=float)
        if path_version in (NEURAL_SHARE_RESIDUAL_PATHS | NEURAL_TOTAL_SHARE_RESIDUAL_PATHS):
            neural_share_delta = np.asarray(
                predict_neural_share_residual_delta(
                    models["neural_share_residual"],
                    x,
                    base_total_mu=np.asarray(total_mu, dtype=float),
                    base_home_share=base_share,
                ),
                dtype=float,
            )
            home_share_raw = base_share + neural_share_delta
            home_share = np.clip(home_share_raw, 0.05, 0.95)
        base_home_mu: list[float] = []
        home_mu: list[float] = []
        away_mu: list[float] = []
        for total_value, base_share_value, share_value in zip(total_mu, base_share, home_share, strict=True):
            base_hm, _ = reconcile_team_means_from_total_share(
                total_mu=float(total_value),
                home_share=float(base_share_value),
            )
            hm, am = reconcile_team_means_from_total_share(
                total_mu=float(total_value),
                home_share=float(share_value),
            )
            base_home_mu.append(base_hm)
            home_mu.append(hm)
            away_mu.append(am)
        home_delta = None
        if path_version == "totals_first_residual":
            home_delta = np.clip(models["home_delta"].predict(x), -8.0, 8.0)
            adjusted_home: list[float] = []
            adjusted_away: list[float] = []
            for total_value, base_home_value, delta_value in zip(
                total_mu,
                base_home_mu,
                home_delta,
                strict=True,
            ):
                hm, am = reconcile_team_means_from_total_home_mu(
                    total_mu=float(total_value),
                    proposed_home_mu=float(base_home_value + delta_value),
                )
                adjusted_home.append(hm)
                adjusted_away.append(am)
            home_mu = adjusted_home
            away_mu = adjusted_away
        out: dict[str, Any] = {
            "total": np.asarray(total_mu, dtype=float),
            "base_total": total_mu_base,
            "total_raw": np.asarray(total_raw, dtype=float),
            "total_residual_delta": (
                np.asarray(total_delta, dtype=float)
                if total_delta is not None
                else np.zeros(len(total_mu), dtype=float)
            ),
            "home_share": np.asarray(home_share, dtype=float),
            "home_share_prior": (
                np.asarray(home_share_prior, dtype=float)
                if home_share_prior is not None
                else np.full(len(total_mu), np.nan, dtype=float)
            ),
            "home_share_raw": (
                np.asarray(home_share_raw, dtype=float)
                if home_share_raw is not None
                else np.asarray(home_share, dtype=float)
            ),
            "base_home": np.asarray(base_home_mu, dtype=float),
            "home_delta": (
                np.asarray(home_delta, dtype=float)
                if home_delta is not None
                else np.zeros(len(total_mu), dtype=float)
            ),
            "home_share_residual_delta": (
                np.asarray(neural_share_delta, dtype=float)
                if neural_share_delta is not None
                else np.zeros(len(total_mu), dtype=float)
            ),
            "home": np.asarray(home_mu, dtype=float),
            "away": np.asarray(away_mu, dtype=float),
        }
        if path_version in PMF_SURFACE_PATHS:
            pmf_models = models["pmf_surface_models"]
            max_count = int(pmf_models.get("max_count", PMF_MAX_COUNT))
            prior_all = _derive_full_market_prior_probs(
                total_mu=np.asarray(total_mu, dtype=float),
                home_mu=np.asarray(home_mu, dtype=float),
                away_mu=np.asarray(away_mu, dtype=float),
                total_r=total_r,
                home_r=home_r,
                away_r=away_r,
            )
            pmf_x = _build_pmf_head_frame(
                x,
                total_mu=np.asarray(total_mu, dtype=float),
                home_mu=np.asarray(home_mu, dtype=float),
                away_mu=np.asarray(away_mu, dtype=float),
                home_share=np.asarray(home_share, dtype=float),
                prior_probs=prior_all,
                path_version=path_version,
            )
            home_pmf = _full_count_probability_matrix(pmf_models["home"], pmf_x, max_count)
            away_pmf = _full_count_probability_matrix(pmf_models["away"], pmf_x, max_count)
            raw_total_probs, raw_team_probs, raw_home_mu, raw_away_mu, raw_total_mu = _derive_distribution_surface_probs(
                home_pmf=home_pmf,
                away_pmf=away_pmf,
            )
            calibrators = models.get("pmf_market_calibrators") or {}
            blend = models.get("pmf_market_blend") or {}
            total_market_probs: dict[str, np.ndarray] = {}
            team_market_probs: dict[str, np.ndarray] = {}
            for market in TOTAL_HEAD_MARKETS:
                calibrated = apply_binary_calibrator(
                    calibrators.get(market, {"method": "identity"}),
                    raw_total_probs[market],
                )
                alpha = float(blend.get(market, 0.35))
                total_market_probs[market] = np.clip(
                    prior_all[market] + alpha * (calibrated - prior_all[market]),
                    0.001,
                    0.999,
                )
            for market in TEAM_HEAD_MARKETS:
                calibrated = apply_binary_calibrator(
                    calibrators.get(market, {"method": "identity"}),
                    raw_team_probs[market],
                )
                alpha = float(blend.get(market, 0.35))
                team_market_probs[market] = np.clip(
                    prior_all[market] + alpha * (calibrated - prior_all[market]),
                    0.001,
                    0.999,
                )
            out["total_market_probs"] = _project_total_head_prob_arrays(total_market_probs)
            out["team_market_probs"] = _project_team_head_prob_arrays(team_market_probs)
            out["pmf_home"] = np.asarray(raw_home_mu, dtype=float)
            out["pmf_away"] = np.asarray(raw_away_mu, dtype=float)
            out["pmf_total"] = np.asarray(raw_total_mu, dtype=float)
        if path_version in TOTAL_SURFACE_PATHS:
            prior_total_probs = _derive_total_market_prior_probs(
                total_mu=np.asarray(total_mu, dtype=float),
                total_r=total_r,
            )
            head_x = _build_total_head_frame(x, prior_total_probs)
            raw_total_probs = {
                market: _positive_class_probability(models["total_market_heads"][market], head_x)
                for market in TOTAL_HEAD_MARKETS
            }
            calibrators = models.get("total_market_head_calibrators") or {}
            blend = models.get("total_market_head_blend") or {}
            total_market_probs: dict[str, np.ndarray] = {}
            for market in TOTAL_HEAD_MARKETS:
                calibrated = apply_binary_calibrator(
                    calibrators.get(market, {"method": "identity"}),
                    raw_total_probs[market],
                )
                alpha = float(blend.get(market, 0.35))
                total_market_probs[market] = np.clip(
                    prior_total_probs[market] + alpha * (calibrated - prior_total_probs[market]),
                    0.001,
                    0.999,
                )
            out["total_market_probs"] = _project_total_head_prob_arrays(total_market_probs)
        if path_version in NEURAL_TOTAL_LADDER_PATHS:
            prior_total_probs = _derive_total_market_prior_probs(
                total_mu=np.asarray(total_mu, dtype=float),
                total_r=total_r,
            )
            raw_total_probs = predict_neural_total_market_ladder_probs(
                models["neural_total_market_ladder"],
                x,
                prior_total_probs=prior_total_probs,
            )
            total_market_probs: dict[str, np.ndarray] = {}
            calibrators = models.get("total_market_ladder_calibrators") or {}
            blend = models.get("total_market_ladder_blend") or {}
            for market in TOTAL_HEAD_MARKETS:
                calibrated = apply_binary_calibrator(
                    calibrators.get(market, {"method": "identity"}),
                    raw_total_probs[market],
                )
                alpha = float(blend.get(market, 0.35))
                total_market_probs[market] = np.clip(
                    prior_total_probs[market] + alpha * (calibrated - prior_total_probs[market]),
                    0.001,
                    0.999,
                )
            out["total_market_probs"] = _project_total_head_prob_arrays(total_market_probs)
        if path_version in MARKET_HEAD_PATHS:
            if path_version in CALIBRATED_TEAM_HEAD_PATHS:
                prior_team_probs = _derive_team_market_prior_probs(
                    home_mu=np.asarray(home_mu, dtype=float),
                    away_mu=np.asarray(away_mu, dtype=float),
                    home_r=home_r,
                    away_r=away_r,
                )
                head_x = _build_team_head_frame(x, prior_team_probs)
                raw_team_probs = _predict_team_market_head_probs(
                    models=models["team_market_heads"],
                    head_x=head_x,
                )
                calibrators = models.get("team_market_head_calibrators") or {}
                blend = models.get("team_market_head_blend") or {}
                team_market_probs: dict[str, np.ndarray] = {}
                for market in TEAM_HEAD_MARKETS:
                    calibrated = apply_binary_calibrator(
                        calibrators.get(market, {"method": "identity"}),
                        raw_team_probs[market],
                    )
                    alpha = float(blend.get(market, 0.35))
                    team_market_probs[market] = np.clip(
                        prior_team_probs[market] + alpha * (calibrated - prior_team_probs[market]),
                        0.001,
                        0.999,
                    )
                out["team_market_probs"] = _project_team_head_prob_arrays(team_market_probs)
            else:
                out["team_market_probs"] = _predict_team_market_head_probs(
                    models=models["team_market_heads"],
                    head_x=x,
                )
        return out
    return {
        "home": np.clip(models["home"].predict(x), 0.05, 20.0),
        "away": np.clip(models["away"].predict(x), 0.05, 20.0),
    }


def upsert_predictions(rows: list[tuple[int, str, str, str, float, str]]) -> int:
    if not rows:
        return 0
    query = """
    INSERT INTO predictions (
        fixture_id,
        market_code,
        model_name,
        model_version,
        p_model,
        metadata_json,
        created_at
    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, NOW())
    ON CONFLICT (fixture_id, market_code, model_name, model_version)
    DO UPDATE SET
        p_model = EXCLUDED.p_model,
        metadata_json = EXCLUDED.metadata_json,
        created_at = NOW();
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.executemany(query, rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def main() -> None:
    args = parse_args()
    models, features, medians, dispersion, path_version, league_regime = load_artifacts(args.artifact_dir)
    model_name, model_version = resolve_model_identity(
        args.artifact_dir,
        default_model_name=MODEL_NAME,
        default_model_version=MODEL_VERSION,
        override_model_name=args.model_name,
        override_model_version=args.model_version,
    )
    scope_markets = set(load_scope_markets(args.scope))
    corners_markets = {
        "c75",
        "c85",
        "c95",
        "c105",
        "hc25",
        "hc35",
        "hc45",
        "hc55",
        "ac25",
        "ac35",
        "ac45",
        "ac55",
    } & scope_markets

    fixtures = fetch_candidate_fixtures(
        days=args.days,
        league=args.league,
        limit=args.limit,
        backfill_days=None,
    )
    if fixtures.empty:
        print("No eligible fixtures for corners v2 prediction.")
        return

    featured = add_derived_features(fixtures)
    featured = add_corners_context_features(featured)
    featured = apply_league_regime(featured, league_regime)
    for feat in features:
        if feat not in featured.columns:
            featured[feat] = np.nan
    scored = _apply_imputation(featured, medians)

    x = scored[features]
    preds = _predict_corner_rates(
        models=models,
        x=x,
        path_version=path_version,
        total_r=dispersion.get("total_r"),
        home_r=dispersion.get("home_r"),
        away_r=dispersion.get("away_r"),
    )
    home_mu = preds["home"]
    away_mu = preds["away"]
    total_mu = preds.get("total")
    total_market_probs = preds.get("total_market_probs")
    home_share = preds.get("home_share")
    home_share_prior = preds.get("home_share_prior")
    home_share_raw = preds.get("home_share_raw")
    home_delta = preds.get("home_delta")
    base_home = preds.get("base_home")
    pmf_home = preds.get("pmf_home")
    pmf_away = preds.get("pmf_away")
    pmf_total = preds.get("pmf_total")
    team_market_probs = preds.get("team_market_probs")

    rows_csv: list[dict[str, Any]] = []
    rows_db: list[tuple[int, str, str, str, float, str]] = []
    for idx, (fixture, hm, am) in enumerate(zip(
        scored.itertuples(index=False),
        home_mu,
        away_mu,
        strict=True,
    )):
        fixture_id = int(fixture.fixture_id)
        hm = float(hm)
        am = float(am)
        total_overrides = None
        if total_market_probs is not None:
            total_overrides = {
                market: float(values[idx])
                for market, values in total_market_probs.items()
            }
        team_overrides = None
        if team_market_probs is not None:
            team_overrides = {
                market: float(values[idx])
                for market, values in team_market_probs.items()
            }
        markets = derive_and_validate_corners(
            home_mu=hm,
            away_mu=am,
            total_r=dispersion.get("total_r"),
            home_r=dispersion.get("home_r"),
            away_r=dispersion.get("away_r"),
            total_mu_override=(float(total_mu[idx]) if total_mu is not None else None),
            total_market_overrides=total_overrides,
            team_market_overrides=team_overrides,
        )
        for market_code, prob in markets.items():
            if market_code not in corners_markets:
                continue
            p_model = float(np.clip(prob, 0.001, 0.999))
            metadata = {
                "model_family": "corners_v2",
                "path_version": path_version,
                "home_corners_mu_pred": hm,
                "away_corners_mu_pred": am,
                "total_corners_mu_pred": (float(total_mu[idx]) if total_mu is not None else hm + am),
                "total_market_head_pred": (float(total_overrides[market_code]) if total_overrides is not None and market_code in total_overrides else None),
                "home_share_pred": (float(home_share[idx]) if home_share is not None else None),
                "home_share_prior_pred": (float(home_share_prior[idx]) if home_share_prior is not None else None),
                "home_share_raw_pred": (float(home_share_raw[idx]) if home_share_raw is not None else None),
                "base_home_corners_mu_pred": (float(base_home[idx]) if base_home is not None else None),
                "home_delta_pred": (float(home_delta[idx]) if home_delta is not None else None),
                "pmf_home_corners_mu_pred": (float(pmf_home[idx]) if pmf_home is not None else None),
                "pmf_away_corners_mu_pred": (float(pmf_away[idx]) if pmf_away is not None else None),
                "pmf_total_corners_mu_pred": (float(pmf_total[idx]) if pmf_total is not None else None),
                "team_market_head_pred": (float(team_overrides[market_code]) if team_overrides is not None and market_code in team_overrides else None),
                "dispersion_total_r": dispersion.get("total_r"),
                "dispersion_home_r": dispersion.get("home_r"),
                "dispersion_away_r": dispersion.get("away_r"),
                "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            }
            rows_csv.append(
                {
                    "fixture_id": fixture_id,
                    "market_code": market_code,
                    "model_name": model_name,
                    "model_version": model_version,
                    "p_model": p_model,
                    "path_version": path_version,
                    "home_corners_mu_pred": hm,
                    "away_corners_mu_pred": am,
                    "total_corners_mu_pred": (float(total_mu[idx]) if total_mu is not None else hm + am),
                    "total_market_head_pred": (float(total_overrides[market_code]) if total_overrides is not None and market_code in total_overrides else None),
                    "home_share_prior_pred": (float(home_share_prior[idx]) if home_share_prior is not None else None),
                    "home_share_raw_pred": (float(home_share_raw[idx]) if home_share_raw is not None else None),
                    "home_delta_pred": (float(home_delta[idx]) if home_delta is not None else None),
                    "pmf_home_corners_mu_pred": (float(pmf_home[idx]) if pmf_home is not None else None),
                    "pmf_away_corners_mu_pred": (float(pmf_away[idx]) if pmf_away is not None else None),
                    "pmf_total_corners_mu_pred": (float(pmf_total[idx]) if pmf_total is not None else None),
                    "team_market_head_pred": (float(team_overrides[market_code]) if team_overrides is not None and market_code in team_overrides else None),
                }
            )
            rows_db.append(
                (
                    fixture_id,
                    market_code,
                    model_name,
                    model_version,
                    p_model,
                    json.dumps(metadata),
                )
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"corners_v2_predictions_{stamp}.csv"
    pd.DataFrame(rows_csv).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote v2 corners predictions: {out_path}")

    if args.write_db:
        written = upsert_predictions(rows_db)
        print(f"Upserted {written} predictions into DB.")


if __name__ == "__main__":
    main()
