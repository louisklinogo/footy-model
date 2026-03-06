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
from src.modeling.evaluation.predict_market_outcomes_fixtures_first import (
    add_derived_features,
    fetch_candidate_fixtures,
)
from src.modeling.v2.families.corners.derive_lines import derive_and_validate_corners
from src.modeling.v2.io.artifact_identity import resolve_model_identity
from src.modeling.v2.io.baseline_registry import load_scope_markets


DEFAULT_ARTIFACT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "corners"
DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "artifacts" / "v2" / "predictions"
MODEL_NAME = "corners_v2"
MODEL_VERSION = "distribution_head_v1"


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
        "--model-version",
        type=str,
        default=None,
        help="Optional override for model_version. Defaults to artifact metadata when present.",
    )
    return parser.parse_args()


def load_artifacts(
    artifact_dir: Path,
) -> tuple[object, object, list[str], dict[str, float], dict[str, float | None]]:
    home_model = joblib.load(artifact_dir / "home_corners_model.pkl")
    away_model = joblib.load(artifact_dir / "away_corners_model.pkl")
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
    return home_model, away_model, list(features), medians, dispersion


def _apply_imputation(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for feat, med in medians.items():
        out[feat] = out[feat].fillna(float(med))
    return out


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
    home_model, away_model, features, medians, dispersion = load_artifacts(args.artifact_dir)
    model_name, model_version = resolve_model_identity(
        args.artifact_dir,
        default_model_name=MODEL_NAME,
        default_model_version=MODEL_VERSION,
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
    for feat in features:
        if feat not in featured.columns:
            featured[feat] = np.nan
    scored = _apply_imputation(featured, medians)

    x = scored[features]
    home_mu = np.clip(home_model.predict(x), 0.05, 20.0)
    away_mu = np.clip(away_model.predict(x), 0.05, 20.0)

    rows_csv: list[dict[str, Any]] = []
    rows_db: list[tuple[int, str, str, str, float, str]] = []
    for fixture, hm, am in zip(
        scored.itertuples(index=False),
        home_mu,
        away_mu,
        strict=True,
    ):
        fixture_id = int(fixture.fixture_id)
        hm = float(hm)
        am = float(am)
        markets = derive_and_validate_corners(
            home_mu=hm,
            away_mu=am,
            total_r=dispersion.get("total_r"),
            home_r=dispersion.get("home_r"),
            away_r=dispersion.get("away_r"),
        )
        for market_code, prob in markets.items():
            if market_code not in corners_markets:
                continue
            p_model = float(np.clip(prob, 0.001, 0.999))
            metadata = {
                "model_family": "corners_v2",
                "home_corners_mu_pred": hm,
                "away_corners_mu_pred": am,
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
                    "home_corners_mu_pred": hm,
                    "away_corners_mu_pred": am,
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
