"""
Generate fixtures-first pre-match predictions and upsert into predictions.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_utils import connect_db


MODEL_DIR = Path("models/v3_fixtures_first")
MODEL_NAME = "fixtures_first_gbm"
MODEL_VERSION = "v3"
MARKETS = ("o15", "o25", "c85")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fixtures-first markets and write to DB")
    parser.add_argument("--league", type=str, default=None, help="Optional league_code filter")
    parser.add_argument("--days", type=int, default=3, help="Prediction horizon in days")
    parser.add_argument("--limit", type=int, default=None, help="Optional max fixtures")
    return parser.parse_args()


def _latest_odds_expr(line: str, side: str) -> str:
    return (
        "CASE "
        f"WHEN (od.ou_json -> '{line}' ->> '{side}') ~ '^[-+]?[0-9]*\\.?[0-9]+$' "
        f"THEN (od.ou_json -> '{line}' ->> '{side}')::double precision "
        "ELSE NULL END"
    )


def load_artifacts() -> tuple[list[str], dict[str, dict[str, dict[str, float] | float]], dict[str, object]]:
    with (MODEL_DIR / "features.json").open("r", encoding="utf-8") as f:
        features = json.load(f)
    with (MODEL_DIR / "imputation.json").open("r", encoding="utf-8") as f:
        imputation = json.load(f)
    models = {market: joblib.load(MODEL_DIR / f"gbm_{market}.pkl") for market in MARKETS}
    return features, imputation, models


def fetch_candidate_fixtures(days: int, league: str | None, limit: int | None) -> pd.DataFrame:
    base_query = f"""
    SELECT
        f.fixture_id,
        f.league_code,
        f.match_datetime_utc,
        {_latest_odds_expr('1.5', 'over')} AS odds_over_15,
        {_latest_odds_expr('1.5', 'under')} AS odds_under_15,
        {_latest_odds_expr('2.5', 'over')} AS odds_over_25,
        {_latest_odds_expr('2.5', 'under')} AS odds_under_25,
        od.snapshot_time_utc AS odds_snapshot_time_utc,
        tph.sample_size AS home_sample_size,
        tph.rolling_xg AS home_rolling_xg,
        tph.rolling_xg_against AS home_rolling_xg_against,
        tph.rolling_xgot AS home_rolling_xgot,
        tph.rolling_xgot_against AS home_rolling_xgot_against,
        tph.rolling_xa AS home_rolling_xa,
        tph.rolling_xa_against AS home_rolling_xa_against,
        tph.rolling_box_touches AS home_rolling_box_touches,
        tph.rolling_box_touches_against AS home_rolling_box_touches_against,
        tph.rolling_big_chances AS home_rolling_big_chances,
        tph.rolling_big_chances_against AS home_rolling_big_chances_against,
        tph.rolling_crosses AS home_rolling_crosses,
        tph.rolling_crosses_against AS home_rolling_crosses_against,
        tph.rolling_sot AS home_rolling_sot,
        tph.rolling_sot_against AS home_rolling_sot_against,
        tph.rolling_corners AS home_rolling_corners,
        tph.rolling_corners_against AS home_rolling_corners_against,
        tph.rolling_goals_prevented AS home_rolling_goals_prevented,
        tph.rolling_goals_prevented_against AS home_rolling_goals_prevented_against,
        tpa.sample_size AS away_sample_size,
        tpa.rolling_xg AS away_rolling_xg,
        tpa.rolling_xg_against AS away_rolling_xg_against,
        tpa.rolling_xgot AS away_rolling_xgot,
        tpa.rolling_xgot_against AS away_rolling_xgot_against,
        tpa.rolling_xa AS away_rolling_xa,
        tpa.rolling_xa_against AS away_rolling_xa_against,
        tpa.rolling_box_touches AS away_rolling_box_touches,
        tpa.rolling_box_touches_against AS away_rolling_box_touches_against,
        tpa.rolling_big_chances AS away_rolling_big_chances,
        tpa.rolling_big_chances_against AS away_rolling_big_chances_against,
        tpa.rolling_crosses AS away_rolling_crosses,
        tpa.rolling_crosses_against AS away_rolling_crosses_against,
        tpa.rolling_sot AS away_rolling_sot,
        tpa.rolling_sot_against AS away_rolling_sot_against,
        tpa.rolling_corners AS away_rolling_corners,
        tpa.rolling_corners_against AS away_rolling_corners_against,
        tpa.rolling_goals_prevented AS away_rolling_goals_prevented,
        tpa.rolling_goals_prevented_against AS away_rolling_goals_prevented_against
    FROM fixtures f
    LEFT JOIN team_premium_snapshots tph
        ON tph.fixture_id = f.fixture_id
       AND tph.is_home = true
    LEFT JOIN team_premium_snapshots tpa
        ON tpa.fixture_id = f.fixture_id
       AND tpa.is_home = false
    LEFT JOIN LATERAL (
        SELECT fos.snapshot_time_utc, fos.ou_json
        FROM fixture_odds_snapshots fos
        WHERE fos.fixture_id = f.fixture_id
          AND fos.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY fos.snapshot_time_utc DESC
        LIMIT 1
    ) od ON true
    WHERE f.status = 'scheduled'
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc > NOW()
      AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
    """
    params: list[object] = [days]
    if league:
        base_query += " AND f.league_code = %s"
        params.append(league)
    base_query += " ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC"
    if limit is not None:
        base_query += " LIMIT %s"
        params.append(limit)

    conn = connect_db()
    try:
        df = pd.read_sql(base_query, conn, params=tuple(params))
    finally:
        conn.close()
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["xg_net_diff"] = out["home_rolling_xg"] - out["away_rolling_xg_against"]
    out["xgot_net_diff"] = out["home_rolling_xgot"] - out["away_rolling_xgot_against"]
    out["xa_net_diff"] = out["home_rolling_xa"] - out["away_rolling_xa_against"]
    out["corners_net_diff"] = out["home_rolling_corners"] - out["away_rolling_corners_against"]
    out["sample_size_diff"] = out["home_sample_size"] - out["away_sample_size"]

    out["implied_over15"] = np.where(out["odds_over_15"] > 1.0, 1.0 / out["odds_over_15"], np.nan)
    out["implied_under15"] = np.where(out["odds_under_15"] > 1.0, 1.0 / out["odds_under_15"], np.nan)
    out["implied_over25"] = np.where(out["odds_over_25"] > 1.0, 1.0 / out["odds_over_25"], np.nan)
    out["implied_under25"] = np.where(out["odds_under_25"] > 1.0, 1.0 / out["odds_under_25"], np.nan)

    out["odds_gap_15"] = out["odds_over_15"] - out["odds_under_15"]
    out["odds_gap_25"] = out["odds_over_25"] - out["odds_under_25"]
    return out


def apply_imputation(df: pd.DataFrame, features: list[str], imputation: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    global_medians = imputation.get("global_medians", {})
    league_medians = imputation.get("league_medians", {})

    for feature in features:
        if feature not in out.columns:
            out[feature] = np.nan

        league_map = league_medians.get(feature, {})
        league_fill = out["league_code"].map(league_map)
        global_fill = global_medians.get(feature, 0.0)
        out[feature] = out[feature].fillna(league_fill).fillna(global_fill).fillna(0.0)

    return out


def build_prediction_rows(
    scored: pd.DataFrame,
    features: list[str],
    models: dict[str, object],
) -> list[tuple[int, str, str, str, float, str]]:
    x_mat = scored[features]
    rows: list[tuple[int, str, str, str, float, str]] = []

    for _, fixture in scored.iterrows():
        fixture_id = int(fixture["fixture_id"])
        metadata = {
            "features_missing_count": int(fixture["features_missing_count"]),
            "home_sample_size": None if pd.isna(fixture["home_sample_size"]) else float(fixture["home_sample_size"]),
            "away_sample_size": None if pd.isna(fixture["away_sample_size"]) else float(fixture["away_sample_size"]),
            "odds_snapshot_time_utc": None
            if pd.isna(fixture["odds_snapshot_time_utc"])
            else fixture["odds_snapshot_time_utc"].isoformat(),
        }
        metadata_json = json.dumps(metadata)

        row_df = x_mat.loc[[fixture.name]]
        for market in MARKETS:
            model = models[market]
            p_model = positive_class_probability(model=model, x_row=row_df)
            rows.append((fixture_id, market, MODEL_NAME, MODEL_VERSION, p_model, metadata_json))

    return rows


def positive_class_probability(model: object, x_row: pd.DataFrame) -> float:
    probs = model.predict_proba(x_row)
    classes = getattr(model, "classes_", None)

    if classes is None:
        return float(np.clip(probs[0, 1], 0.001, 0.999))

    classes_arr = np.asarray(classes)
    if len(classes_arr) == 1:
        only_class = int(classes_arr[0])
        return 0.999 if only_class == 1 else 0.001

    idx = int(np.where(classes_arr == 1)[0][0]) if np.any(classes_arr == 1) else 1
    return float(np.clip(probs[0, idx], 0.001, 0.999))


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
    features, imputation, models = load_artifacts()

    fixtures = fetch_candidate_fixtures(days=args.days, league=args.league, limit=args.limit)
    if fixtures.empty:
        print("No eligible fixtures found.")
        return

    featured = add_derived_features(fixtures)
    featured["features_missing_count"] = featured[features].isna().sum(axis=1)
    scored = apply_imputation(featured, features, imputation)

    prediction_rows = build_prediction_rows(scored=scored, features=features, models=models)
    written = upsert_predictions(prediction_rows)

    print(f"Processed {len(scored)} fixtures, upserted {written} market predictions ({len(scored) * len(MARKETS)} expected).")


if __name__ == "__main__":
    main()
