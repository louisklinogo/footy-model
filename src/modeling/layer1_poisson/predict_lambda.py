"""
Predict λ_home and λ_away for fixtures and save to predictions table.

This is Layer 1 of the 3-Layer syndicate architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.features.build_features import build_point_in_time_features

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

MODEL_NAME = "lambda_xgb"


def get_fixtures_with_snapshots(limit: int | None = None, leagues: list[str] | None = None) -> list[int]:
    """Fetch fixtures that have team_premium_snapshots built."""
    conn = connect_db()
    
    # We want to predict for BOTH finished (for historical scoring/tracking) 
    # AND scheduled (for the daily slips/situational layer) fixtures.
    conditions = ["(f.status = 'ft' OR f.status = 'scheduled')"]
    params = []
    
    if leagues:
        conditions.append("f.league_code = ANY(%s)")
        params.append(leagues)
    
    limit_sql = ""
    if limit:
        limit_sql = "LIMIT %s"
        params.append(limit)
    
    query = f"""
        SELECT DISTINCT f.fixture_id
        FROM fixtures f
        JOIN team_premium_snapshots s ON s.fixture_id = f.fixture_id
        WHERE {' AND '.join(conditions)}
        ORDER BY f.fixture_id
        {limit_sql}
    """
    
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    
    conn.close()
    return [r[0] for r in rows]


def save_predictions_to_db(predictions: list[dict]) -> int:
    """Upsert predictions to DB. Stores lambda in metadata_json since p_model has 0-1 constraint."""
    if not predictions:
        return 0
    
    conn = connect_db()
    upsert_sql = """
        INSERT INTO predictions 
            (fixture_id, model_name, model_version, market_code, p_model, metadata_json, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (fixture_id, market_code, model_name, model_version) DO UPDATE SET
            p_model = EXCLUDED.p_model,
            metadata_json = EXCLUDED.metadata_json,
            created_at = NOW()
    """
    
    with conn.cursor() as cur:
        for pred in predictions:
            meta = {"lambda": pred["lambda_value"]}
            if pred.get("is_low_confidence"):
                meta["is_low_confidence"] = True
                meta["low_confidence_reason"] = pred["low_confidence_reason"]
                
            cur.execute(upsert_sql, (
                pred["fixture_id"],
                pred["model_name"],
                pred["model_version"],
                pred["market_code"],
                pred["p_model"],
                json.dumps(meta),
            ))
        conn.commit()
    
    conn.close()
    return len(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict λ values for fixtures")
    parser.add_argument("--limit", type=int, default=None, help="Max fixtures to process")
    parser.add_argument("--leagues", type=str, default=None, help="Comma-separated league codes")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for processing")
    args = parser.parse_args()
    
    leagues_list = None
    if args.leagues:
        leagues_list = [l.strip() for l in args.leagues.split(",")]
    
    artifacts_dir = ROOT_DIR / 'model_artifacts' / 'poisson_model'
    
    if not (artifacts_dir / "xgb_lambda_home_v1.json").exists():
        print("Model artifacts not found. Run train_lambda.py first.")
        return
    
    with open(artifacts_dir / "calibration_params.json", "r") as f:
        params = json.load(f)
    
    feature_cols = params["features"]
    print(f"Loaded {len(feature_cols)} features")
    
    model_h = xgb.Booster()
    model_h.load_model(artifacts_dir / "xgb_lambda_home_v1.json")
    
    model_a = xgb.Booster()
    model_a.load_model(artifacts_dir / "xgb_lambda_away_v1.json")
    
    fixture_ids = get_fixtures_with_snapshots(limit=args.limit, leagues=leagues_list)
    
    if not fixture_ids:
        print("No fixtures with snapshots found.")
        return
    
    print(f"Processing {len(fixture_ids)} fixtures in batches of {args.batch_size}...")
    
    total_saved = 0
    
    for i in range(0, len(fixture_ids), args.batch_size):
        batch_ids = fixture_ids[i:i + args.batch_size]
        print(f"  Batch {i//args.batch_size + 1}: {len(batch_ids)} fixtures")
        
        df = build_point_in_time_features(batch_ids, target_window_mins=60)
        
        if df.empty:
            print("    No features built, skipping...")
            continue
        
        df = df.dropna(subset=["h_xg", "a_xg", "league_avg_xg"])
        
        if df.empty:
            print("    No valid features after filtering, skipping...")
            continue
        
        d_matrix = xgb.DMatrix(df[feature_cols])
        df["lambda_home"] = model_h.predict(d_matrix)
        df["lambda_away"] = model_a.predict(d_matrix)
        
        # Vectorised confidence guard: flag early-season fixtures (< 6 games played by either team)
        df["is_low_confidence"] = (df["h_sample_size"] < 6) | (df["a_sample_size"] < 6)

        predictions = []
        for _, row in df.iterrows():
            is_low = bool(row["is_low_confidence"])
            reason = "early_season_min_6_games_not_met" if is_low else None

            predictions.append({
                "fixture_id": int(row["fixture_id"]),
                "model_name": MODEL_NAME,
                "model_version": "v1",
                "market_code": "lambda_home",
                "p_model": 0.5,
                "lambda_value": float(row["lambda_home"]),
                "is_low_confidence": is_low,
                "low_confidence_reason": reason
            })
            predictions.append({
                "fixture_id": int(row["fixture_id"]),
                "model_name": MODEL_NAME,
                "model_version": "v1",
                "market_code": "lambda_away",
                "p_model": 0.5,
                "lambda_value": float(row["lambda_away"]),
                "is_low_confidence": is_low,
                "low_confidence_reason": reason
            })
        
        saved = save_predictions_to_db(predictions)
        total_saved += saved
        print(f"    Saved {saved} predictions")
    
    print(f"\nTotal predictions saved: {total_saved}")


if __name__ == "__main__":
    main()
