"""
Inference for leakage-free pre-match models.

Uses the exact feature schema from train_v2_prematch_clean.py and
applies saved league/global imputation artifacts for feature parity.
"""

from pathlib import Path
from datetime import date
import json
import sys
import joblib
import numpy as np
import pandas as pd
import psycopg2

from models.elo_utils import ELO_BASE, build_latest_elo_ratings, expected_from_ratings


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_utils import connect_db


MODEL_DIR = Path("models/v2_clean")
OUT_PATH = Path("data/daily/predictions_v2_clean.csv")


def load_artifacts():
    with open(MODEL_DIR / "features.json", "r", encoding="utf-8") as f:
        features = json.load(f)
    with open(MODEL_DIR / "imputation.json", "r", encoding="utf-8") as f:
        impute = json.load(f)
    models = {
        "target_o15": joblib.load(MODEL_DIR / "gbm_target_o15.pkl"),
        "target_o25": joblib.load(MODEL_DIR / "gbm_target_o25.pkl"),
        "target_c85": joblib.load(MODEL_DIR / "gbm_target_c85.pkl"),
    }
    return features, impute, models


def fetch_upcoming(days: int = 3) -> pd.DataFrame:
    query = f"""
    SELECT
      id AS fixture_id,
      league_code,
      league_name,
      home_team,
      away_team,
      match_date,
      match_time,
      match_datetime_utc
    FROM upcoming_fixtures
    WHERE status = 'scheduled'
      AND match_datetime_utc IS NOT NULL
      AND match_datetime_utc > NOW()
      AND match_datetime_utc <= NOW() + INTERVAL '{days} days'
    ORDER BY match_datetime_utc, id;
    """
    conn = connect_db()
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def fetch_completed_for_elo() -> pd.DataFrame:
    query = """
    SELECT date, id, home_team, away_team, fthg, ftag
    FROM matches
    WHERE fthg IS NOT NULL
    ORDER BY date, id;
    """
    conn = connect_db()
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def fetch_latest_team_snapshot(conn, team_name: str, cutoff_date: date):
    query = """
    SELECT
      m.div AS league_code,
      tms.rolling_goals_scored_5,
      tms.rolling_goals_conceded_5,
      tms.rolling_sot_for_5,
      tms.rolling_sot_against_5,
      tms.rolling_corners_for_5,
      tms.rolling_corners_against_5,
      tms.rolling_pts_5,
      tms.attack_strength,
      tms.defense_strength,
      tms.days_since_last_match
    FROM matches m
    JOIN team_match_snapshots tms ON m.id = tms.match_id
    WHERE (m.home_team = %s OR m.away_team = %s)
      AND m.date < %s
      AND m.fthg IS NOT NULL
    ORDER BY m.date DESC, m.id DESC
    LIMIT 1;
    """
    cur = conn.cursor()
    cur.execute(query, (team_name, team_name, cutoff_date))
    row = cur.fetchone()
    cols = [d[0] for d in cur.description] if cur.description else []
    cur.close()
    if not row:
        return None
    return dict(zip(cols, row))


def build_feature_row(fix_row, home_snap, away_snap, elo_ratings):
    row = {
        "fixture_id": fix_row["fixture_id"],
        "league_code": fix_row["league_code"],
        "league_name": fix_row["league_name"],
        "home_team": fix_row["home_team"],
        "away_team": fix_row["away_team"],
        "match_date": fix_row["match_date"],
        "match_time": fix_row["match_time"],
        # Odds placeholders (imputed later if unavailable)
        "b365h": np.nan,
        "b365d": np.nan,
        "b365a": np.nan,
        "b365_over25": np.nan,
        "b365_under25": np.nan,
        "psh": np.nan,
        "psd": np.nan,
        "psa": np.nan,
        "avgh": np.nan,
        "avgd": np.nan,
        "avga": np.nan,
        "b365_implied_h": np.nan,
        "ps_implied_h": np.nan,
        "sharp_diff_h": np.nan,
        "market_move_h": np.nan,
        "overround_b365": np.nan,
        "overround_ps": np.nan,
    }

    home_map = {
        "rolling_goals_scored_5": "home_rolling_goals_scored_5",
        "rolling_goals_conceded_5": "home_rolling_goals_conceded_5",
        "rolling_sot_for_5": "home_rolling_sot_for_5",
        "rolling_sot_against_5": "home_rolling_sot_against_5",
        "rolling_corners_for_5": "home_rolling_corners_for_5",
        "rolling_corners_against_5": "home_rolling_corners_against_5",
        "rolling_pts_5": "home_rolling_pts_5",
        "attack_strength": "home_attack_strength",
        "defense_strength": "home_defense_strength",
        "days_since_last_match": "home_days_since_last_match",
    }
    away_map = {
        "rolling_goals_scored_5": "away_rolling_goals_scored_5",
        "rolling_goals_conceded_5": "away_rolling_goals_conceded_5",
        "rolling_sot_for_5": "away_rolling_sot_for_5",
        "rolling_sot_against_5": "away_rolling_sot_against_5",
        "rolling_corners_for_5": "away_rolling_corners_for_5",
        "rolling_corners_against_5": "away_rolling_corners_against_5",
        "rolling_pts_5": "away_rolling_pts_5",
        "attack_strength": "away_attack_strength",
        "defense_strength": "away_defense_strength",
        "days_since_last_match": "away_days_since_last_match",
    }

    for k, out_k in home_map.items():
        row[out_k] = np.nan if home_snap is None else home_snap.get(k, np.nan)
    for k, out_k in away_map.items():
        row[out_k] = np.nan if away_snap is None else away_snap.get(k, np.nan)

    h_elo = float(elo_ratings.get(row["home_team"], ELO_BASE))
    a_elo = float(elo_ratings.get(row["away_team"], ELO_BASE))
    row["home_elo"] = h_elo
    row["away_elo"] = a_elo
    row["elo_diff"] = h_elo - a_elo
    row["elo_home_win_prob"] = expected_from_ratings(h_elo, a_elo)

    # Derived features
    row["form_goals_diff"] = row["home_rolling_goals_scored_5"] - row["away_rolling_goals_conceded_5"]
    row["form_sot_diff"] = row["home_rolling_sot_for_5"] - row["away_rolling_sot_against_5"]
    row["form_corners_diff"] = row["home_rolling_corners_for_5"] - row["away_rolling_corners_against_5"]
    row["form_pts_diff"] = row["home_rolling_pts_5"] - row["away_rolling_pts_5"]
    row["strength_diff"] = row["home_attack_strength"] - row["away_defense_strength"]

    hr = row["home_attack_strength"] / (row["away_defense_strength"] if pd.notna(row["away_defense_strength"]) and row["away_defense_strength"] != 0 else np.nan)
    ar = row["away_attack_strength"] / (row["home_defense_strength"] if pd.notna(row["home_defense_strength"]) and row["home_defense_strength"] != 0 else np.nan)
    row["relative_strength"] = hr / ar if pd.notna(hr) and pd.notna(ar) and ar != 0 else np.nan

    row["rest_diff"] = row["home_days_since_last_match"] - row["away_days_since_last_match"]
    row["implied_prob_home"] = np.nan
    row["implied_prob_draw"] = np.nan
    row["implied_prob_away"] = np.nan
    row["implied_prob_over25"] = np.nan
    row["odds_competitiveness"] = np.nan
    row["favorite_confidence"] = np.nan
    row["exp_goals_proxy"] = row["home_rolling_goals_scored_5"] + row["away_rolling_goals_scored_5"]
    row["exp_corners_proxy"] = row["home_rolling_corners_for_5"] + row["away_rolling_corners_for_5"]
    return row


def apply_imputation(df: pd.DataFrame, features, impute):
    out = df.copy()
    gmed = impute["global_medians"]
    lmed = impute["league_medians"]
    for feat in features:
        if feat not in out.columns:
            out[feat] = np.nan
        med_map = lmed.get(feat, {})
        out[feat] = out.apply(
            lambda r: med_map.get(str(r["league_code"]), np.nan) if pd.isna(r[feat]) else r[feat],
            axis=1,
        )
        out[feat] = out[feat].fillna(gmed.get(feat, 0.0))
        out[feat] = out[feat].fillna(0.0)
    return out


def main():
    features, impute, models = load_artifacts()
    upcoming = fetch_upcoming(days=3)
    if upcoming.empty:
        print("No upcoming fixtures found.")
        return

    completed = fetch_completed_for_elo()
    elo_ratings = build_latest_elo_ratings(completed)

    conn = connect_db()
    rows = []
    for _, f in upcoming.iterrows():
        home_team = str(f["home_team"])
        away_team = str(f["away_team"])
        cutoff_date = date.fromisoformat(str(f["match_date"]))
        home_snap = fetch_latest_team_snapshot(conn, home_team, cutoff_date)
        away_snap = fetch_latest_team_snapshot(conn, away_team, cutoff_date)
        rows.append(build_feature_row(f, home_snap, away_snap, elo_ratings))
    conn.close()

    feats = pd.DataFrame(rows)
    feats = apply_imputation(feats, features, impute)
    X = feats[features]

    feats["p_o15"] = np.clip(models["target_o15"].predict_proba(X)[:, 1], 0.0, 0.92)
    feats["p_o25"] = np.clip(models["target_o25"].predict_proba(X)[:, 1], 0.0, 0.92)
    feats["p_c85"] = np.clip(models["target_c85"].predict_proba(X)[:, 1], 0.0, 0.92)

    out_cols = [
        "fixture_id",
        "match_date",
        "match_time",
        "league_name",
        "league_code",
        "home_team",
        "away_team",
        "p_o15",
        "p_o25",
        "p_c85",
    ]
    output = pd.DataFrame(feats[out_cols]).sort_values(
        by=["match_date", "match_time", "p_o15"],
        ascending=[True, True, False],
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT_PATH, index=False)
    print(f"Saved predictions to {OUT_PATH} ({len(output)} fixtures)")


if __name__ == "__main__":
    main()
