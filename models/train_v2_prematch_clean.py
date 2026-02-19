"""
Leakage-free pre-match training pipeline.

All model features are available before kickoff.
Evaluation is done on 25/26 holdout after training on older seasons.
"""

from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from elo_utils import add_pre_match_elo


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_utils import connect_db


OUT_DIR = Path("models/v2_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)


BASE_FEATURES = [
    "home_rolling_goals_scored_5",
    "home_rolling_goals_conceded_5",
    "home_rolling_sot_for_5",
    "home_rolling_sot_against_5",
    "home_rolling_corners_for_5",
    "home_rolling_corners_against_5",
    "home_rolling_pts_5",
    "home_attack_strength",
    "home_defense_strength",
    "home_days_since_last_match",
    "away_rolling_goals_scored_5",
    "away_rolling_goals_conceded_5",
    "away_rolling_sot_for_5",
    "away_rolling_sot_against_5",
    "away_rolling_corners_for_5",
    "away_rolling_corners_against_5",
    "away_rolling_pts_5",
    "away_attack_strength",
    "away_defense_strength",
    "away_days_since_last_match",
    "b365h",
    "b365d",
    "b365a",
    "b365_over25",
    "b365_under25",
    "psh",
    "psd",
    "psa",
    "avgh",
    "avgd",
    "avga",
    "b365_implied_h",
    "ps_implied_h",
    "sharp_diff_h",
    "market_move_h",
    "overround_b365",
    "overround_ps",
    "form_goals_diff",
    "form_sot_diff",
    "form_corners_diff",
    "form_pts_diff",
    "strength_diff",
    "relative_strength",
    "rest_diff",
    "implied_prob_home",
    "implied_prob_draw",
    "implied_prob_away",
    "implied_prob_over25",
    "odds_competitiveness",
    "favorite_confidence",
    "exp_goals_proxy",
    "exp_corners_proxy",
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_home_win_prob",
]


def fetch_dataset() -> pd.DataFrame:
    query = """
    SELECT
        m.id AS match_id,
        m.season,
        m.date,
        m.div AS league_code,
        m.home_team,
        m.away_team,
        m.fthg,
        m.ftag,
        m.hc,
        m.ac,
        m.b365h,
        m.b365d,
        m.b365a,
        m.b365_over25,
        m.b365_under25,
        m.psh,
        m.psd,
        m.psa,
        m.avgh,
        m.avgd,
        m.avga,
        mf.b365_implied_h,
        mf.ps_implied_h,
        mf.sharp_diff_h,
        mf.market_move_h,
        mf.overround_b365,
        mf.overround_ps,
        tms_h.rolling_goals_scored_5 AS home_rolling_goals_scored_5,
        tms_h.rolling_goals_conceded_5 AS home_rolling_goals_conceded_5,
        tms_h.rolling_sot_for_5 AS home_rolling_sot_for_5,
        tms_h.rolling_sot_against_5 AS home_rolling_sot_against_5,
        tms_h.rolling_corners_for_5 AS home_rolling_corners_for_5,
        tms_h.rolling_corners_against_5 AS home_rolling_corners_against_5,
        tms_h.rolling_pts_5 AS home_rolling_pts_5,
        tms_h.attack_strength AS home_attack_strength,
        tms_h.defense_strength AS home_defense_strength,
        tms_h.days_since_last_match AS home_days_since_last_match,
        tms_a.rolling_goals_scored_5 AS away_rolling_goals_scored_5,
        tms_a.rolling_goals_conceded_5 AS away_rolling_goals_conceded_5,
        tms_a.rolling_sot_for_5 AS away_rolling_sot_for_5,
        tms_a.rolling_sot_against_5 AS away_rolling_sot_against_5,
        tms_a.rolling_corners_for_5 AS away_rolling_corners_for_5,
        tms_a.rolling_corners_against_5 AS away_rolling_corners_against_5,
        tms_a.rolling_pts_5 AS away_rolling_pts_5,
        tms_a.attack_strength AS away_attack_strength,
        tms_a.defense_strength AS away_defense_strength,
        tms_a.days_since_last_match AS away_days_since_last_match
    FROM matches m
    LEFT JOIN team_match_snapshots tms_h ON m.id = tms_h.match_id AND tms_h.is_home = true
    LEFT JOIN team_match_snapshots tms_a ON m.id = tms_a.match_id AND tms_a.is_home = false
    LEFT JOIN market_features mf ON m.id = mf.match_id
    WHERE m.fthg IS NOT NULL
    ORDER BY m.date, m.id;
    """
    conn = connect_db()
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_o15"] = ((df["fthg"] + df["ftag"]) >= 2).astype(int)
    df["target_o25"] = ((df["fthg"] + df["ftag"]) >= 3).astype(int)
    df["target_c85"] = ((df["hc"] + df["ac"]) >= 9).astype(int)

    df["form_goals_diff"] = df["home_rolling_goals_scored_5"] - df["away_rolling_goals_conceded_5"]
    df["form_sot_diff"] = df["home_rolling_sot_for_5"] - df["away_rolling_sot_against_5"]
    df["form_corners_diff"] = df["home_rolling_corners_for_5"] - df["away_rolling_corners_against_5"]
    df["form_pts_diff"] = df["home_rolling_pts_5"] - df["away_rolling_pts_5"]

    df["strength_diff"] = df["home_attack_strength"] - df["away_defense_strength"]
    hr = df["home_attack_strength"] / df["away_defense_strength"].replace(0, np.nan)
    ar = df["away_attack_strength"] / df["home_defense_strength"].replace(0, np.nan)
    df["relative_strength"] = hr / ar.replace(0, np.nan)
    df["rest_diff"] = df["home_days_since_last_match"] - df["away_days_since_last_match"]

    df["implied_prob_home"] = np.where(df["b365h"] > 1, 1.0 / df["b365h"], np.nan)
    df["implied_prob_draw"] = np.where(df["b365d"] > 1, 1.0 / df["b365d"], np.nan)
    df["implied_prob_away"] = np.where(df["b365a"] > 1, 1.0 / df["b365a"], np.nan)
    df["implied_prob_over25"] = np.where(df["b365_over25"] > 1, 1.0 / df["b365_over25"], np.nan)
    df["odds_competitiveness"] = (df["b365h"] - df["b365a"]).abs()
    mmin = df[["b365h", "b365d", "b365a"]].min(axis=1)
    mmax = df[["b365h", "b365d", "b365a"]].max(axis=1)
    df["favorite_confidence"] = mmin / mmax.replace(0, np.nan)

    df["exp_goals_proxy"] = df["home_rolling_goals_scored_5"] + df["away_rolling_goals_scored_5"]
    df["exp_corners_proxy"] = df["home_rolling_corners_for_5"] + df["away_rolling_corners_for_5"]
    return df


def league_impute(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for feat in features:
        med = out.groupby("league_code")[feat].transform("median")
        out[feat] = out[feat].fillna(med)
        out[feat] = out[feat].fillna(out[feat].median())
        out[feat] = out[feat].fillna(0.0)
    return out


def build_imputation_artifacts(df: pd.DataFrame, features: list[str]) -> dict:
    artifacts = {
        "global_medians": {},
        "league_medians": {},
    }
    for feat in features:
        artifacts["global_medians"][feat] = float(df[feat].median()) if not df[feat].dropna().empty else 0.0
        league_series = df.groupby("league_code")[feat].median()
        artifacts["league_medians"][feat] = {
            str(k): (0.0 if pd.isna(v) else float(v)) for k, v in league_series.to_dict().items()
        }
    return artifacts


def train_eval_market(df: pd.DataFrame, target_col: str, label: str) -> dict:
    data = league_impute(df, BASE_FEATURES)
    train = data[data["season"] != "2526"].copy()
    test = data[data["season"] == "2526"].copy()

    X_train = train[BASE_FEATURES]
    y_train = train[target_col]
    X_test = test[BASE_FEATURES]
    y_test = test[target_col]

    model = CalibratedClassifierCV(
        estimator=GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=42,
        ),
        method="isotonic",
        cv=3,
    )
    model.fit(X_train, y_train)

    prob = np.clip(model.predict_proba(X_test)[:, 1], 0.0, 0.92)
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "market": label,
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "base_rate": float(y_test.mean()),
        "auc": float(roc_auc_score(y_test, prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "brier": float(brier_score_loss(y_test, prob)),
    }

    # Simple value filter using b365_over25 as available odds proxy.
    odds = test["b365_over25"].astype(float)
    valid = (odds > 1) & (~odds.isna())
    edge = prob - (1.0 / odds.where(valid, np.nan))
    bet_mask = valid & (prob >= 0.72) & (edge > 0.03)
    if bet_mask.sum() > 0:
        wins = (y_test[bet_mask] == 1).sum()
        profits = np.where(y_test[bet_mask] == 1, odds[bet_mask] - 1.0, -1.0)
        metrics["bets"] = int(bet_mask.sum())
        metrics["win_rate"] = float(wins / bet_mask.sum())
        metrics["roi"] = float(profits.mean())
    else:
        metrics["bets"] = 0
        metrics["win_rate"] = 0.0
        metrics["roi"] = 0.0

    joblib.dump(model, OUT_DIR / f"gbm_{target_col}.pkl")
    return metrics


def main() -> None:
    df = fetch_dataset()
    df = add_pre_match_elo(df)
    df = add_derived(df)

    impute_artifacts = build_imputation_artifacts(df, BASE_FEATURES)
    with open(OUT_DIR / "imputation.json", "w", encoding="utf-8") as f:
        json.dump(impute_artifacts, f, indent=2)

    markets = [
        ("target_o15", "Over 1.5 Goals"),
        ("target_o25", "Over 2.5 Goals"),
        ("target_c85", "Over 8.5 Corners"),
    ]

    results = []
    for target_col, label in markets:
        metrics = train_eval_market(df, target_col, label)
        results.append(metrics)
        print(
            f"{label}: AUC={metrics['auc']:.3f}, Acc={metrics['accuracy']:.1%}, "
            f"Brier={metrics['brier']:.4f}, ROI={metrics['roi']:+.2%}, Bets={metrics['bets']}"
        )

    with open(OUT_DIR / "metrics_2526_holdout.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(OUT_DIR / "features.json", "w", encoding="utf-8") as f:
        json.dump(BASE_FEATURES, f, indent=2)

    print(f"Saved models and reports to {OUT_DIR}")


if __name__ == "__main__":
    main()
