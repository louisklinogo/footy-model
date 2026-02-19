"""
Evaluate saved daily predictions against final results for a specific date.

Usage:
  python models/evaluate_prediction_day.py 2026-02-17
"""

from pathlib import Path
import sys
import pandas as pd
import psycopg2
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_utils import connect_db


PRED_PATH = Path("data/daily/predictions_v2_clean.csv")
NON_SCORABLE_STATUSES = ("cancelled", "postponed", "abandoned")


def load_results(target_date: str) -> pd.DataFrame:
    conn = connect_db()
    query = """
    SELECT date, div AS league_code, home_team, away_team, fthg, ftag,
           CASE WHEN (fthg + ftag) >= 2 THEN 1 ELSE 0 END AS y_o15,
           CASE WHEN (fthg + ftag) >= 3 THEN 1 ELSE 0 END AS y_o25,
           CASE WHEN (hc + ac) >= 9 THEN 1 ELSE 0 END AS y_c85
    FROM matches
    WHERE date = %s AND fthg IS NOT NULL
    """
    df = pd.read_sql(query, conn, params=[target_date])
    conn.close()
    return df


def load_non_scorable_fixture_ids(target_date: str) -> set[int]:
    conn = connect_db()
    query = """
    SELECT id AS fixture_id
    FROM upcoming_fixtures
    WHERE match_date = %s
      AND status IN ('cancelled', 'postponed', 'abandoned')
    """
    df = pd.read_sql(query, conn, params=[target_date])
    conn.close()
    if df.empty:
        return set()
    return set(df["fixture_id"].astype(int).tolist())


def main() -> None:
    if len(sys.argv) < 2:
        print("Provide date as YYYY-MM-DD")
        sys.exit(1)

    target_date = sys.argv[1]
    if not PRED_PATH.exists():
        print(f"Predictions file not found: {PRED_PATH}")
        sys.exit(1)

    pred = pd.DataFrame(pd.read_csv(PRED_PATH))
    pred = pred.loc[pred["match_date"] == target_date].copy()
    if pred.empty:
        print(f"No predictions found for {target_date}")
        return

    non_scorable_fixture_ids = load_non_scorable_fixture_ids(target_date)
    if non_scorable_fixture_ids:
        non_scorable_ids = list(non_scorable_fixture_ids)
        pred["fixture_id"] = pd.to_numeric(pred["fixture_id"], errors="coerce")
        pred = pred.loc[pred["fixture_id"].notna()].copy()
        before = len(pred)
        pred = pred.loc[~pred["fixture_id"].isin(non_scorable_ids)].copy()
        invalidated = before - len(pred)
        if invalidated > 0:
            print(
                f"Invalidated {invalidated} predictions due to lifecycle status in {NON_SCORABLE_STATUSES}."
            )
    if pred.empty:
        print(f"No scorable predictions found for {target_date} after lifecycle invalidation.")
        return

    res = load_results(target_date)
    if res.empty:
        print(f"No completed results in matches table for {target_date} yet.")
        print(f"Predictions available: {len(pred)} fixtures")
        return

    merged = pred.merge(
        res,
        on=["league_code", "home_team", "away_team"],
        how="inner",
    )

    if merged.empty:
        print("No exact team-name matches between predictions and results.")
        return

    print(f"Evaluated fixtures: {len(merged)}")
    for name, p_col, y_col in [
        ("Over 1.5", "p_o15", "y_o15"),
        ("Over 2.5", "p_o25", "y_o25"),
        ("Corners 8.5", "p_c85", "y_c85"),
    ]:
        probs = merged[p_col].astype(float)
        y = merged[y_col].astype(int)
        pred_bin = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, pred_bin)
        brier = brier_score_loss(y, probs.clip(0, 1))
        auc = roc_auc_score(y, probs) if y.nunique() > 1 else float("nan")
        print(f"{name}: Acc={acc:.1%}, AUC={auc:.3f}, Brier={brier:.4f}")


if __name__ == "__main__":
    main()
