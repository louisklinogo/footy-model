"""
Diagnose scored prediction quality by market and operational slices.

Outputs JSON and Markdown reports under artifacts/reports/v2_auc_diagnostics/.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


DEFAULT_MODEL = "market_outcome_gbm"
DEFAULT_VERSION = "fixtures_first_prematch_v1"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "reports" / "v2_auc_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose scored prediction quality by slices."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help="Model version.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Scored fixture lookback window in days.",
    )
    parser.add_argument(
        "--league",
        type=str,
        default=None,
        help="Optional league filter.",
    )
    parser.add_argument(
        "--market",
        action="append",
        default=None,
        help="Optional market filter. Repeat for multiple markets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=120,
        help="Minimum samples for weak-segment shortlist.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top weak segments to include per view.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory.",
    )
    return parser.parse_args()


def _clip_probability(value: Any) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return 0.5
    if math.isnan(p):
        return 0.5
    return float(min(max(p, 1e-6), 1.0 - 1e-6))


def _bucket_odds(odds_used: float | None) -> str:
    if odds_used is None or math.isnan(odds_used):
        return "missing"
    if odds_used <= 1.5:
        return "<=1.50"
    if odds_used <= 1.8:
        return "1.51-1.80"
    if odds_used <= 2.1:
        return "1.81-2.10"
    if odds_used <= 2.6:
        return "2.11-2.60"
    if odds_used <= 3.5:
        return "2.61-3.50"
    return ">3.50"


def _bucket_kickoff_hours(hours_to_kickoff: float | None) -> str:
    if hours_to_kickoff is None or math.isnan(hours_to_kickoff):
        return "missing"
    if hours_to_kickoff <= 6.0:
        return "<=6h"
    if hours_to_kickoff <= 24.0:
        return "6-24h"
    if hours_to_kickoff <= 72.0:
        return "24-72h"
    if hours_to_kickoff <= 168.0:
        return "72-168h"
    return ">168h"


def _bucket_edge(edge: float | None) -> str:
    if edge is None or math.isnan(edge):
        return "missing"
    if edge < 0.0:
        return "<0%"
    if edge < 0.02:
        return "0-2%"
    if edge < 0.04:
        return "2-4%"
    if edge < 0.06:
        return "4-6%"
    return "6%+"


def _compute_ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float | None:
    if y_true.size == 0 or p_pred.size == 0:
        return None
    if y_true.size != p_pred.size:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_pred, bins[1:-1], right=True)
    ece = 0.0
    total = float(y_true.size)
    for i in range(n_bins):
        mask = idx == i
        count = int(mask.sum())
        if count <= 0:
            continue
        conf = float(p_pred[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (count / total) * abs(acc - conf)
    return float(ece)


def _safe_auc(y_true: np.ndarray, p_pred: np.ndarray) -> float | None:
    if y_true.size <= 1:
        return None
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    try:
        return float(roc_auc_score(y_true, p_pred))
    except ValueError:
        return None


def fetch_scored_rows(
    *,
    model: str,
    version: str,
    days: int,
    league: str | None,
    markets: list[str] | None,
    limit: int | None,
) -> pd.DataFrame:
    query = """
    SELECT
        p.prediction_id,
        p.fixture_id,
        p.market_code,
        p.model_name,
        p.model_version,
        COALESCE(p.p_final, p.p_model) AS p_pred,
        p.created_at AS prediction_created_at,
        f.league_code,
        f.match_datetime_utc,
        ps.actual,
        ps.brier,
        ps.edge,
        ps.odds_used
    FROM predictions p
    JOIN prediction_scores ps
      ON ps.prediction_id = p.prediction_id
    JOIN fixtures f
      ON f.fixture_id = p.fixture_id
    WHERE p.model_name = %s
      AND p.model_version = %s
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
    """
    params: list[object] = [model, version, days]

    if league:
        query += " AND f.league_code = %s"
        params.append(league)
    if markets:
        query += " AND p.market_code = ANY(%s)"
        params.append(markets)

    query += " ORDER BY f.match_datetime_utc DESC, p.prediction_id DESC"
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            cols = [desc[0] for desc in (cur.description or [])]
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    return df


def _compute_group_metrics(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        payload: dict[str, Any] = {}
        for col, value in zip(group_cols, key):
            payload[col] = value

        y = pd.to_numeric(group["actual"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        p = pd.to_numeric(group["p_pred"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        brier = pd.to_numeric(group["brier"], errors="coerce")
        edge = pd.to_numeric(group["edge"], errors="coerce")
        odds = pd.to_numeric(group["odds_used"], errors="coerce")
        kickoff = pd.to_numeric(group["hours_to_kickoff"], errors="coerce")

        n = int(len(group))
        positives = int(y.sum())
        negatives = int(n - positives)
        payload.update(
            {
                "n": n,
                "positives": positives,
                "negatives": negatives,
                "positive_rate": float(y.mean()) if n > 0 else None,
                "p_pred_mean": float(p.mean()) if n > 0 else None,
                "auc": _safe_auc(y, p),
                "brier_mean": (
                    float(brier.mean()) if brier.notna().any() else None
                ),
                "ece": _compute_ece(y_true=y, p_pred=p, n_bins=10),
                "edge_mean": float(edge.mean()) if edge.notna().any() else None,
                "odds_mean": float(odds.mean()) if odds.notna().any() else None,
                "kickoff_hours_mean": (
                    float(kickoff.mean()) if kickoff.notna().any() else None
                ),
            }
        )
        rows.append(payload)
    return rows


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def _serialize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({k: _to_serializable(v) for k, v in row.items()})
    return out


def _weak_segments(
    rows: list[dict[str, Any]],
    *,
    min_samples: int,
    top_k: int,
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        n = int(row.get("n") or 0)
        pos = int(row.get("positives") or 0)
        neg = int(row.get("negatives") or 0)
        auc = row.get("auc")
        if n < min_samples or pos < 10 or neg < 10:
            continue
        if not isinstance(auc, (int, float)):
            continue
        filtered.append(row)
    filtered.sort(
        key=lambda r: (
            float(r.get("auc", 1.0)),
            -int(r.get("n", 0)),
            -float(r.get("brier_mean", 0.0) or 0.0),
        )
    )
    return filtered[:top_k]


def _render_table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    lines: list[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        values = []
        for col in cols:
            val = row.get(col)
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Prediction Slice Diagnostics")
    lines.append("")
    lines.append(f"- generated_at_utc: {report['generated_at_utc']}")
    lines.append(f"- model_name: {report['model_name']}")
    lines.append(f"- model_version: {report['model_version']}")
    lines.append(f"- days: {report['days']}")
    lines.append(f"- row_count: {report['row_count']}")
    lines.append(
        "- overall: "
        f"auc={report['overall']['auc']}, "
        f"brier={report['overall']['brier_mean']}, "
        f"ece={report['overall']['ece']}"
    )
    lines.append("")

    for view_name, view in report["views"].items():
        lines.append(f"## {view_name}")
        lines.append("")
        lines.append(f"- segments: {view['segment_count']}")
        lines.append(f"- weak_shortlist (n>={report['min_samples']}): {len(view['weak_segments'])}")
        lines.append("")
        cols = [col for col in view["group_cols"]] + [
            "n",
            "positives",
            "auc",
            "brier_mean",
            "ece",
        ]
        lines.extend(_render_table(view["weak_segments"], cols))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    df = fetch_scored_rows(
        model=str(args.model),
        version=str(args.version),
        days=int(args.days),
        league=args.league,
        markets=list(args.market) if args.market else None,
        limit=args.limit,
    )
    if df.empty:
        print("No scored predictions found for requested filters.")
        return

    df["p_pred"] = df["p_pred"].map(_clip_probability)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce").fillna(0.0)
    df["prediction_created_at"] = pd.to_datetime(
        df["prediction_created_at"], utc=True, errors="coerce"
    )
    df["match_datetime_utc"] = pd.to_datetime(
        df["match_datetime_utc"], utc=True, errors="coerce"
    )
    delta = df["match_datetime_utc"] - df["prediction_created_at"]
    df["hours_to_kickoff"] = delta.dt.total_seconds() / 3600.0
    df["odds_band"] = pd.to_numeric(df["odds_used"], errors="coerce").map(_bucket_odds)
    df["kickoff_band"] = pd.to_numeric(df["hours_to_kickoff"], errors="coerce").map(
        _bucket_kickoff_hours
    )
    df["edge_band"] = pd.to_numeric(df["edge"], errors="coerce").map(_bucket_edge)

    y_all = df["actual"].to_numpy(dtype=float)
    p_all = df["p_pred"].to_numpy(dtype=float)
    overall = {
        "auc": _safe_auc(y_all, p_all),
        "brier_mean": float(pd.to_numeric(df["brier"], errors="coerce").mean()),
        "ece": _compute_ece(y_true=y_all, p_pred=p_all, n_bins=10),
        "positive_rate": float(y_all.mean()),
    }

    views_spec: dict[str, list[str]] = {
        "market": ["market_code"],
        "league": ["league_code"],
        "market_league": ["market_code", "league_code"],
        "market_odds_band": ["market_code", "odds_band"],
        "market_kickoff_band": ["market_code", "kickoff_band"],
        "market_edge_band": ["market_code", "edge_band"],
    }

    views: dict[str, dict[str, Any]] = {}
    for view_name, cols in views_spec.items():
        rows = _compute_group_metrics(df, cols)
        weak = _weak_segments(
            rows,
            min_samples=int(args.min_samples),
            top_k=int(args.top_k),
        )
        views[view_name] = {
            "group_cols": cols,
            "segment_count": len(rows),
            "rows": _serialize_rows(rows),
            "weak_segments": _serialize_rows(weak),
        }

    report = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "model_name": str(args.model),
        "model_version": str(args.version),
        "days": int(args.days),
        "league_filter": args.league,
        "market_filter": list(args.market) if args.market else None,
        "row_count": int(len(df)),
        "min_samples": int(args.min_samples),
        "top_k": int(args.top_k),
        "overall": _serialize_rows([overall])[0],
        "views": views,
    }

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "diagnostic_report.json"
    md_path = run_dir / "diagnostic_report.md"
    csv_path = run_dir / "scored_rows_snapshot.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    df.to_csv(csv_path, index=False)

    print(f"Saved report: {json_path}")
    print(f"Saved report: {md_path}")
    print(f"Saved data snapshot: {csv_path}")
    print(
        "Overall: "
        f"rows={report['row_count']} "
        f"auc={report['overall']['auc']} "
        f"brier={report['overall']['brier_mean']} "
        f"ece={report['overall']['ece']}"
    )


if __name__ == "__main__":
    main()
