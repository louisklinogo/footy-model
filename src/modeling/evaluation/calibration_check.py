"""
Calibration check for model predictions.

Computes Expected Calibration Error (ECE) and generates reliability diagrams
to assess whether predicted probabilities match actual hit rates.

Usage:
    python src/modeling/evaluation/calibration_check.py --model market_outcome_gbm --version fixtures_first_prematch_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false


N_BINS = 10


def fetch_scored_predictions(
    model_name: str,
    model_version: str,
    market: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    conn = connect_db()
    query = """
    SELECT
        p.fixture_id,
        p.market_code,
        p.p_model,
        ps.actual,
        ps.brier
    FROM predictions p
    JOIN prediction_scores ps ON ps.prediction_id = p.prediction_id
    WHERE p.model_name = %s
      AND p.model_version = %s
    """
    params: list = [model_name, model_version]

    if market:
        query += " AND p.market_code = %s"
        params.append(market)

    query += " ORDER BY p.fixture_id ASC"

    if limit:
        query += " LIMIT %s"
        params.append(limit)

    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

    conn.close()
    return [dict(zip(cols, row)) for row in rows]


def compute_ece(predictions: list[dict], n_bins: int = N_BINS) -> tuple[float, dict]:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum(bin_count / total) * |bin_accuracy - bin_confidence|

    Returns: (ece_value, bin_statistics)
    """
    if not predictions:
        return 0.0, {}

    probs = np.array([p["p_model"] for p in predictions])
    actuals = np.array([p["actual"] for p in predictions])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges[1:-1], right=True)

    ece = 0.0
    bin_stats = {}

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            bin_stats[i] = {
                "count": 0,
                "confidence": 0.0,
                "accuracy": 0.0,
                "gap": 0.0,
            }
            continue

        bin_probs = probs[mask]
        bin_actuals = actuals[mask]

        confidence = bin_probs.mean()
        accuracy = bin_actuals.mean()
        count = mask.sum()

        gap = abs(accuracy - confidence)
        ece += (count / len(predictions)) * gap

        bin_stats[i] = {
            "count": int(count),
            "confidence": float(confidence),
            "accuracy": float(accuracy),
            "gap": float(gap),
        }

    return ece, bin_stats


def plot_reliability_diagram(
    bin_stats: dict,
    title: str,
    output_path: Path | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping reliability diagram.")
        return
    bin_centers = [(i + 0.5) / N_BINS for i in range(N_BINS)]
    accuracies = [bin_stats.get(i, {}).get("accuracy", 0) for i in range(N_BINS)]
    confidences = [bin_stats.get(i, {}).get("confidence", 0) for i in range(N_BINS)]
    counts = [bin_stats.get(i, {}).get("count", 0) for i in range(N_BINS)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(
        bin_centers,
        accuracies,
        width=1 / N_BINS * 0.8,
        alpha=0.7,
        label="Accuracy",
        color="steelblue",
    )
    ax.scatter(bin_centers, confidences, color="red", s=50, label="Confidence", zorder=5)

    ax.set_xlabel("Confidence (predicted probability)")
    ax.set_ylabel("Accuracy (observed hit rate)")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for i, (bc, acc, cnt) in enumerate(zip(bin_centers, accuracies, counts)):
        if cnt > 0:
            ax.annotate(
                f"n={cnt}",
                (bc, acc),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration check for model predictions")
    parser.add_argument("--model", type=str, default="market_outcome_gbm", help="Model name")
    parser.add_argument("--version", type=str, default="fixtures_first_prematch_v1", help="Model version")
    parser.add_argument("--market", type=str, default=None, help="Filter by market")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of predictions")
    parser.add_argument("--output-dir", type=Path, default=Path("data/v1/calibration"), help="Output directory")
    args = parser.parse_args()

    print(f"Fetching scored predictions for {args.model} ({args.version})...")
    predictions = fetch_scored_predictions(
        model_name=args.model,
        model_version=args.version,
        market=args.market,
        limit=args.limit,
    )

    if not predictions:
        print("No scored predictions found.")
        return

    print(f"Found {len(predictions)} scored predictions")

    if args.market:
        markets = [args.market]
    else:
        markets = sorted(set(p["market_code"] for p in predictions))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for market in markets:
        market_preds = [p for p in predictions if p["market_code"] == market]
        if len(market_preds) < 50:
            print(f"  [SKIP] {market}: only {len(market_preds)} predictions")
            continue

        ece, bin_stats = compute_ece(market_preds)
        results[market] = {
            "n": len(market_preds),
            "ece": ece,
            "bins": bin_stats,
        }

        print(f"  {market}: n={len(market_preds)}, ECE={ece:.4f}")

        diagram_path = args.output_dir / f"reliability_{market}.png"
        plot_reliability_diagram(
            bin_stats,
            title=f"Calibration: {market} (ECE={ece:.4f})",
            output_path=diagram_path,
        )

    results_path = args.output_dir / "calibration_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print("\n=== Summary ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["ece"], reverse=True)
    for market, data in sorted_results[:10]:
        print(f"  {market}: ECE={data['ece']:.4f} (n={data['n']})")


if __name__ == "__main__":
    main()
