"""
Unified runner for the clean pre-match model stack.

Commands:
  train      Train clean models + write holdout metrics
  predict    Score upcoming fixtures and save daily prediction CSV
  eval-day   Evaluate saved predictions vs completed results for a date
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path("models")


def run_cmd(parts: list[str]) -> int:
    proc = subprocess.run(parts)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v2 clean pipeline tasks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Train clean models and save holdout metrics")
    sub.add_parser("predict", help="Generate upcoming predictions")

    eval_day = sub.add_parser("eval-day", help="Evaluate one prediction day")
    eval_day.add_argument("--date", required=True, help="Date in YYYY-MM-DD")

    args = parser.parse_args()

    if args.cmd == "train":
        return run_cmd([sys.executable, str(ROOT / "train_v2_prematch_clean.py")])
    if args.cmd == "predict":
        return run_cmd([sys.executable, str(ROOT / "predict_v2_prematch_clean.py")])
    if args.cmd == "eval-day":
        return run_cmd([sys.executable, str(ROOT / "evaluate_prediction_day.py"), args.date])

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
