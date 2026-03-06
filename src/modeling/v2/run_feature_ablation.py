from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


FAMILY_CONFIG: dict[str, dict[str, Path]] = {
    "scoreline": {
        "trainer": ROOT_DIR / "src" / "modeling" / "v2" / "families" / "scoreline" / "train_scoreline.py",
        "contract": ROOT_DIR / "model_v2" / "feature_contracts" / "scoreline.yaml",
    },
    "corners": {
        "trainer": ROOT_DIR / "src" / "modeling" / "v2" / "families" / "corners" / "train_corners.py",
        "contract": ROOT_DIR / "model_v2" / "feature_contracts" / "corners.yaml",
    },
    "anytime": {
        "trainer": ROOT_DIR / "src" / "modeling" / "v2" / "families" / "anytime" / "train_anytime.py",
        "contract": ROOT_DIR / "model_v2" / "feature_contracts" / "anytime.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run drop-one feature ablation for v2 family trainers."
    )
    parser.add_argument(
        "--family",
        type=str,
        choices=tuple(FAMILY_CONFIG.keys()),
        required=True,
        help="v2 model family to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "artifacts" / "v2" / "ablation",
        help="Directory for ablation reports.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap forwarded to family trainer.",
    )
    parser.add_argument(
        "--max-drops",
        type=int,
        default=None,
        help="Optional cap on number of features to ablate.",
    )
    parser.add_argument(
        "--feature-regex",
        type=str,
        default=None,
        help="Optional regex to filter ablation candidates.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run trainer subprocesses.",
    )
    return parser.parse_args()


def _run_trainer(
    *,
    python_bin: str,
    trainer_path: Path,
    contract_path: Path,
    output_dir: Path,
    max_rows: int | None,
) -> tuple[bool, str]:
    cmd = [
        python_bin,
        str(trainer_path),
        "--contract",
        str(contract_path),
        "--output-dir",
        str(output_dir),
    ]
    if max_rows is not None and int(max_rows) > 0:
        cmd.extend(["--max-rows", str(int(max_rows))])
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout or "").strip()
    return True, (completed.stdout or "").strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_summary(metrics_rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    aucs = [float(row["auc"]) for row in metrics_rows if row.get("auc") is not None]
    briers = [float(row["brier"]) for row in metrics_rows if row.get("brier") is not None]
    return {
        "markets_scored": int(len(metrics_rows)),
        "auc_mean": float(sum(aucs) / len(aucs)) if aucs else None,
        "brier_mean": float(sum(briers) / len(briers)) if briers else None,
    }


def _write_minimal_contract(path: Path, family: str, features: list[str]) -> None:
    lines = [
        "version: 1",
        f"family: {family}",
        "required_features:",
    ]
    for feat in features:
        lines.append(f"  - {feat}")
    lines.extend(
        [
            "optional_features: []",
            "missingness_indicators: []",
            "forbidden_cross_family_features: []",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# V2 Feature Ablation Report")
    lines.append("")
    lines.append(f"- generated_at_utc: {report['generated_at_utc']}")
    lines.append(f"- family: {report['family']}")
    lines.append(f"- baseline_auc_mean: {report['baseline']['auc_mean']}")
    lines.append(f"- baseline_brier_mean: {report['baseline']['brier_mean']}")
    lines.append("")
    lines.append("| feature_dropped | auc_mean | auc_delta_vs_baseline | brier_mean | brier_delta_vs_baseline |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in report["results"]:
        lines.append(
            f"| {row['feature_dropped']} | {row['auc_mean']} | {row['auc_delta_vs_baseline']} | {row['brier_mean']} | {row['brier_delta_vs_baseline']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    family_cfg = FAMILY_CONFIG[str(args.family)]
    trainer_path = Path(family_cfg["trainer"])
    default_contract = Path(family_cfg["contract"])
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir / str(args.family) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"v2_ablation_{args.family}_") as td:
        tmp = Path(td)
        baseline_out = tmp / "baseline_out"
        ok, output = _run_trainer(
            python_bin=str(args.python_bin),
            trainer_path=trainer_path,
            contract_path=default_contract,
            output_dir=baseline_out,
            max_rows=args.max_rows,
        )
        if not ok:
            raise RuntimeError(f"Baseline trainer failed for {args.family}: {output}")

        baseline_metrics = _load_json(baseline_out / "metrics_holdout.json")
        baseline_summary = _metrics_summary(list(baseline_metrics))
        selected_features = list(_load_json(baseline_out / "features.json"))

        candidates = list(selected_features)
        if args.feature_regex:
            pattern = re.compile(str(args.feature_regex))
            candidates = [feat for feat in candidates if pattern.search(feat)]
        if args.max_drops is not None and int(args.max_drops) > 0:
            candidates = candidates[: int(args.max_drops)]

        results: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []

        for feature in candidates:
            kept = [feat for feat in selected_features if feat != feature]
            if not kept:
                continue
            variant_contract = tmp / f"contract_drop_{feature}.yaml"
            _write_minimal_contract(variant_contract, str(args.family), kept)
            variant_out = tmp / f"drop_{feature}"
            ok, output = _run_trainer(
                python_bin=str(args.python_bin),
                trainer_path=trainer_path,
                contract_path=variant_contract,
                output_dir=variant_out,
                max_rows=args.max_rows,
            )
            if not ok:
                failures.append({"feature_dropped": feature, "error": output})
                continue

            variant_metrics = _load_json(variant_out / "metrics_holdout.json")
            summary = _metrics_summary(list(variant_metrics))
            auc_base = baseline_summary.get("auc_mean")
            brier_base = baseline_summary.get("brier_mean")
            auc_variant = summary.get("auc_mean")
            brier_variant = summary.get("brier_mean")
            auc_delta = (
                float(auc_variant) - float(auc_base)
                if auc_variant is not None and auc_base is not None
                else None
            )
            brier_delta = (
                float(brier_variant) - float(brier_base)
                if brier_variant is not None and brier_base is not None
                else None
            )
            results.append(
                {
                    "feature_dropped": feature,
                    "auc_mean": auc_variant,
                    "auc_delta_vs_baseline": auc_delta,
                    "brier_mean": brier_variant,
                    "brier_delta_vs_baseline": brier_delta,
                    "markets_scored": summary.get("markets_scored"),
                }
            )

    results.sort(
        key=lambda row: (
            0 if row["auc_delta_vs_baseline"] is not None else 1,
            row["auc_delta_vs_baseline"] if row["auc_delta_vs_baseline"] is not None else 0.0,
        )
    )
    report = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "family": str(args.family),
        "baseline": baseline_summary,
        "candidate_count": int(len(candidates)),
        "results": results,
        "failures": failures,
    }

    json_path = out_dir / "ablation_report.json"
    md_path = out_dir / "ablation_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"Saved ablation report: {json_path}")
    print(f"Saved ablation report: {md_path}")
    print(
        f"Ablation completed family={args.family} candidates={len(candidates)} failures={len(failures)}"
    )


if __name__ == "__main__":
    main()

