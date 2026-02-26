from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.layer2_situational.rule_layer import load_rule_layer_config, save_rule_layer_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep key-absence-only deterministic rule-layer configs and rank by "
            "triggered-segment improvement."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model_artifacts/situational_model"),
        help="Layer 2 model artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation"),
        help="Directory for sweep artifacts.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="layer2_rule_layer_key_absent_sweep",
        help="Output file stem.",
    )
    parser.add_argument(
        "--home-pcts",
        type=str,
        default="-0.03,-0.05,-0.07,-0.08,-0.10,-0.12",
        help="Comma-separated home key_absent adjustment percentages to test.",
    )
    parser.add_argument(
        "--max-down-pct",
        type=float,
        default=0.15,
        help="Max downward cap for tested variants.",
    )
    parser.add_argument(
        "--promote-best",
        action="store_true",
        help="Promote best passing variant into model_dir/rule_layer_config.json.",
    )
    return parser.parse_args()


def _parse_float_csv(raw: str) -> list[float]:
    out: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise ValueError("Expected at least one value in --home-pcts.")
    return out


def _base_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    configs_dir = args.output_dir / f"{args.output_stem}_configs"
    runs_dir = args.output_dir / f"{args.output_stem}_runs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir, runs_dir


def _run_backtest(
    model_dir: Path,
    config_path: Path,
    output_dir: Path,
    stem: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT_DIR / "src/modeling/layer2_situational/backtest_rule_layer_overrides.py"),
        "--model-dir",
        str(model_dir),
        "--rule-layer-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--output-stem",
        stem,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Backtest failed for {config_path.name}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    payload_path = output_dir / f"{stem}.json"
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not parse backtest payload: {payload_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected payload shape in {payload_path}")
    return payload


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float | int]:
    global_metrics = payload.get("global_metrics", {})
    triggered_metrics = payload.get("triggered_metrics", {})
    directional = payload.get("directional_metrics", {})
    summary = payload.get("summary", {})

    l2_global = global_metrics.get("layer1_plus_layer2", {})
    l2r_global = global_metrics.get("layer1_plus_layer2_plus_rules", {})
    l2_trig = triggered_metrics.get("layer1_plus_layer2", {})
    l2r_trig = triggered_metrics.get("layer1_plus_layer2_plus_rules", {})
    home_dir = directional.get("home_trigger_events", {})

    def _f(d: dict[str, Any], k: str) -> float:
        try:
            return float(d.get(k, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _i(d: dict[str, Any], k: str) -> int:
        try:
            return int(d.get(k, 0))
        except (TypeError, ValueError):
            return 0

    return {
        "triggered_fixtures": _i(summary, "n_triggered_fixtures"),
        "triggered_home_events": _i(summary, "n_triggered_home_events"),
        "triggered_away_events": _i(summary, "n_triggered_away_events"),
        "global_home_lift_l2": _f(l2_global, "home_lift"),
        "global_away_lift_l2": _f(l2_global, "away_lift"),
        "global_home_lift_rules": _f(l2r_global, "home_lift"),
        "global_away_lift_rules": _f(l2r_global, "away_lift"),
        "triggered_home_lift_l2": _f(l2_trig, "home_lift"),
        "triggered_away_lift_l2": _f(l2_trig, "away_lift"),
        "triggered_home_lift_rules": _f(l2r_trig, "home_lift"),
        "triggered_away_lift_rules": _f(l2r_trig, "away_lift"),
        "home_hit_rate_l2": _f(home_dir, "layer1_plus_layer2_hit_rate"),
        "home_hit_rate_rules": _f(home_dir, "layer1_plus_layer2_plus_rules_hit_rate"),
    }


def _build_reference_row(reference_payload: dict[str, Any]) -> dict[str, Any]:
    ref_metrics = _extract_metrics(reference_payload)
    row: dict[str, Any] = {
        "variant": "home_only_v1_reference",
        "config_path": "",
        "config_version": "v1_home_only",
        "home_key_absent_pct": "",
        "away_key_absent_pct": "",
        "max_down_pct": "",
        "rule_family": "reference",
        **ref_metrics,
    }
    row["delta_global_home_vs_l2"] = (
        row["global_home_lift_rules"] - row["global_home_lift_l2"]
    )
    row["delta_global_away_vs_l2"] = (
        row["global_away_lift_rules"] - row["global_away_lift_l2"]
    )
    row["delta_triggered_home_vs_l2"] = (
        row["triggered_home_lift_rules"] - row["triggered_home_lift_l2"]
    )
    row["delta_triggered_away_vs_l2"] = (
        row["triggered_away_lift_rules"] - row["triggered_away_lift_l2"]
    )
    row["delta_home_hit_rate_vs_l2"] = row["home_hit_rate_rules"] - row["home_hit_rate_l2"]
    row["gate_pass"] = True
    return row


def _format_pct(value: float) -> str:
    return f"{value:+.2%}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            bool(r.get("gate_pass", False)),
            float(r.get("delta_triggered_home_vs_l2", 0.0)),
            float(r.get("delta_global_home_vs_l2", 0.0)),
            float(r.get("delta_home_hit_rate_vs_l2", 0.0)),
        ),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        raise RuntimeError(f"Model directory not found: {model_dir}")

    home_pcts = _parse_float_csv(args.home_pcts)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir, runs_dir = _base_paths(args)

    active_cfg_path = model_dir / "rule_layer_config.json"
    active_cfg = load_rule_layer_config(active_cfg_path)

    reference_payload = _run_backtest(
        model_dir=model_dir,
        config_path=active_cfg_path,
        output_dir=runs_dir,
        stem=f"{args.output_stem}_reference_home_only_v1",
    )
    results: list[dict[str, Any]] = [_build_reference_row(reference_payload)]

    for home_pct in home_pcts:
        cfg = dict(active_cfg)
        cfg["config_version"] = f"keyabs_home_only_{home_pct:+.3f}".replace("+", "plus").replace(
            "-", "minus"
        )
        cfg["key_absent_pct"] = float(home_pct)
        cfg["key_absent_pct_home"] = float(home_pct)
        cfg["key_absent_pct_away"] = 0.0
        cfg["upcoming_tier_2_pct"] = 0.0
        cfg["upcoming_tier_3_pct"] = 0.0
        cfg["congestion_pct"] = 0.0
        cfg["congestion_pct_home"] = 0.0
        cfg["congestion_pct_away"] = 0.0
        cfg["rest_disadvantage_pct"] = 0.0
        cfg["rest_advantage_pct"] = 0.0
        cfg["max_down_pct"] = float(args.max_down_pct)

        variant = f"keyabs_only_home_{home_pct:+.2f}".replace("+", "plus").replace("-", "minus")
        cfg_path = configs_dir / f"{variant}.json"
        save_rule_layer_config(cfg_path, cfg)

        payload = _run_backtest(
            model_dir=model_dir,
            config_path=cfg_path,
            output_dir=runs_dir,
            stem=f"{args.output_stem}_{variant}",
        )
        metrics = _extract_metrics(payload)
        row: dict[str, Any] = {
            "variant": variant,
            "config_path": str(cfg_path),
            "config_version": str(cfg.get("config_version", "")),
            "home_key_absent_pct": float(cfg["key_absent_pct_home"]),
            "away_key_absent_pct": float(cfg["key_absent_pct_away"]),
            "max_down_pct": float(cfg["max_down_pct"]),
            "rule_family": "keyabs_only",
            **metrics,
        }
        row["delta_global_home_vs_l2"] = (
            row["global_home_lift_rules"] - row["global_home_lift_l2"]
        )
        row["delta_global_away_vs_l2"] = (
            row["global_away_lift_rules"] - row["global_away_lift_l2"]
        )
        row["delta_triggered_home_vs_l2"] = (
            row["triggered_home_lift_rules"] - row["triggered_home_lift_l2"]
        )
        row["delta_triggered_away_vs_l2"] = (
            row["triggered_away_lift_rules"] - row["triggered_away_lift_l2"]
        )
        row["delta_home_hit_rate_vs_l2"] = (
            row["home_hit_rate_rules"] - row["home_hit_rate_l2"]
        )
        row["gate_pass"] = bool(
            row["triggered_home_events"] >= 30
            and row["delta_triggered_home_vs_l2"] > 0.0
            and row["delta_global_home_vs_l2"] >= 0.0
            and row["delta_global_away_vs_l2"] >= -1e-6
        )
        results.append(row)

    ranked = _rank_rows(results)
    ranked_keyabs = [r for r in ranked if r.get("rule_family") == "keyabs_only"]
    best_keyabs = ranked_keyabs[0] if ranked_keyabs else None

    promoted = False
    if args.promote_best and best_keyabs is not None and bool(best_keyabs.get("gate_pass")):
        best_cfg_path = Path(str(best_keyabs["config_path"]))
        best_cfg = load_rule_layer_config(best_cfg_path)
        best_cfg["config_version"] = str(best_cfg.get("config_version", "keyabs_only_promoted"))
        save_rule_layer_config(active_cfg_path, best_cfg)
        promoted = True

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "active_rule_layer_config": str(active_cfg_path),
        "home_pcts_tested": home_pcts,
        "max_down_pct": float(args.max_down_pct),
        "gates": {
            "min_triggered_home_events": 30,
            "triggered_home_lift_delta_gt": 0.0,
            "global_home_lift_delta_gte": 0.0,
            "global_away_lift_delta_gte": -1e-6,
        },
        "best_keyabs_variant": best_keyabs,
        "promoted_best_variant": promoted,
        "rows": ranked,
    }

    json_path = output_dir / f"{args.output_stem}.json"
    csv_path = output_dir / f"{args.output_stem}.csv"
    md_path = output_dir / f"{args.output_stem}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(csv_path, ranked)

    lines: list[str] = []
    lines.append("# Layer 2 Rule-Layer Key-Absence Sweep")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append("")
    lines.append("## Gate Definition")
    lines.append("- `triggered_home_events >= 30`")
    lines.append("- `delta_triggered_home_vs_l2 > 0`")
    lines.append("- `delta_global_home_vs_l2 >= 0`")
    lines.append("- `delta_global_away_vs_l2 >= 0` (tolerance `-1e-6`)")
    lines.append("")
    lines.append("## Top Variants")
    lines.append(
        "| Variant | Family | Home Key % | Triggered Home Delta vs L2 | Global Home Delta vs L2 | Global Away Delta vs L2 | Home Hit Delta vs L2 | Gate |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in ranked[:8]:
        home_key = row.get("home_key_absent_pct")
        home_key_str = (
            "-"
            if home_key in ("", None)
            else _format_pct(float(home_key))
        )
        lines.append(
            f"| {row['variant']} | {row['rule_family']} | {home_key_str} | "
            f"{_format_pct(float(row['delta_triggered_home_vs_l2']))} | "
            f"{_format_pct(float(row['delta_global_home_vs_l2']))} | "
            f"{_format_pct(float(row['delta_global_away_vs_l2']))} | "
            f"{_format_pct(float(row['delta_home_hit_rate_vs_l2']))} | "
            f"{'pass' if row['gate_pass'] else 'fail'} |"
        )

    lines.append("")
    lines.append("## Decision")
    if best_keyabs is None:
        lines.append("- No key-absence variants were executed.")
    else:
        lines.append(
            f"- Best key-absence variant: `{best_keyabs['variant']}` "
            f"(gate: `{'pass' if best_keyabs['gate_pass'] else 'fail'}`)."
        )
        lines.append(
            f"- Triggered home delta vs Layer2-only: "
            f"{_format_pct(float(best_keyabs['delta_triggered_home_vs_l2']))}."
        )
        lines.append(
            f"- Global home delta vs Layer2-only: "
            f"{_format_pct(float(best_keyabs['delta_global_home_vs_l2']))}."
        )
        lines.append(
            f"- Global away delta vs Layer2-only: "
            f"{_format_pct(float(best_keyabs['delta_global_away_vs_l2']))}."
        )
        if promoted:
            lines.append(
                "- Promotion: best passing key-absence config promoted to active "
                "`model_artifacts/situational_model/rule_layer_config.json`."
            )
        else:
            lines.append("- Promotion: not applied.")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
