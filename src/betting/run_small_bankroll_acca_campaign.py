"""Small-bankroll accumulator campaign runner."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.backtest_accumulators import (
    build_candidates,
    build_slips_walk_forward,
    fetch_rows,
    load_policy_bundle,
)
from src.db.db_utils import get_database_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small-bankroll accumulator campaign using existing backtest data."
    )
    parser.add_argument("--since-days", type=int, default=365)
    parser.add_argument("--initial-bankroll", type=float, default=100.0)
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=Path(
            "model_artifacts/market_models/accumulator_policy_small_bankroll.json"
        ),
    )
    parser.add_argument(
        "--whitelist-path",
        type=Path,
        default=Path(
            "model_artifacts/market_models/accumulator_market_whitelist_small_bankroll.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/reports/backtests/small_bankroll_campaign"),
    )
    parser.add_argument("--slip-size", type=int, default=None)
    parser.add_argument("--simulation-count", type=int, default=2000)
    parser.add_argument(
        "--campaign-slips",
        type=int,
        default=150,
        help="Number of bootstrapped slips per simulated bankroll path.",
    )
    parser.add_argument(
        "--bankroll-floor",
        type=float,
        default=20.0,
        help="Campaign is treated as ruined when bankroll drops to or below this level.",
    )
    parser.add_argument(
        "--milestones",
        type=str,
        default="200,500,1000,5000",
        help="Comma-separated bankroll targets.",
    )
    parser.add_argument(
        "--include-watchlist",
        action="store_true",
        help="Include watchlist markets in addition to tradable markets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _resolve_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = payload.get("backtest_metrics", {})
    sim = payload.get("simulation", {})
    support = payload.get("support_assessment", {})
    lines = [
        "# Small-Bankroll Accumulator Campaign",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc')}",
        f"- policy_path: {payload.get('policy_path')}",
        f"- whitelist_path: {payload.get('whitelist_path')}",
        f"- since_days: {payload.get('since_days')}",
        f"- slip_size: {payload.get('slip_size')}",
        f"- initial_bankroll: {payload.get('initial_bankroll')}",
        "",
        "## Market Universe",
        f"- eligible_markets_from_gating: {payload.get('eligible_market_count')}",
        f"- tradable_markets_resolved: {payload.get('resolved_market_count')}",
        f"- resolved_markets: {', '.join(payload.get('resolved_markets', []))}",
        "",
        "## Walk-Forward Backtest",
        f"- slip_count: {metrics.get('slip_count')}",
        f"- ticket_hit_rate: {metrics.get('ticket_hit_rate')}",
        f"- roi: {metrics.get('roi')}",
        f"- max_drawdown: {metrics.get('max_drawdown')}",
        f"- starting_bankroll: {metrics.get('starting_bankroll')}",
        f"- ending_bankroll: {metrics.get('ending_bankroll')}",
        "",
        "## Bootstrap Campaign Simulation",
        f"- simulation_count: {sim.get('simulation_count')}",
        f"- campaign_slips: {sim.get('campaign_slips')}",
        f"- bankroll_floor: {sim.get('bankroll_floor')}",
        f"- mean_ending_bankroll: {sim.get('mean_ending_bankroll')}",
        f"- median_ending_bankroll: {sim.get('median_ending_bankroll')}",
        f"- ruin_probability: {sim.get('ruin_probability')}",
        "",
        "## Support Assessment",
        f"- observed_slips: {support.get('observed_slips')}",
        f"- min_recommended_slips_for_bootstrap: {support.get('min_recommended_slips_for_bootstrap')}",
        f"- bootstrap_support_ok: {support.get('bootstrap_support_ok')}",
        f"- warning: {support.get('warning')}",
        "",
        "## Milestones",
    ]
    milestone_probs = sim.get("milestone_hit_probability", {})
    if isinstance(milestone_probs, dict) and milestone_probs:
        for key, value in milestone_probs.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_milestones(value: str) -> list[float]:
    out: list[float] = []
    for raw in value.split(","):
        text = raw.strip()
        if not text:
            continue
        out.append(float(text))
    deduped = sorted(set(out))
    if not deduped:
        raise ValueError("At least one milestone is required")
    return deduped


def resolve_campaign_markets(
    *,
    eligible_markets: set[str],
    whitelist_payload: dict[str, Any],
    include_watchlist: bool,
) -> tuple[set[str], dict[str, Any]]:
    tradable = {
        str(item)
        for item in whitelist_payload.get("tradable_markets", [])
        if str(item).strip()
    }
    watchlist = {
        str(item)
        for item in whitelist_payload.get("watchlist_markets", [])
        if str(item).strip()
    }
    blocked = {
        str(item)
        for item in whitelist_payload.get("blocked_markets", [])
        if str(item).strip()
    }

    requested = set(tradable)
    if include_watchlist:
        requested |= watchlist
    requested -= blocked
    resolved = set(eligible_markets) & requested
    summary = {
        "requested_tradable_count": len(tradable),
        "requested_watchlist_count": len(watchlist),
        "requested_blocked_count": len(blocked),
        "eligible_market_count": len(eligible_markets),
        "resolved_market_count": len(resolved),
        "requested_but_ineligible": sorted(requested - set(eligible_markets)),
        "blocked_markets": sorted(blocked),
        "include_watchlist": include_watchlist,
    }
    return resolved, summary


@dataclass(frozen=True)
class SimulationSummary:
    simulation_count: int
    campaign_slips: int
    bankroll_floor: float
    mean_ending_bankroll: float
    median_ending_bankroll: float
    ruin_probability: float
    milestone_hit_probability: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulation_count": self.simulation_count,
            "campaign_slips": self.campaign_slips,
            "bankroll_floor": self.bankroll_floor,
            "mean_ending_bankroll": self.mean_ending_bankroll,
            "median_ending_bankroll": self.median_ending_bankroll,
            "ruin_probability": self.ruin_probability,
            "milestone_hit_probability": dict(self.milestone_hit_probability),
        }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def simulate_target_paths(
    *,
    return_factors: list[float],
    stake_fraction: float,
    initial_bankroll: float,
    bankroll_floor: float,
    milestones: list[float],
    simulation_count: int,
    campaign_slips: int,
    seed: int,
) -> SimulationSummary:
    if not return_factors:
        return SimulationSummary(
            simulation_count=simulation_count,
            campaign_slips=campaign_slips,
            bankroll_floor=bankroll_floor,
            mean_ending_bankroll=initial_bankroll,
            median_ending_bankroll=initial_bankroll,
            ruin_probability=0.0,
            milestone_hit_probability={str(int(m)): 0.0 for m in milestones},
        )

    rng = random.Random(seed)
    endings: list[float] = []
    ruined = 0
    milestone_hits = {str(int(m)): 0 for m in milestones}

    for _ in range(simulation_count):
        bankroll = float(initial_bankroll)
        hit_state = {milestone: False for milestone in milestones}
        for _ticket_idx in range(campaign_slips):
            if bankroll <= bankroll_floor:
                ruined += 1
                break
            stake = bankroll * stake_fraction
            factor = rng.choice(return_factors)
            bankroll += stake * (factor - 1.0)
            for milestone in milestones:
                if bankroll >= milestone:
                    hit_state[milestone] = True
        endings.append(bankroll)
        for milestone, hit in hit_state.items():
            if hit:
                milestone_hits[str(int(milestone))] += 1

    return SimulationSummary(
        simulation_count=simulation_count,
        campaign_slips=campaign_slips,
        bankroll_floor=bankroll_floor,
        mean_ending_bankroll=(sum(endings) / len(endings)) if endings else initial_bankroll,
        median_ending_bankroll=_quantile(endings, 0.5),
        ruin_probability=(ruined / simulation_count) if simulation_count > 0 else 0.0,
        milestone_hit_probability={
            key: (count / simulation_count) if simulation_count > 0 else 0.0
            for key, count in milestone_hits.items()
        },
    )


def main() -> int:
    args = parse_args()

    try:
        _ = get_database_url()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    milestones = _parse_milestones(args.milestones)
    whitelist_path = _resolve_path(args.whitelist_path)
    whitelist_payload = _load_json(whitelist_path)

    (
        policy,
        policy_path,
        _gating,
        model_name,
        model_version,
        eligible_markets,
        limited_to_one_leg_markets,
    ) = load_policy_bundle(args.policy_path)

    resolved_markets, resolution_summary = resolve_campaign_markets(
        eligible_markets=eligible_markets,
        whitelist_payload=whitelist_payload,
        include_watchlist=bool(args.include_watchlist),
    )
    if not resolved_markets:
        print("ERROR: resolved market universe is empty", file=sys.stderr)
        return 1

    slip_size_default = int(policy.get("slip_size_default", 3))
    slip_size = int(args.slip_size) if args.slip_size is not None else slip_size_default

    rows = fetch_rows(
        model_name=model_name,
        model_version=model_version,
        since_days=int(args.since_days),
        eligible_markets=resolved_markets,
    )
    candidates, rejects = build_candidates(
        rows,
        policy=policy,
        eligible_markets=resolved_markets,
    )
    slips, slip_rows, metrics = build_slips_walk_forward(
        candidates=candidates,
        slip_size=slip_size,
        policy=policy,
        limited_to_one_leg_markets=limited_to_one_leg_markets & resolved_markets,
        initial_bankroll=float(args.initial_bankroll),
    )

    simulation = simulate_target_paths(
        return_factors=[float(slip.result_return_factor) for slip in slips],
        stake_fraction=float(policy.get("stake_fraction_per_ticket", 0.01)),
        initial_bankroll=float(args.initial_bankroll),
        bankroll_floor=float(args.bankroll_floor),
        milestones=milestones,
        simulation_count=int(args.simulation_count),
        campaign_slips=int(args.campaign_slips),
        seed=int(args.seed),
    )

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    stem = f"small_bankroll_campaign_{ts}"
    out_dir = _resolve_path(args.out_dir)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    csv_path = out_dir / f"{stem}_slips.csv"
    slips_json_path = out_dir / f"{stem}_slips.json"

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "policy_path": str(policy_path),
        "whitelist_path": str(whitelist_path),
        "since_days": int(args.since_days),
        "slip_size": slip_size,
        "initial_bankroll": float(args.initial_bankroll),
        "eligible_market_count": len(eligible_markets),
        "resolved_market_count": len(resolved_markets),
        "resolved_markets": sorted(resolved_markets),
        "resolution_summary": resolution_summary,
        "raw_rows": len(rows),
        "candidate_legs": len(candidates),
        "reject_reasons": dict(sorted(rejects.items())),
        "backtest_metrics": metrics,
        "simulation": simulation.to_dict(),
        "support_assessment": {
            "observed_slips": int(metrics["slip_count"]),
            "min_recommended_slips_for_bootstrap": 30,
            "bootstrap_support_ok": bool(metrics["slip_count"] >= 30),
            "warning": (
                "Bootstrap milestone probabilities are low-support and should be treated as directional only."
                if int(metrics["slip_count"]) < 30
                else ""
            ),
        },
        "outputs": {
            "json": str(json_path),
            "md": str(md_path),
            "slips_csv": str(csv_path),
            "slips_json": str(slips_json_path),
        },
    }

    _write_json(json_path, payload)
    _write_md(md_path, payload)
    _write_csv(csv_path, slip_rows)
    _write_json(
        slips_json_path,
        {
            "generated_at_utc": payload["generated_at_utc"],
            "slips": [slip.to_dict() for slip in slips],
        },
    )

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote MD: {md_path}")
    print(f"Wrote slips CSV: {csv_path}")
    print(f"Wrote slips JSON: {slips_json_path}")
    print(
        "Campaign: "
        f"slips={metrics['slip_count']} "
        f"roi={metrics['roi']:.4f} "
        f"max_drawdown={metrics['max_drawdown']:.4f} "
        f"ruin_probability={simulation.ruin_probability:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
