"""Walk-forward accumulator backtester."""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false, reportExplicitAny=false

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.settlement import settle_market
from src.betting.slip_schema import Leg, Slip
from src.db.db_utils import connect_db, get_database_url


@dataclass(frozen=True)
class CandidateLeg:
    prediction_id: int
    fixture_id: int
    market_code: str
    league_code: str
    kickoff_utc: datetime
    home_team_id: int
    away_team_id: int
    odds_used: float
    p_model: float
    p_conservative: float
    edge_adjusted: float
    risk_score: float
    line_num: float
    risk_flags: list[str]
    home_goals: int | None
    away_goals: int | None
    home_corners: int | None
    away_corners: int | None
    home_led_by_1_any: bool | None
    away_led_by_1_any: bool | None
    home_led_by_2_any: bool | None
    away_led_by_2_any: bool | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward accumulator backtest with drawdown simulation."
    )
    parser.add_argument("--since-days", type=int, default=365)
    parser.add_argument("--slip-size", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/reports/backtests"),
    )
    parser.add_argument("--initial-bankroll", type=float, default=100.0)
    return parser.parse_args()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_list_of_str(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
    return None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _resolve_model_from_gating(gating: dict[str, Any]) -> tuple[str, str]:
    model_name = str(gating.get("model_name") or "")
    model_version = str(gating.get("model_version") or "")
    if not model_name or not model_version:
        raise ValueError("market_gating.json missing model_name/model_version")
    return model_name, model_version


def _eligible_markets(gating: dict[str, Any]) -> tuple[set[str], set[str]]:
    items = gating.get("markets")
    if not isinstance(items, list):
        raise ValueError("market_gating.json markets must be a list")
    eligible: set[str] = set()
    limited_one_leg: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        market_code = str(item.get("market_code") or "")
        if not market_code:
            continue
        if bool(item.get("eligible", False)):
            eligible.add(market_code)
            if bool(item.get("limited_to_one_leg", False)):
                limited_one_leg.add(market_code)
    return eligible, limited_one_leg


def fetch_rows(
    *,
    model_name: str,
    model_version: str,
    since_days: int,
    eligible_markets: set[str],
) -> list[dict[str, Any]]:
    conn = connect_db()
    has_ps_metadata_json = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'prediction_scores'
                  AND column_name = 'metadata_json'
                LIMIT 1
                """
            )
            has_ps_metadata_json = cur.fetchone() is not None
    finally:
        conn.close()

    score_metadata_expr = "ps.metadata_json" if has_ps_metadata_json else "'{}'::jsonb"

    query = f"""
    SELECT
        p.prediction_id,
        p.fixture_id,
        p.market_code,
        COALESCE(p.p_final, p.p_model) AS p_model,
        ps.odds_used,
        ps.edge,
        {score_metadata_expr} AS score_metadata_json,
        pra.p_conservative,
        pra.edge_adjusted,
        pra.risk_score,
        pra.risk_flags_json,
        pra.metadata_json AS risk_metadata_json,
        f.league_code,
        f.match_datetime_utc,
        f.home_team_id,
        f.away_team_id,
        fr.home_goals,
        fr.away_goals,
        fs.h_corners,
        fs.a_corners,
        ils.home_led_by_1_any,
        ils.away_led_by_1_any,
        ils.home_led_by_2_any,
        ils.away_led_by_2_any
    FROM prediction_scores ps
    JOIN predictions p
      ON p.prediction_id = ps.prediction_id
    JOIN fixtures f
      ON f.fixture_id = p.fixture_id
    LEFT JOIN prediction_risk_assessments pra
      ON pra.prediction_id = p.prediction_id
    LEFT JOIN fixture_results fr
      ON fr.fixture_id = f.fixture_id
    LEFT JOIN fixture_stats_premium fs
      ON fs.fixture_id = f.fixture_id
    LEFT JOIN fixture_incident_lead_states ils
      ON ils.fixture_id = f.fixture_id
    WHERE p.model_name = %s
      AND p.model_version = %s
      AND p.market_code = ANY(%s)
      AND f.status = 'ft'
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
    ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC, p.market_code ASC
    """

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    model_name,
                    model_version,
                    list(sorted(eligible_markets)),
                    since_days,
                ),
            )
            rows = cur.fetchall()
            cols = [desc[0] for desc in (cur.description or [])]
    finally:
        conn.close()
    return [dict(zip(cols, row)) for row in rows]


def build_candidates(
    rows: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    eligible_markets: set[str],
) -> tuple[list[CandidateLeg], Counter[str]]:
    rejects: Counter[str] = Counter()
    candidates: list[CandidateLeg] = []

    allow_missing_odds = bool(policy.get("allow_missing_odds", False))
    min_p_cons = float(policy.get("min_leg_p_conservative", 0.0))
    min_edge_adjusted = float(policy.get("min_leg_edge_adjusted", -1.0))
    max_risk_score = float(policy.get("max_leg_risk_score", 100.0))
    odds_min = float(policy.get("per_leg_odds_min", 1.0))
    odds_max = float(policy.get("per_leg_odds_max", 1e9))
    integrity = _as_dict(policy.get("odds_time_integrity"))
    require_snapshot_le_kickoff = bool(
        integrity.get("require_snapshot_le_kickoff", True)
    )
    allow_closing_if_pre_kickoff = bool(
        integrity.get("allow_closing_if_pre_kickoff", True)
    )
    closing_adds_risk_flag = bool(integrity.get("closing_adds_risk_flag", True))

    for row in rows:
        market_code = str(row.get("market_code") or "")
        if market_code not in eligible_markets:
            rejects["market_not_eligible"] += 1
            continue

        kickoff_utc = _to_dt(row.get("match_datetime_utc"))
        if kickoff_utc is None:
            rejects["missing_kickoff"] += 1
            continue

        odds_used = _to_float(row.get("odds_used"))
        if odds_used is None or odds_used <= 1.0:
            if not allow_missing_odds:
                rejects["missing_odds"] += 1
                continue
            odds_used = 1.0

        p_model = _to_float(row.get("p_model"))
        if p_model is None:
            rejects["missing_p_model"] += 1
            continue

        p_conservative = _to_float(row.get("p_conservative"))
        if p_conservative is None:
            p_conservative = p_model
            rejects["missing_p_conservative_fallback"] += 1

        edge_adjusted = _to_float(row.get("edge_adjusted"))
        if edge_adjusted is None:
            edge_adjusted = _to_float(row.get("edge"))
        if edge_adjusted is None:
            rejects["missing_edge"] += 1
            continue

        risk_score = _to_float(row.get("risk_score"))
        if risk_score is None:
            risk_score = 100.0
            rejects["missing_risk_score_fallback"] += 1

        if p_conservative < min_p_cons:
            rejects["below_min_leg_p_conservative"] += 1
            continue
        if edge_adjusted < min_edge_adjusted:
            rejects["below_min_leg_edge_adjusted"] += 1
            continue
        if risk_score > max_risk_score:
            rejects["above_max_leg_risk_score"] += 1
            continue
        if odds_used < odds_min or odds_used > odds_max:
            rejects["outside_per_leg_odds_band"] += 1
            continue

        score_meta = _as_dict(row.get("score_metadata_json"))
        risk_meta = _as_dict(row.get("risk_metadata_json"))
        odds_trace = _as_dict(score_meta.get("odds_trace"))
        if not odds_trace:
            odds_trace = {
                "snapshot_type": risk_meta.get("odds_snapshot_type"),
                "snapshot_time_utc": risk_meta.get("odds_snapshot_time_utc"),
                "line_num": risk_meta.get("odds_line_num"),
                "odds_field": risk_meta.get("odds_field"),
            }

        snapshot_type = str(odds_trace.get("snapshot_type") or "")
        snapshot_time_utc = _to_dt(odds_trace.get("snapshot_time_utc"))

        if require_snapshot_le_kickoff:
            if snapshot_time_utc is None:
                rejects["missing_odds_trace_time"] += 1
                continue
            if snapshot_time_utc > kickoff_utc:
                rejects["post_kickoff_odds_rejected"] += 1
                continue

        if snapshot_type == "closing" and not allow_closing_if_pre_kickoff:
            rejects["closing_snapshot_disallowed"] += 1
            continue

        risk_flags = _as_list_of_str(row.get("risk_flags_json"))
        if snapshot_type == "closing" and closing_adds_risk_flag:
            risk_flags.append("closing_snapshot")

        line_num = _to_float(odds_trace.get("line_num"))
        if line_num is None:
            line_num = 0.0

        home_goals = _to_int(row.get("home_goals"))
        away_goals = _to_int(row.get("away_goals"))
        home_corners = _to_int(row.get("h_corners"))
        away_corners = _to_int(row.get("a_corners"))

        candidate = CandidateLeg(
            prediction_id=int(row["prediction_id"]),
            fixture_id=int(row["fixture_id"]),
            market_code=market_code,
            league_code=str(row.get("league_code") or ""),
            kickoff_utc=kickoff_utc,
            home_team_id=int(row["home_team_id"]),
            away_team_id=int(row["away_team_id"]),
            odds_used=odds_used,
            p_model=p_model,
            p_conservative=p_conservative,
            edge_adjusted=edge_adjusted,
            risk_score=risk_score,
            line_num=line_num,
            risk_flags=risk_flags,
            home_goals=home_goals,
            away_goals=away_goals,
            home_corners=home_corners,
            away_corners=away_corners,
            home_led_by_1_any=_boolish(row.get("home_led_by_1_any")),
            away_led_by_1_any=_boolish(row.get("away_led_by_1_any")),
            home_led_by_2_any=_boolish(row.get("home_led_by_2_any")),
            away_led_by_2_any=_boolish(row.get("away_led_by_2_any")),
        )
        candidates.append(candidate)

    candidates.sort(
        key=lambda c: (
            c.kickoff_utc,
            -c.edge_adjusted,
            -c.p_conservative,
            c.risk_score,
            c.fixture_id,
            c.market_code,
        )
    )
    return candidates, rejects


def _same_league_haircut_extra_legs(legs: list[CandidateLeg]) -> int:
    counts = Counter(leg.league_code for leg in legs)
    return sum(max(0, count - 1) for count in counts.values())


def _settle_ticket_return_factor(legs: list[CandidateLeg]) -> float:
    return_factor = 1.0
    for leg in legs:
        settlement = settle_market(
            leg.market_code,
            odds=leg.odds_used,
            home_goals=leg.home_goals,
            away_goals=leg.away_goals,
            home_corners=leg.home_corners,
            away_corners=leg.away_corners,
            incident_lead_states={
                "home_led_by_1_any": bool(leg.home_led_by_1_any),
                "away_led_by_1_any": bool(leg.away_led_by_1_any),
                "home_led_by_2_any": bool(leg.home_led_by_2_any),
                "away_led_by_2_any": bool(leg.away_led_by_2_any),
            },
        )
        return_factor *= settlement.return_factor
    return return_factor


def _make_slip(
    *,
    slip_id: str,
    ticket_date: date,
    stake: float,
    legs: list[CandidateLeg],
    same_league_haircut: float,
) -> Slip:
    ticket_odds = 1.0
    p_ticket = 1.0
    leg_models: list[Leg] = []
    for leg in legs:
        ticket_odds *= leg.odds_used
        p_ticket *= leg.p_conservative
        leg_models.append(
            Leg(
                fixture_id=str(leg.fixture_id),
                market_code=leg.market_code,
                selection=leg.market_code,
                line_num=float(leg.line_num),
                odds_used=float(leg.odds_used),
                p_model=float(leg.p_model),
                p_conservative=float(leg.p_conservative),
                edge=float(leg.edge_adjusted),
                risk_flags=list(leg.risk_flags),
            )
        )

    extra_same_league_legs = _same_league_haircut_extra_legs(legs)
    p_ticket *= same_league_haircut**extra_same_league_legs
    result_return_factor = _settle_ticket_return_factor(legs)
    pnl = stake * (result_return_factor - 1.0)
    ticket_ev = stake * (p_ticket * ticket_odds - 1.0)
    return Slip(
        slip_id=slip_id,
        date=ticket_date.isoformat(),
        stake=stake,
        legs=leg_models,
        ticket_odds=ticket_odds,
        ticket_win_prob=p_ticket,
        ticket_ev=ticket_ev,
        result_return_factor=result_return_factor,
        pnl=pnl,
    )


def build_slips_walk_forward(
    *,
    candidates: list[CandidateLeg],
    slip_size: int,
    policy: dict[str, Any],
    limited_to_one_leg_markets: set[str],
    initial_bankroll: float,
) -> tuple[list[Slip], list[dict[str, Any]], dict[str, float | int]]:
    correlation = _as_dict(policy.get("correlation_policy"))
    forbid_same_fixture = bool(correlation.get("forbid_same_fixture", True))
    forbid_same_team = bool(correlation.get("forbid_same_team_across_legs", True))
    max_legs_per_league = int(correlation.get("max_legs_per_league", slip_size))
    same_league_haircut = float(correlation.get("same_league_probability_haircut", 1.0))

    max_tickets_per_day = int(policy.get("max_tickets_per_day", 1))
    stake_fraction = float(policy.get("stake_fraction_per_ticket", 0.01))
    daily_stop_loss_fraction = float(policy.get("daily_stop_loss_fraction", 0.03))
    ticket_total_odds_min = float(policy.get("ticket_total_odds_min", 1.0))
    ticket_total_odds_max = float(policy.get("ticket_total_odds_max", 1e9))

    grouped: dict[date, list[CandidateLeg]] = {}
    for leg in candidates:
        grouped.setdefault(leg.kickoff_utc.date(), []).append(leg)

    bankroll = float(initial_bankroll)
    peak = bankroll
    max_drawdown = 0.0
    slips: list[Slip] = []
    slip_rows: list[dict[str, Any]] = []

    for day in sorted(grouped.keys()):
        day_legs = sorted(
            grouped[day],
            key=lambda c: (
                -c.edge_adjusted,
                -c.p_conservative,
                c.risk_score,
                c.kickoff_utc,
                c.fixture_id,
            ),
        )
        day_start_bankroll = bankroll
        day_loss = 0.0
        tickets_built = 0

        used_prediction_ids: set[int] = set()
        while tickets_built < max_tickets_per_day:
            if bankroll <= 0.0:
                break
            if day_loss >= day_start_bankroll * daily_stop_loss_fraction:
                break

            selected: list[CandidateLeg] = []
            used_fixtures: set[int] = set()
            used_teams: set[int] = set()
            league_counts: Counter[str] = Counter()
            limited_counts: Counter[str] = Counter()

            for leg in day_legs:
                if leg.prediction_id in used_prediction_ids:
                    continue
                if forbid_same_fixture and leg.fixture_id in used_fixtures:
                    continue
                if forbid_same_team and (
                    leg.home_team_id in used_teams or leg.away_team_id in used_teams
                ):
                    continue
                if league_counts[leg.league_code] >= max_legs_per_league:
                    continue
                if (
                    leg.market_code in limited_to_one_leg_markets
                    and limited_counts[leg.market_code] >= 1
                ):
                    continue

                selected.append(leg)
                used_fixtures.add(leg.fixture_id)
                used_teams.add(leg.home_team_id)
                used_teams.add(leg.away_team_id)
                league_counts[leg.league_code] += 1
                limited_counts[leg.market_code] += 1

                if len(selected) == slip_size:
                    break

            if len(selected) < slip_size:
                break

            ticket_odds = 1.0
            for leg in selected:
                ticket_odds *= leg.odds_used
            if (
                ticket_odds < ticket_total_odds_min
                or ticket_odds > ticket_total_odds_max
            ):
                for leg in selected:
                    used_prediction_ids.add(leg.prediction_id)
                continue

            stake = bankroll * stake_fraction
            if stake <= 0.0:
                break

            slip_id = f"{day.isoformat()}-{tickets_built + 1:03d}"
            slip = _make_slip(
                slip_id=slip_id,
                ticket_date=day,
                stake=stake,
                legs=selected,
                same_league_haircut=same_league_haircut,
            )

            bankroll_before = bankroll
            bankroll += slip.pnl
            day_loss += max(0.0, -slip.pnl)
            peak = max(peak, bankroll)
            if peak > 0.0:
                drawdown = (peak - bankroll) / peak
                max_drawdown = max(max_drawdown, drawdown)

            slips.append(slip)
            slip_rows.append(
                {
                    "slip_id": slip.slip_id,
                    "date": slip.date,
                    "stake": slip.stake,
                    "ticket_odds": slip.ticket_odds,
                    "ticket_win_prob": slip.ticket_win_prob,
                    "ticket_ev": slip.ticket_ev,
                    "result_return_factor": slip.result_return_factor,
                    "pnl": slip.pnl,
                    "hit": int(slip.result_return_factor > 1.0),
                    "bankroll_before": bankroll_before,
                    "bankroll_after": bankroll,
                }
            )

            for leg in selected:
                used_prediction_ids.add(leg.prediction_id)
            tickets_built += 1

    total_stake = sum(slip.stake for slip in slips)
    total_pnl = sum(slip.pnl for slip in slips)
    slip_count = len(slips)
    hit_count = sum(1 for slip in slips if slip.result_return_factor > 1.0)
    ticket_hit_rate = (hit_count / slip_count) if slip_count > 0 else 0.0
    roi = (total_pnl / total_stake) if total_stake > 0 else 0.0

    metrics: dict[str, float | int] = {
        "slip_count": slip_count,
        "ticket_hit_rate": ticket_hit_rate,
        "roi": roi,
        "max_drawdown": max_drawdown,
        "total_stake": total_stake,
        "total_pnl": total_pnl,
        "ending_bankroll": bankroll,
        "starting_bankroll": float(initial_bankroll),
    }
    return slips, slip_rows, metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = _as_dict(payload.get("metrics"))
    lines = [
        "# Accumulator Backtest",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc')}",
        f"- model_name: {payload.get('model_name')}",
        f"- model_version: {payload.get('model_version')}",
        f"- since_days: {payload.get('since_days')}",
        f"- slip_size: {payload.get('slip_size')}",
        "",
        "## Metrics",
        f"- slip_count: {metrics.get('slip_count')}",
        f"- ticket_hit_rate: {metrics.get('ticket_hit_rate')}",
        f"- roi: {metrics.get('roi')}",
        f"- max_drawdown: {metrics.get('max_drawdown')}",
        f"- total_stake: {metrics.get('total_stake')}",
        f"- total_pnl: {metrics.get('total_pnl')}",
        f"- starting_bankroll: {metrics.get('starting_bankroll')}",
        f"- ending_bankroll: {metrics.get('ending_bankroll')}",
        "",
        "## Candidate Counts",
        f"- raw_rows: {payload.get('raw_rows')}",
        f"- candidate_legs: {payload.get('candidate_legs')}",
        "",
        "## Reject Reasons",
    ]
    reject_reasons = payload.get("reject_reasons", {})
    if isinstance(reject_reasons, dict) and reject_reasons:
        for key, value in sorted(reject_reasons.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        _ = get_database_url()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    policy_path = ROOT_DIR / "model_artifacts/market_models/accumulator_policy.json"
    policy = _load_json(policy_path)

    gating_path = Path(str(policy.get("market_gating_path") or "")).expanduser()
    if not gating_path.is_absolute():
        gating_path = ROOT_DIR / gating_path
    gating = _load_json(gating_path)

    model_name, model_version = _resolve_model_from_gating(gating)
    eligible_markets, limited_to_one_leg_markets = _eligible_markets(gating)
    if not eligible_markets:
        print("ERROR: no eligible markets in market_gating.json", file=sys.stderr)
        return 1

    slip_size_default = int(policy.get("slip_size_default", 3))
    slip_size = int(args.slip_size) if args.slip_size is not None else slip_size_default
    if slip_size <= 0:
        raise ValueError("slip_size must be > 0")
    if slip_size == 4 and not bool(policy.get("slip_size_allow_4", False)):
        raise ValueError("slip_size=4 not allowed by accumulator policy")

    rows = fetch_rows(
        model_name=model_name,
        model_version=model_version,
        since_days=int(args.since_days),
        eligible_markets=eligible_markets,
    )
    candidates, rejects = build_candidates(
        rows,
        policy=policy,
        eligible_markets=eligible_markets,
    )
    slips, slip_rows, metrics = build_slips_walk_forward(
        candidates=candidates,
        slip_size=slip_size,
        policy=policy,
        limited_to_one_leg_markets=limited_to_one_leg_markets,
        initial_bankroll=float(args.initial_bankroll),
    )

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    stem = f"accumulator_backtest_{ts}"
    out_dir = args.out_dir
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    slips_json_path = out_dir / f"{stem}_slips.json"

    summary_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "since_days": int(args.since_days),
        "slip_size": slip_size,
        "policy_path": str(policy_path),
        "gating_path": str(gating_path),
        "raw_rows": len(rows),
        "candidate_legs": len(candidates),
        "reject_reasons": dict(sorted(rejects.items())),
        "metrics": metrics,
        "slip_count": metrics["slip_count"],
        "ticket_hit_rate": metrics["ticket_hit_rate"],
        "roi": metrics["roi"],
        "max_drawdown": metrics["max_drawdown"],
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "md": str(md_path),
            "slips_json": str(slips_json_path),
        },
    }

    _write_csv(csv_path, slip_rows)
    _write_json(json_path, summary_payload)
    _write_md(md_path, summary_payload)
    _write_json(
        slips_json_path,
        {
            "generated_at_utc": summary_payload["generated_at_utc"],
            "slips": [slip.to_dict() for slip in slips],
        },
    )

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote MD: {md_path}")
    print(f"Wrote slips JSON: {slips_json_path}")
    print(
        f"Metrics: slip_count={metrics['slip_count']} "
        f"ticket_hit_rate={metrics['ticket_hit_rate']:.4f} "
        f"roi={metrics['roi']:.4f} "
        f"max_drawdown={metrics['max_drawdown']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
