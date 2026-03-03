from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from decimal import ROUND_HALF_UP
from decimal import getcontext


getcontext().prec = 12

GoalsRule = Callable[[int, int], bool]


@dataclass(frozen=True)
class SettlementResult:
    market_code: str
    actual: float | None
    return_factor: float
    outcome: str


def settle_asian_handicap(
    *,
    market_code: str,
    selected_goals: int,
    opponent_goals: int,
    handicap: float,
    odds: float,
) -> SettlementResult:
    if odds <= 0.0:
        raise ValueError("odds must be positive")

    handicap_dec = Decimal(str(handicap))
    selected_dec = Decimal(selected_goals)
    opponent_dec = Decimal(opponent_goals)

    component_lines = _split_quarter_line(handicap_dec)
    component_returns = [
        _settle_single_ah_line(
            selected_goals=selected_dec,
            opponent_goals=opponent_dec,
            handicap=line,
            odds=odds,
        )
        for line in component_lines
    ]
    return_factor = sum(component_returns) / float(len(component_returns))
    outcome = _return_factor_to_ah_outcome(return_factor=return_factor, odds=odds)
    actual = _actual_from_outcome(outcome)
    return SettlementResult(
        market_code=market_code,
        actual=actual,
        return_factor=return_factor,
        outcome=outcome,
    )


def settle_market(
    market_code: str,
    *,
    odds: float,
    home_goals: int | None = None,
    away_goals: int | None = None,
    home_corners: int | None = None,
    away_corners: int | None = None,
    incident_lead_states: Mapping[str, bool] | None = None,
) -> SettlementResult:
    if odds <= 0.0:
        raise ValueError("odds must be positive")

    if market_code.startswith("ah_"):
        if home_goals is None or away_goals is None:
            return _void(market_code)
        side, handicap = _parse_ah_market_code(market_code)
        selected, opponent = (
            (home_goals, away_goals) if side == "home" else (away_goals, home_goals)
        )
        return settle_asian_handicap(
            market_code=market_code,
            selected_goals=selected,
            opponent_goals=opponent,
            handicap=handicap,
            odds=odds,
        )

    if market_code.startswith("eh_"):
        if home_goals is None or away_goals is None:
            return _void(market_code)
        side, line = _parse_eh_market_code(market_code)
        selected, opponent = (
            (home_goals, away_goals) if side == "home" else (away_goals, home_goals)
        )
        margin = selected - opponent
        if margin > line:
            return _binary(market_code, True, odds)
        if margin == line:
            return SettlementResult(market_code, None, 1.0, "push")
        return _binary(market_code, False, odds)

    goals_markets: dict[str, GoalsRule] = {
        "dc_1x": lambda h, a: h >= a,
        "dc_x2": lambda h, a: a >= h,
        "dc_12": lambda h, a: h != a,
        "ho15": lambda h, a: h >= 2,
        "ao15": lambda h, a: a >= 2,
        "o15": lambda h, a: (h + a) >= 2,
        "u35": lambda h, a: (h + a) <= 3,
    }
    if market_code in goals_markets:
        if home_goals is None or away_goals is None:
            return _void(market_code)
        return _binary(
            market_code, goals_markets[market_code](home_goals, away_goals), odds
        )

    corners_thresholds = {
        "c75": 8,
        "c85": 9,
        "c95": 10,
        "c105": 11,
    }
    if market_code in corners_thresholds:
        if home_corners is None or away_corners is None:
            return _void(market_code)
        total = home_corners + away_corners
        return _binary(market_code, total >= corners_thresholds[market_code], odds)

    home_corner_markets = {
        "hc25": 3,
        "hc35": 4,
        "hc45": 5,
        "hc55": 6,
    }
    if market_code in home_corner_markets:
        if home_corners is None:
            return _void(market_code)
        return _binary(
            market_code, home_corners >= home_corner_markets[market_code], odds
        )

    away_corner_markets = {
        "ac25": 3,
        "ac35": 4,
        "ac45": 5,
        "ac55": 6,
    }
    if market_code in away_corner_markets:
        if away_corners is None:
            return _void(market_code)
        return _binary(
            market_code, away_corners >= away_corner_markets[market_code], odds
        )

    incident_markets = {
        "h_1up": "home_led_by_1_any",
        "a_1up": "away_led_by_1_any",
        "h_2up": "home_led_by_2_any",
        "a_2up": "away_led_by_2_any",
    }
    if market_code in incident_markets:
        if incident_lead_states is None:
            return _void(market_code)
        key = incident_markets[market_code]
        if key not in incident_lead_states:
            return _void(market_code)
        return _binary(market_code, bool(incident_lead_states[key]), odds)

    raise ValueError(f"unsupported market_code: {market_code}")


def supported_market_codes() -> tuple[str, ...]:
    return (
        "a_1up",
        "a_2up",
        "ac25",
        "ac35",
        "ac45",
        "ac55",
        "ah_a05",
        "ah_a15",
        "ah_h05",
        "ah_h15",
        "ao15",
        "c105",
        "c75",
        "c85",
        "c95",
        "dc_12",
        "dc_1x",
        "dc_x2",
        "eh_a1",
        "eh_h1",
        "h_1up",
        "h_2up",
        "hc25",
        "hc35",
        "hc45",
        "hc55",
        "ho15",
        "o15",
        "u35",
    )


def _void(market_code: str) -> SettlementResult:
    return SettlementResult(
        market_code=market_code, actual=None, return_factor=1.0, outcome="void"
    )


def _binary(market_code: str, won: bool, odds: float) -> SettlementResult:
    if won:
        return SettlementResult(
            market_code=market_code, actual=1.0, return_factor=odds, outcome="full_win"
        )
    return SettlementResult(
        market_code=market_code, actual=0.0, return_factor=0.0, outcome="full_loss"
    )


def _actual_from_outcome(outcome: str) -> float | None:
    if outcome == "full_win":
        return 1.0
    if outcome == "full_loss":
        return 0.0
    return None


def _parse_ah_market_code(market_code: str) -> tuple[str, float]:
    side = market_code[3:4]
    raw = market_code[4:]
    if side not in {"h", "a"} or not raw or not raw.isdigit():
        raise ValueError(f"invalid AH market_code: {market_code}")
    magnitude = float(int(raw) / 10.0)
    handicap = -magnitude
    return ("home" if side == "h" else "away", handicap)


def _parse_eh_market_code(market_code: str) -> tuple[str, int]:
    side = market_code[3:4]
    raw = market_code[4:]
    if side not in {"h", "a"} or not raw or not raw.isdigit():
        raise ValueError(f"invalid EH market_code: {market_code}")
    return ("home" if side == "h" else "away", int(raw))


def _split_quarter_line(handicap: Decimal) -> list[Decimal]:
    quarter = Decimal("0.25")
    fractional = abs(handicap % Decimal("1"))
    if fractional in {Decimal("0.25"), Decimal("0.75")}:
        return [handicap - quarter, handicap + quarter]
    return [handicap]


def _settle_single_ah_line(
    *,
    selected_goals: Decimal,
    opponent_goals: Decimal,
    handicap: Decimal,
    odds: float,
) -> float:
    net = selected_goals + handicap - opponent_goals
    if net > 0:
        return odds
    if net == 0:
        return 1.0
    return 0.0


def _return_factor_to_ah_outcome(return_factor: float, odds: float) -> str:
    quantized = Decimal(str(return_factor)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )
    full_win = Decimal(str(odds)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    push = Decimal("1.000000")
    full_loss = Decimal("0.000000")
    half_win = Decimal(str((odds + 1.0) / 2.0)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )
    half_loss = Decimal("0.500000")

    if quantized == full_win:
        return "full_win"
    if quantized == push:
        return "push"
    if quantized == full_loss:
        return "full_loss"
    if quantized == half_win:
        return "half_win"
    if quantized == half_loss:
        return "half_loss"
    raise RuntimeError(f"unexpected AH return_factor={return_factor} for odds={odds}")
