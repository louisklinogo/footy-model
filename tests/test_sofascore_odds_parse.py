# pyright: reportUnknownParameterType=false, reportMissingParameterType=false

from __future__ import annotations

from src.ingest.backfill_sofascore_odds_markets_v1 import (
    extract_markets,
    extract_line,
    normalize_prices,
    parse_fractional,
)


def test_parse_fractional_to_decimal() -> None:
    assert parse_fractional("1/1") == 2.0
    assert parse_fractional("1/2") == 1.5
    assert parse_fractional("2/1") == 3.0


def test_extract_1x2_market_with_provider() -> None:
    payload = {
        "markets": [
            {
                "marketName": "Full time",
                "choices": [
                    {"name": "1", "initialFractionalValue": "1/1", "fractionalValue": "1/1", "providerId": 1},
                    {"name": "X", "initialFractionalValue": "2/1", "fractionalValue": "2/1", "providerId": 1},
                    {"name": "2", "initialFractionalValue": "3/1", "fractionalValue": "3/1", "providerId": 1},
                ],
            }
        ]
    }
    out = extract_markets(payload, provider_id=1)
    assert out
    assert out[0]["market_code"] == "1x2"


def test_normalize_prices_implied() -> None:
    choices = {
        "home": {"initialFractionalValue": "1/1", "fractionalValue": "1/1"},
        "draw": {"initialFractionalValue": "2/1", "fractionalValue": "2/1"},
        "away": {"initialFractionalValue": "3/1", "fractionalValue": "3/1"},
    }
    norm = normalize_prices(choices)
    assert norm is not None
    implied = norm["implied_prob"]
    assert abs(sum(implied.values()) - 1.0) < 1e-9


def test_extract_line() -> None:
    assert extract_line("Over 2.5") == 2.5
    assert extract_line("Under 10.0") == 10.0
    assert extract_line("AH -0.25") == -0.25


def test_extract_ah_market_with_team_name_choices() -> None:
    payload = {
        "markets": [
            {
                "marketName": "Asian handicap",
                "choiceGroup": "-0.5",
                "homeTeamName": "Leeds United",
                "awayTeamName": "Arsenal",
                "choices": [
                    {
                        "name": "Leeds United (-0.5)",
                        "initialFractionalValue": "10/11",
                        "fractionalValue": "10/11",
                        "providerId": 1,
                    },
                    {
                        "name": "Arsenal (+0.5)",
                        "initialFractionalValue": "6/5",
                        "fractionalValue": "6/5",
                        "providerId": 1,
                    },
                ],
            }
        ]
    }
    out = extract_markets(payload, provider_id=1)
    assert len(out) == 1
    row = out[0]
    assert row["market_code"] == "ah"
    assert row["line_num"] == -0.5
    latest = row["prices_latest"]
    assert set(latest.keys()) == {"home", "away"}
    assert latest["home"] > 1.0
    assert latest["away"] > 1.0


def test_extract_ah_market_fallbacks_to_choice_order_when_side_is_unknown() -> None:
    payload = {
        "markets": [
            {
                "marketName": "Asian handicap",
                "choiceGroup": "-0.5",
                "choices": [
                    {
                        "name": "(-0.5)",
                        "initialFractionalValue": "2/1",
                        "fractionalValue": "2/1",
                        "providerId": 1,
                    },
                    {
                        "name": "(+0.5)",
                        "initialFractionalValue": "1/1",
                        "fractionalValue": "1/1",
                        "providerId": 1,
                    },
                ],
            }
        ]
    }
    out = extract_markets(payload, provider_id=1)
    assert len(out) == 1
    row = out[0]
    assert row["market_code"] == "ah"
    latest = row["prices_latest"]
    assert latest["home"] == 3.0
    assert latest["away"] == 2.0
