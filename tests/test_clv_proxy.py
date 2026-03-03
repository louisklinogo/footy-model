from __future__ import annotations

import pytest

from src.betting import report_clv_proxy
from src.betting.report_clv_proxy import _extract_price


def test_clv_proxy_math_from_opening_vs_latest_prices() -> None:
    odds_json = {
        "prices_opening": {"home": 2.50},
        "prices_latest": {"home": 2.00},
    }

    opening_price = _extract_price(odds_json, "home", "prices_opening")
    latest_price = _extract_price(odds_json, "home", "prices_latest")

    assert opening_price == 2.50
    assert latest_price == 2.00
    assert opening_price is not None
    assert latest_price is not None

    implied_open = 1.0 / opening_price
    implied_latest = 1.0 / latest_price
    clv = implied_latest - implied_open
    assert clv == pytest.approx(0.1)


def test_clv_proxy_missing_price_returns_none() -> None:
    odds_json = {
        "prices_opening": {"away": 2.40},
        "prices_latest": {},
    }

    opening_price = _extract_price(odds_json, "home", "prices_opening")
    latest_price = _extract_price(odds_json, "home", "prices_latest")

    assert opening_price is None
    assert latest_price is None


def test_clv_proxy_main_handles_missing_db_runtime_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        report_clv_proxy,
        "parse_args",
        lambda: report_clv_proxy.argparse.Namespace(
            since_days=90, out="artifacts/reports/backtests/clv_proxy_test.md"
        ),
    )

    def _raise_runtime_error(_: int) -> list[dict[str, object]]:
        raise RuntimeError("Missing DATABASE_URL")

    monkeypatch.setattr(report_clv_proxy, "fetch_clv_data", _raise_runtime_error)

    exit_code = report_clv_proxy.main()
    output = capsys.readouterr().out.strip()

    assert exit_code == 0
    assert output == "CLV unavailable: Missing DATABASE_URL"
