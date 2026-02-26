# pyright: reportUnknownParameterType=false, reportMissingParameterType=false

from __future__ import annotations

import datetime

from src.ingest.ingest_sofascore_availability import parse_expected_return


def test_parse_expected_return_none() -> None:
    assert parse_expected_return(None) is None
    assert parse_expected_return("") is None
    assert parse_expected_return("not-a-date") is None


def test_parse_expected_return_unix_seconds() -> None:
    dt = parse_expected_return(1700000000)
    assert isinstance(dt, datetime.datetime)
    assert dt.tzinfo is not None
    assert int(dt.timestamp()) == 1700000000


def test_parse_expected_return_unix_millis() -> None:
    dt = parse_expected_return(1700000000000)
    assert isinstance(dt, datetime.datetime)
    assert dt.tzinfo is not None
    assert int(dt.timestamp()) == 1700000000


def test_parse_expected_return_iso_string() -> None:
    dt = parse_expected_return("2025-01-02T03:04:05Z")
    assert isinstance(dt, datetime.datetime)
    assert dt.tzinfo is not None
    assert dt.year == 2025
    assert dt.month == 1
    assert dt.day == 2
