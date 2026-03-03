from pathlib import Path
import shutil
import pytest
from src.betting.slip_schema import Slip, Leg


@pytest.fixture
def local_tmp_path():
    path = Path("tests/tmp_test_data")
    path.mkdir(parents=True, exist_ok=True)
    yield path
    if path.exists():
        shutil.rmtree(path)

def test_slip_round_trip_json(local_tmp_path: Path):
    json_path = local_tmp_path / "slip.json"
    leg1 = Leg(
        fixture_id="f1",
        market_code="1x2_h",
        selection="home",
        line_num=0.0,
        odds_used=2.0,
        p_model=0.55,
        p_conservative=0.52,
        edge=0.1,
        risk_flags=["low_liquidity"],
    )

    leg2 = Leg(
        fixture_id="f2",
        market_code="o15",
        selection="over",
        line_num=1.5,
        odds_used=1.9,
        p_model=0.6,
        p_conservative=0.58,
        edge=0.14,
        risk_flags=[],
    )

    slip = Slip(
        slip_id="slip_001",
        date="2026-03-03",
        stake=100.0,
        legs=[leg1, leg2],
        ticket_odds=3.8,
        ticket_win_prob=0.3,
        ticket_ev=1.14,
        result_return_factor=0.0,
        pnl=-100.0,
    )

    slip.save_json(str(json_path))
    loaded_slip = Slip.load_json(str(json_path))

    assert loaded_slip.slip_id == slip.slip_id
    assert loaded_slip.date == slip.date
    assert loaded_slip.stake == slip.stake
    assert len(loaded_slip.legs) == 2
    assert loaded_slip.legs[0].fixture_id == "f1"
    assert loaded_slip.legs[1].selection == "over"
    assert loaded_slip.ticket_odds == 3.8
    assert loaded_slip.pnl == -100.0
    assert "low_liquidity" in loaded_slip.legs[0].risk_flags

def test_leg_to_from_dict():
    data = {
        "fixture_id": "f3",
        "market_code": "u35",
        "selection": "under",
        "line_num": 3.5,
        "odds_used": 1.8,
        "p_model": 0.6,
        "p_conservative": 0.55,
        "edge": 0.08,
        "risk_flags": ["flag1"],
    }
    leg = Leg.from_dict(data)
    assert leg.fixture_id == "f3"
    assert leg.to_dict() == data
