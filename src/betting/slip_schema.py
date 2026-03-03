from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Leg:
    fixture_id: str
    market_code: str
    selection: str
    line_num: float
    odds_used: float
    p_model: float
    p_conservative: float
    edge: float
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Leg:
        return cls(**data)


@dataclass
class Slip:
    slip_id: str
    date: str
    stake: float
    legs: list[Leg]
    ticket_odds: float
    ticket_win_prob: float
    ticket_ev: float
    result_return_factor: float
    pnl: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["legs"] = [leg.to_dict() for leg in self.legs]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Slip:
        legs_data = data.pop("legs", [])
        legs = [Leg.from_dict(l) for l in legs_data]
        return cls(legs=legs, **data)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> Slip:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
