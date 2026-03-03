import pandas as pd
from typing import List, Dict, Any
from src.pricing.poisson import PoissonPricer

class ValueFinder:
    """
    Compares model-derived probabilities against bookmaker odds 
    to identify positive Expected Value (EV) opportunities.
    """
    
    def __init__(self, min_ev: float = 0.05):
        self.pricer = PoissonPricer()
        self.min_ev = min_ev

    def find_edges(self, lambda_h: float, lambda_a: float, market_odds: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Calculates EV for a set of markets given the predicted lambdas.
        
        Args:
            market_odds: A dict like {"1x2_h": 1.95, "o15": 1.55, ...}
        """
        model_report = self.pricer.audit_report(lambda_h, lambda_a)
        edges = []
        
        # Canonical market codes first; legacy aliases kept for compatibility.
        mappings = {
            "1x2_h": ("1x2", "home"),
            "1x2_d": ("1x2", "draw"),
            "1x2_a": ("1x2", "away"),
            "o15": ("totals", "over_1.5"),
            "u35": ("totals", "over_3.5", True),
            "home_win": ("1x2", "home"),
            "draw": ("1x2", "draw"),
            "away_win": ("1x2", "away"),
            "over_2.5": ("totals", "over_2.5"),
            "under_2.5": ("totals", "over_2.5", True),  # True means invert key
        }

        # 1. 1X2 and Totals
        for market_key, report_path in mappings.items():
            if market_key not in market_odds:
                continue
            
            odds = market_odds[market_key]
            if not odds or odds <= 1.0:
                continue
                
            # Navigate the report dict
            if len(report_path) == 2:
                prob = model_report[report_path[0]][report_path[1]]
            else:
                # Handle "under" style markets by inverting their paired "over" key.
                prob = 1.0 - model_report[report_path[0]][report_path[1]]
                
            ev = (prob * odds) - 1.0
            
            if ev >= self.min_ev:
                edges.append({
                    "market": market_key,
                    "odds": odds,
                    "model_prob": prob,
                    "fair_odds": 1.0 / prob if prob > 0 else 999,
                    "ev": ev
                })
        
        return edges

    def apply_kelly_staking(self, ev: float, odds: float, fraction: float = 0.1) -> float:
        """
        Calculates recommended stake using Fractional Kelly Criterion.
        f = (bp - q) / b
        where b = odds - 1, p = prob, q = 1-p
        Simplified f = ev / (odds - 1)
        """
        if ev <= 0 or odds <= 1:
            return 0.0
        
        full_kelly = ev / (odds - 1)
        return full_kelly * fraction

if __name__ == "__main__":
    # Test Scenarios
    finder = ValueFinder(min_ev=0.02)
    
    # Imagine our model predicts 1.8 vs 1.1 (Home Win prob ~55%)
    # Bookie offers 2.10 (Value!)
    test_odds = {"1x2_h": 2.10, "o15": 1.45}
    
    found_edges = finder.find_edges(1.8, 1.1, test_odds)
    print("--- VALUE DISCOVERY TEST ---")
    for edge in found_edges:
        stake = finder.apply_kelly_staking(edge['ev'], edge['odds'], fraction=0.25)
        print(f"Edge Found: {edge['market']} @ {edge['odds']}")
        print(f"  EV: {edge['ev']:.1%}")
        print(f"  Recommended Stake: {stake:.1%} of bankroll")
