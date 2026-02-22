import numpy as np
from typing import Dict

class MatchSimulator:
    """
    Monte Carlo Football Match Simulator.
    Simulates goal timings to price complex 'Anytime' markets like SportyBet 1UP/2UP.
    """
    
    def __init__(self, iterations: int = 50000, seed: int | None = None):
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)

    def simulate_match(self, lambda_h: float, lambda_a: float) -> Dict[str, float]:
        """
        Simulates a match and tracks Lead events.
        """
        # Draws total goals from Poisson
        h_total_goals = self.rng.poisson(lambda_h, self.iterations)
        a_total_goals = self.rng.poisson(lambda_a, self.iterations)
        
        results = {
            "h_1up_pct": 0,
            "h_2up_pct": 0,
            "a_1up_pct": 0,
            "a_2up_pct": 0,
            "h_win_ft": 0
        }
        
        for i in range(self.iterations):
            h_goals = h_total_goals[i]
            a_goals = a_total_goals[i]
            
            # Simple FT Win
            if h_goals > a_goals:
                results["h_win_ft"] += 1
            
            # 1UP/2UP Logic
            # For 1UP (1 goal lead at any time):
            # If Home scores first, they hit 1UP. 
            # If Away scores first (0-1), Home can still hit 1UP if they later lead (2-1).
            # The simplest approximation for Poisson is that if a team scores ANY goal, 
            # and the other team hasn't scored yet OR they equalize and overtake, 
            # they hit 1UP.
            
            # For a more accurate simulation, we distribute goals randomly in time (90 mins).
            # This is a Poisson Process.
            h_times = np.sort(self.rng.uniform(0, 90, h_goals))
            a_times = np.sort(self.rng.uniform(0, 90, a_goals))
            
            # Walk through the timeline
            all_goals = []
            for t in h_times: all_goals.append((t, 'H'))
            for t in a_times: all_goals.append((t, 'A'))
            all_goals.sort() # Sort by time
            
            h_curr, a_curr = 0, 0
            h_max_lead, a_max_lead = 0, 0
            
            for _, team in all_goals:
                if team == 'H':
                    h_curr += 1
                else:
                    a_curr += 1
                
                h_max_lead = max(h_max_lead, h_curr - a_curr)
                a_max_lead = max(a_max_lead, a_curr - h_curr)
            
            if h_max_lead >= 1: results["h_1up_pct"] += 1
            if h_max_lead >= 2: results["h_2up_pct"] += 1
            if a_max_lead >= 1: results["a_1up_pct"] += 1
            if a_max_lead >= 2: results["a_2up_pct"] += 1
            
        return {k: v / self.iterations for k, v in results.items()}

if __name__ == "__main__":
    sim = MatchSimulator(iterations=50000, seed=42)
    # Typical tight match (1.2 vs 1.1)
    # 1UP is huge here because of the draw risk
    leagues_lambdas = [
        (1.2, 1.1),
        (2.5, 0.8)
    ]
    
    print("--- LEAD PROBABILITY SIMULATION (SportyBet Focus) ---")
    for lh, la in leagues_lambdas:
        res = sim.simulate_match(lh, la)
        print(f"\nMatch: H_Lambda={lh}, A_Lambda={la}")
        print(f"  Standard FT Home Win: {res['h_win_ft']:.1%}")
        print(f"  SportyBet Home 1UP  : {res['h_1up_pct']:.1%} (EDGE: {res['h_1up_pct'] - res['h_win_ft']:.1%})")
        print(f"  SportyBet Home 2UP  : {res['h_2up_pct']:.1%}")
