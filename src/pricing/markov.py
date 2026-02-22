import numpy as np
from scipy.linalg import expm
from typing import Dict

class MarkovPricer:
    """
    Continuous-Time Markov Chain (CTMC) Pricing Engine.
    Uses Matrix Exponentials to calculate exact 'Anytime' probabilities.
    Standard Poisson only tells us the final result; CTMC tells us the path.
    """

    def __init__(self, max_goals: int = 5):
        self.max_goals = max_goals
        # Total number of score states: (max+1) * (max+1)
        self.num_states = (max_goals + 1)**2

    def _get_state_idx(self, h: int, a: int) -> int:
        return h * (self.max_goals + 1) + a

    def calculate_lead_probs(self, lambda_h: float, lambda_a: float) -> Dict[str, float]:
        """
        Calculates the probability of a team leading by N goals at ANY point 
        during the 90 minutes. 
        
        We do this by making 'Lead States' absorbing in the Transition Matrix (Q).
        """
        # We'll calculate for Home 1UP, Home 2UP, Away 1UP, Away 2UP
        results = {}
        
        for market in ["h_1up", "h_2up", "a_1up", "a_2up"]:
            # Q is the Generator Matrix: Q_ij is the transition rate from state i to j
            Q = np.zeros((self.num_states, self.num_states))
            
            for h in range(self.max_goals + 1):
                for a in range(self.max_goals + 1):
                    idx = self._get_state_idx(h, a)
                    
                    # Define Absorption Condition (Once you hit 1UP/2UP, you stay there)
                    is_absorbing = False
                    if market == "h_1up" and h > a: is_absorbing = True
                    elif market == "h_2up" and h - a >= 2: is_absorbing = True
                    elif market == "a_1up" and a > h: is_absorbing = True
                    elif market == "a_2up" and a - h >= 2: is_absorbing = True
                    
                    if is_absorbing:
                        # Absorbing state: no transitions OUT
                        continue
                    
                    # Rates of scoring
                    # Transition to (h+1, a)
                    if h + 1 <= self.max_goals:
                        next_idx = self._get_state_idx(h + 1, a)
                        Q[idx, next_idx] = lambda_h
                    
                    # Transition to (h, a+1)
                    if a + 1 <= self.max_goals:
                        next_idx = self._get_state_idx(h, a + 1)
                        Q[idx, next_idx] = lambda_a
                        
                    # Diagonal: Sum of out-rates (with negative sign)
                    Q[idx, idx] = - (lambda_h + lambda_a)

            # Solve the Kolmogorov Forward Equation: P(t) = exp(Q*t)
            # t=1 represents the full match (lambda is per-match intensity)
            P_matrix = expm(Q * 1.0)
            
            # Start state is (0,0)
            start_idx = self._get_state_idx(0, 0)
            
            # Sum probabilities of all 'absorbing' states
            prob_hit = 0.0
            for h in range(self.max_goals + 1):
                for a in range(self.max_goals + 1):
                    idx = self._get_state_idx(h, a)
                    success = False
                    if market == "h_1up" and h > a: success = True
                    elif market == "h_2up" and h - a >= 2: success = True
                    elif market == "a_1up" and a > h: success = True
                    elif market == "a_2up" and a - h >= 2: success = True
                    
                    if success:
                        prob_hit += P_matrix[start_idx, idx]
            
            results[market] = float(prob_hit)
            
        return results

if __name__ == "__main__":
    markov = MarkovPricer(max_goals=6)
    lh, la = 1.3, 1.1 # Tight match
    
    probs = markov.calculate_lead_probs(lh, la)
    print("--- MARKOV CHAIN (CTMC) PRICING REPORT ---")
    print(f"Match Intensities: H={lh} | A={la}")
    print(f"  Home 1UP Probability: {probs['h_1up']:.2%}")
    print(f"  Home 2UP Probability: {probs['h_2up']:.2%}")
    print(f"  Away 1UP Probability: {probs['a_1up']:.2%}")
    print(f"  Away 2UP Probability: {probs['a_2up']:.2%}")
