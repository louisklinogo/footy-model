"""
Stake Calculator based on Elite Model Probability and Kelly Criterion.
Usage: python calculate_stake.py [prob] [odds] [bankroll]
"""

import sys

def calculate_kelly(prob, odds, bankroll, fraction=0.5):
    """
    prob: Model probability (e.g. 0.92 for 92%)
    odds: Current bookmaker odds (e.g. 1.45)
    bankroll: Current bankroll in GHS
    fraction: 0.5 for Half-Kelly (Senior DS Recommended)
    """
    if odds <= 1:
        return 0, 0
    
    implied_prob = 1 / odds
    edge = prob - implied_prob
    
    if edge <= 0:
        return edge, 0
    
    # Kelly Formula: (p*b - q) / b where b is net odds (odds - 1)
    # Simplified: (edge / (odds - 1))
    kelly_stake = (edge / (odds - 1)) * fraction
    
    actual_stake = bankroll * kelly_stake
    
    # Apply scientific guardrails
    final_stake = max(50, actual_stake) # 50 GHS Floor
    
    # Max exposure cap (10% of bankroll if bankroll > 500)
    if bankroll > 500:
        final_stake = min(final_stake, bankroll * 0.1)
        
    return edge, final_stake

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python calculate_stake.py [prob_pct] [odds] [bankroll]")
        print("Example: python calculate_stake.py 92 1.45 50")
        sys.exit(1)
        
    prob = float(sys.argv[1]) / 100
    odds = float(sys.argv[2])
    br = float(sys.argv[3])
    
    edge, stake = calculate_kelly(prob, odds, br)
    
    print("-" * 40)
    print(f"MODEL PROBABILITY: {prob:.1%}")
    print(f"BOOKMAKER ODDS:    {odds:.2f}")
    print(f"CURRENT BANKROLL:  {br:.2f} GHS")
    print("-" * 40)
    print(f"IMPLIED PROB:      {1/odds:.1%}")
    print(f"CALCULATED EDGE:   {edge:+.1%}")
    print("-" * 40)
    
    if edge > 0.03:
        print(f"✅ ACTION: BET")
        print(f"👉 STAKE:  {stake:.2f} GHS")
    elif edge > 0:
        print(f"⚠️ ACTION: CAUTION (Low Edge)")
        print(f"👉 STAKE:  {stake:.2f} GHS")
    else:
        print(f"❌ ACTION: NO BET (Negative Edge)")
    print("-" * 40)
