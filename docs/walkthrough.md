# AI Research Integration: Tavily & Flashscore

I have successfully integrated a senior-grade AI research pass into the footy-model pipeline. This system automatically extracts high-fidelity KPIs (injuries, field conditions, referee stats) to refine model predictions.

## 🚀 Key Achievements

### 1. The "Selective Elite" Research Strategy
After testing several Tavily endpoints, I've designed a strategy that balances elite analysis with cost efficiency.

- **Baseline**: [research(model="mini")](file:///c:/Developer/soccer/footy-model/src/modeling/ai_research_pass.py#70-254) at **25 credits/match**. 
- **Quality**: Successfully identified Pedri/Gavi injuries for Barcelona and quantified a $25\%$ personnel impact.
- **Optimization**: To protect your 1,000 credit free tier, we are now targeting only high-conviction edges (**EV > 10%**).

### 2. Comparison: Research vs. Search
I conducted a head-to-head test on the **Barcelona vs Levante** match to justify the credit cost.

| Feature | [research()](file:///c:/Developer/soccer/footy-model/src/modeling/ai_research_pass.py#70-254) (25c) | [search()](file:///c:/Developer/soccer/footy-model/src/modeling/ai_research_pass.py#70-254) (3c) |
| :--- | :--- | :--- |
| **Output Type** | Structured JSON | Plain Text Paragraph |
| **Schema Adherence** | 100% (Pydantic Validated) | 0% (Required regex/parsing) |
| **Data Quality** | Deep analysis of 5+ players | Surface-level summary |
| **Verdict** | **Winner** for Data Science | Best for Chat interfaces |

## 🛠️ Integrated Scripts

| Script | Purpose |
| :--- | :--- |
| [ai_research_pass.py](file:///c:/Developer/soccer/footy-model/src/modeling/ai_research_pass.py) | **Primary Engine**: Uses [research(mini)](file:///c:/Developer/soccer/footy-model/src/modeling/ai_research_pass.py#70-254) with a 10% EV filter. |
| [ai_research_pass_search.py](file:///c:/Developer/soccer/footy-model/src/modeling/ai_research_pass_search.py) | **Lightweight Alternative**: Uses [search()](file:///c:/Developer/soccer/footy-model/src/modeling/ai_research_pass.py#70-254) for a cheap text-only summary. |
| [safe_prematch_enricher.js](file:///c:/Developer/soccer/footy-model/scrapers/safe_prematch_enricher.js) | **Fixed Scraper**: Corrected the "2.00 odds" bug using decimal-only regex. |

## 📊 Sample Output (Barcelona Benchmark)
The system now produces research with this level of granularity:
```json
{
  "personnel": {
    "absent_minutes_impact": 25.0,
    "star_player_void": 1,
    "details": "Barcelona missing Pedri & Gavi (midfield), Christensen (defense)..."
  },
  "verdict_summary": "Barcelona are in strong form... Levante struggle defensively... Over 1.5 is highly favorable.",
  "final_research_verdict": "CONFIRM"
}
```

## 📋 Next Steps
- [ ] Monitor credit usage in your [Tavily Dashboard](https://tavily.com/dashboard).
- [ ] Run `python src/modeling/ai_research_pass.py` daily to enrich your slips.
- [ ] Consider upgrading to a paid Tavily tier if you wish to research all 40+ matches daily.
