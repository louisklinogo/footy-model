# SofaScore Wrapper Evaluation

**Date:** 2026-02-22
**Status:** Evaluated, ready for implementation

---

## Summary

The `sofascore-wrapper` Python package CAN replace Flashscore scraping for prematch data (injuries + odds). It provides structured API access to SofaScore's data, including injury/availability information.

---

## Key Discovery: Injury Data IS Available

### Where to Find Injuries

| Endpoint | Data |
|----------|------|
| `/event/{match_id}/lineups` | `missingPlayers[]` - players out for specific match |
| `/team/{team_id}/players` | `injury{}` object per player - detailed injury status |

### Injury Reason Codes

```python
INJURY_REASONS = {
    1: "Injury",
    2: "Suspended",
    3: "International Duty",
    4: "COVID-19",
    5: "Rest",
    6: "Unknown",
    7: "Knock",
    8: "Illness",
    9: "Personal Reasons",
    10: "Transfer Pending",
    11: "Not in Squad",
    12: "Doubtful",
    13: "Coach Decision",
    14: "Yellow Card Suspension",
    15: "Red Card Suspension",
}
```

### Sample Output Format

```json
{
  "id": "14025215",
  "availability": {
    "home": {
      "missing": [
        {
          "name": "Boubacar Kamara",
          "position": "M",
          "reason": "Injury",
          "reason_code": 1
        }
      ]
    },
    "away": {
      "missing": [
        {
          "name": "Noah Okafor",
          "position": "F",
          "reason": "Injury",
          "reason_code": 1
        }
      ]
    }
  }
}
```

---

## Comparison: SofaScore vs Flashscore

| Feature | Flashscore Scraper | sofascore-wrapper |
|---------|-------------------|-------------------|
| Injuries/Missing | ✅ Scrapes from lineups | ✅ Via API |
| Player positions | ❌ | ✅ Included |
| Reason codes | ❌ Text only | ✅ Structured codes |
| Questionable players | ✅ Separate category | ❌ Combined with missing |
| Odds | ✅ O/U, AH, 1X2 | ✅ `match_odds()`, `featured_odds()` |
| Fixtures discovery | Manual ID list | ✅ `games_by_date()` |
| Team squad | ❌ | ✅ `team.squad()` |
| H2H data | ❌ | ✅ `match.h2h()` |
| Implementation | JS + Playwright + DOM parsing | Python async API |

---

## Technical Requirements

### Dependencies

```bash
pip install sofascore-wrapper
python -m playwright install chromium
```

### How It Works

```python
# sofascore-wrapper uses Playwright under the hood
from sofascore_wrapper.api import SofascoreAPI

api = SofascoreAPI()  # Spins up headless Chromium
lineups = await api._get(f"/event/{match_id}/lineups")
await api.close()  # Must close to free browser
```

**Important:** Each API call uses a headless browser. Not lightweight.

---

## Automation Options (Real Costs)

### Estimated Monthly Usage

| Task | Frequency | Duration | Monthly |
|------|-----------|----------|---------|
| Daily pipeline | 1x/day | 15-20 min | ~500 min |
| Hourly tick | 24x/day | 5-10 min | ~3,600 min |
| **Total** | | | **~4,100 min** |

### Option 1: Your Own PC (FREE)

| Cost | Requirements |
|------|--------------|
| **$0** | PC must be on at scheduled times |

**Scheduler:** Windows Task Scheduler (built-in, no install)

```powershell
# Create daily task at 6AM
$action = New-ScheduledTaskAction -Execute "python" -Argument "C:\Developer\soccer\footy-model\src\pipelines\sofascore_prematch_pipeline.py"
$trigger = New-ScheduledTaskTrigger -Daily -At 6am
Register-ScheduledTask -TaskName "PrematchPipeline" -Action $action -Trigger $trigger
```

### Option 2: VPS (~$5/month)

| Provider | Cost | Specs |
|----------|------|-------|
| Hetzner CX22 | ~$4/mo | 2 vCPU, 4GB RAM |
| DigitalOcean | $6/mo | 1 vCPU, 1GB RAM |
| Linode | $5/mo | 1 vCPU, 1GB RAM |

**Setup:** Install Python + Playwright + cron

### Option 3: trigger.dev (~$16/month)

| Tier | Monthly | Included | Your cost |
|------|---------|----------|-----------|
| Free | $0 | $5 compute credit | ~$16 (overage) |
| Hobby | $10 | $10 compute credit | ~$11 |

**Pros:** Cloud-managed, retries, logging
**Cons:** Exceeds free tier, needs account

### Option 4: GitHub Actions (~$25-35/month)

| Cost Breakdown | Amount |
|----------------|--------|
| 4,100 min usage | ~$16.80 (after 2000 free) |
| Browser install overhead | ~$5-10 |
| New 2026 platform fee | ~$8.20 |
| **Total** | ~$25-35/mo |

**Cons:** No persistent browsers, new fees coming March 2026

---

## Recommendation

| Budget | Best Option |
|--------|-------------|
| **$0** | Run on your PC with Windows Task Scheduler |
| **$5/mo** | Cheap VPS + cron |
| **$15+/mo** | trigger.dev (easiest, fully managed) |

---

## Files Created During Testing

```
test_sofascore_wrapper.py     - Basic API test
test_sofascore_lineups.py     - Lineups exploration
test_sofascore_injuries.py    - Injury data extraction
test_sofascore_final.py       - Full extraction with reason codes
```

---

## Next Steps (When Ready to Implement)

1. Create `src/pipelines/sofascore_prematch_pipeline.py`
   - Get fixtures from DB
   - Fetch injuries/odds via sofascore-wrapper
   - Write to PostgreSQL

2. Create `src/scrapers/sofascore_fetcher.py`
   - Reusable module for injury/odds fetching
   - Maps SofaScore format to existing DB schema

3. Set up Windows Task Scheduler
   - Daily run at 6AM UTC
   - Optional: hourly tick for live updates

4. Update `ingest_prematch_v1.py` if needed
   - Ensure compatibility with new data format

---

## References

- **PyPI:** https://pypi.org/project/sofascore-wrapper/
- **GitHub:** https://github.com/tommhe14/sofascore-wrapper
- **Docs:** https://tommhe14.github.io/sofascore-wrapper/
- **trigger.dev pricing:** https://trigger.dev/pricing
- **GitHub Actions pricing:** https://github.com/resources/insights/2026-pricing-changes-for-github-actions


---

## Post-Match Data Availability

### Match Statistics

| Endpoint | Method | Data Available |
|----------|--------|----------------|
| `/event/{id}/incidents` | `match.incidents()` | Goals, cards, substitutions, penalties |
| `/event/{id}/lineups` | `api._get()` | Player ratings, positions, missing players |
| `/event/{id}/best-players/summary` | `match.motm()`, `match.best_home_players()` | MOTM, best rated players per team |

### Player Match Statistics (from lineups)

```json
{
  "player": {
    "name": "Bukayo Saka",
    "position": "F",
    "jerseyNumber": "7"
  },
  "rating": 7.2,
  "substitute": false,
  "shirtNumber": 7
}
```

### xG Data

Match objects include `hasXg: true` flag. xG values available via incidents:

```python
incidents = await match.incidents()
# incidents include xG data for shots
```

---

## Season-Level Player Statistics

### League.top_players(season)

Returns top players by various metrics:

| Category | Fields Available |
|----------|------------------|
| **Attack** | `goals`, `assists`, `expectedGoals (xG)`, `expectedAssists (xA)`, `rating` |
| **Passing** | `accuratePasses`, `keyPasses`, `bigChancesCreated` |
| **Defense** | `tackles`, `interceptions`, `clearances`, `cleanSheet`, `saves`, `goalsPrevented` |
| **Duels** | `groundDuelsWon`, `aerialDuelsWon`, `totalDuelsWon` |
| **Discipline** | `yellowCards`, `redCards`, `fouls` |

### Player.league_stats(league_id, season)

Full statistics breakdown per player per season:

```python
stats = await player.league_stats(league_id=17, season=61627)
# Returns 100+ stat fields including:
# - goals, assists, xG, xA
# - tackles, interceptions, clearances
# - passes, dribbles, duels won
# - minutes played, appearances
```

### Complete Stat Categories

```python
STAT_CATEGORIES = {
    # Attack
    "goals": "Total goals scored",
    "assists": "Total assists",
    "expectedGoals": "xG (expected goals)",
    "expectedAssists": "xA (expected assists)",
    "bigChancesCreated": "Big chances created",
    "bigChancesMissed": "Big chances missed",
    "shotsOnTarget": "Shots on target",
    "shotsOffTarget": "Shots off target",
    
    # Defense
    "tackles": "Total tackles",
    "tacklesWon": "Tackles won",
    "interceptions": "Interceptions",
    "clearances": "Clearances",
    "cleanSheet": "Clean sheets (G/D)",
    "saves": "Saves (GK)",
    "goalsPrevented": "Goals prevented (GK)",
    "blockedShots": "Blocked shots",
    
    # Passing
    "accuratePasses": "Accurate passes",
    "accuratePassesPercentage": "Pass accuracy %",
    "keyPasses": "Key passes",
    "accurateLongBalls": "Accurate long balls",
    "accurateCrosses": "Accurate crosses",
    
    # Duels & Dribbles
    "successfulDribbles": "Successful dribbles",
    "groundDuelsWon": "Ground duels won",
    "aerialDuelsWon": "Aerial duels won",
    "totalDuelsWon": "Total duels won",
    
    # Discipline
    "yellowCards": "Yellow cards",
    "redCards": "Red cards",
    "fouls": "Fouls committed",
    "wasFouled": "Times fouled",
    
    # General
    "rating": "Average match rating (0-10)",
    "appearances": "Total appearances",
    "minutesPlayed": "Total minutes played",
    "matchesStarted": "Matches started",
}
```

---

## Team Standings & Home/Away Splits

### League.standings(season)

```python
standings = await league.standings(season=61627)
for row in standings["standings"][0]["rows"]:
    print(f"{row['team']['name']}: {row['points']} pts, GF:{row['scoresFor']}, GA:{row['scoresAgainst']}")
```

### Home/Away Splits

```python
home_standings = await league.standings_home(season=61627)
away_standings = await league.standings_away(season=61627)

# Example: Arsenal defensive record
# HOME: 8 GA in 13 games (0.62 GA/game)
# AWAY: 13 GA in 15 games (0.87 GA/game)
```

---

## Player Impact Calculation

### Attack Impact Formula

```python
def calculate_attack_impact(player_stats, team_standings, team_avg_rating):
    """
    Calculate how much a player contributes to team attack.
    Returns score 0.0-1.0 where 1.0 = most impactful.
    """
    goals = player_stats.get("goals", 0)
    assists = player_stats.get("assists", 0)
    xG = player_stats.get("expectedGoals", 0)
    xA = player_stats.get("expectedAssists", 0)
    rating = player_stats.get("rating", 0)
    appearances = player_stats.get("appearances", 0)
    
    team_GF = team_standings.get("scoresFor", 0)
    team_matches = team_standings.get("matches", 1)
    
    # Goal contribution share (goals + assists as % of team GF)
    goal_share = (goals + assists) / team_GF if team_GF > 0 else 0
    
    # xG contribution (expected goals vs actual - shows quality)
    xG_delta = (goals - xG) if goals > 0 else 0
    
    # Rating above team average
    rating_delta = (rating - team_avg_rating) / 10 if rating > 0 else 0
    
    # Appearance consistency
    appearance_ratio = appearances / team_matches if team_matches > 0 else 0
    
    # Weighted composite score
    impact = (
        goal_share * 0.35 +           # Direct goal contribution
        min(xG_delta / 5, 0.15) +      # Overperformance bonus (capped)
        max(rating_delta, 0) * 0.25 +  # Rating above average
        appearance_ratio * 0.25        # Consistency/availability
    )
    
    return min(impact, 1.0)
```

### Defense Impact Formula

```python
def calculate_defense_impact(player_stats, team_standings, position: str):
    """
    Calculate how much a player contributes to team defense.
    Position weights: G=1.5, D=1.2, M=0.8, F=0.3
    Returns score 0.0-1.0.
    """
    position_weights = {"G": 1.5, "D": 1.2, "M": 0.8, "F": 0.3}
    pos_factor = position_weights.get(position, 0.5)
    
    tackles = player_stats.get("tackles", 0)
    interceptions = player_stats.get("interceptions", 0)
    clearances = player_stats.get("clearances", 0)
    clean_sheets = player_stats.get("cleanSheet", 0)
    saves = player_stats.get("saves", 0)
    goals_prevented = player_stats.get("goalsPrevented", 0)
    rating = player_stats.get("rating", 0)
    appearances = player_stats.get("appearances", 1)
    
    team_GA = team_standings.get("scoresAgainst", 0)
    team_matches = team_standings.get("matches", 1)
    
    # Per-game defensive actions
    def_actions_per_game = (tackles + interceptions + clearances) / appearances
    
    # Clean sheet contribution (for GK/D)
    cs_ratio = clean_sheets / appearances if appearances > 0 else 0
    
    # Goals prevented (GK specific)
    gk_impact = (saves + goals_prevented) / (team_GA + 1) if team_GA > 0 else 0
    
    # Composite score
    impact = (
        min(def_actions_per_game / 10, 0.4) * pos_factor +  # Defensive actions
        cs_ratio * 0.3 * pos_factor +                        # Clean sheets
        gk_impact * 0.3 if position == "G" else 0 +          # GK saves
        (rating / 10) * 0.2                                  # Overall rating
    )
    
    return min(impact, 1.0)
```

### Home/Away Intensity Impact

```python
def calculate_home_away_impact(player_stats, home_standings, away_standings):
    """
    Calculate how player absence affects home vs away performance.
    Returns dict with home_impact and away_impact scores.
    """
    home_GA = home_standings.get("scoresAgainst", 0)
    home_matches = home_standings.get("matches", 1)
    home_GA_per_game = home_GA / home_matches
    
    away_GA = away_standings.get("scoresAgainst", 0)
    away_matches = away_standings.get("matches", 1)
    away_GA_per_game = away_GA / away_matches
    
    # Defensive impact is more critical where team concedes more
    defense_criticality = {
        "home": away_GA_per_game / (home_GA_per_game + away_GA_per_game),
        "away": home_GA_per_game / (home_GA_per_game + away_GA_per_game)
    }
    
    return defense_criticality

# Example usage:
# Arsenal: HOME 0.62 GA/game, AWAY 0.87 GA/game
# => Defense more critical away (0.58) than home (0.42)
```

---

## Wrapper Source Code Structure

```
vendor/sofascore-wrapper/sofascore_wrapper/
├── __init__.py
├── api.py              # Core API client (Playwright-based)
├── league.py           # League class (standings, top_players, seasons)
├── match.py            # Match class (odds, incidents, h2h, lineups)
├── player.py           # Player class (stats, attributes, history)
├── team.py             # Team class (squad, fixtures, performance)
├── search.py           # Search functionality
├── manager.py          # Manager data
├── transfers.py        # Transfer data
├── news.py             # News data
├── user_data.py        # User preferences
├── flag.py             # Flag images
└── tools/
    └── enums.json      # Sport/position enums
```

### api.py - Core Client

```python
class SofascoreAPI:
    BASE_URL = "https://www.sofascore.com/api/v1"
    
    async def _get(self, endpoint):
        # Spins up headless Chromium
        # Makes request to API endpoint
        # Returns JSON response
    
    async def close():
        # Must call to free browser resources
```

### Key Methods Summary

| Class | Method | Purpose |
|-------|--------|---------|
| `League` | `standings(season)` | Total table |
| `League` | `standings_home(season)` | Home table |
| `League` | `standings_away(season)` | Away table |
| `League` | `top_players(season)` | Top players by rating/stats |
| `League` | `current_season()` | Get current season ID |
| `Match` | `match_odds()` | All odds markets |
| `Match` | `featured_odds()` | Featured odds (1X2, AH) |
| `Match` | `h2h()` | Head-to-head history |
| `Match` | `incidents()` | Match events |
| `Match` | `motm()` | Man of the match |
| `Match` | `games_by_date(sport, date)` | Fixtures by date |
| `Player` | `league_stats(league_id, season)` | Full season stats |
| `Player` | `attributes()` | Skill ratings (attacking, defending, etc.) |
| `Team` | `squad()` | Current squad with player details |
| `Team` | `performance()` | Recent match performance |
| `Team` | `last_fixtures()` | Recent results |

---

## Data Availability Matrix

| Data Type | Prematch | Post-Match | Season-Level |
|-----------|----------|------------|--------------|
| Injuries/Missing | ✅ lineups | - | - |
| Odds (1X2, AH, O/U) | ✅ `match_odds()` | - | - |
| Player positions | ✅ lineups | ✅ lineups | ✅ squad |
| Player ratings | - | ✅ lineups | ✅ `top_players()` |
| Goals/Assists | - | ✅ incidents | ✅ `league_stats()` |
| xG/xA | - | ✅ incidents | ✅ `league_stats()` |
| Tackles/Interceptions | - | - | ✅ `league_stats()` |
| Clean Sheets | - | - | ✅ `league_stats()` |
| Team GF/GA | - | - | ✅ `standings()` |
| Home/Away splits | - | - | ✅ `standings_home/away()` |
| H2H history | ✅ `h2h()` | ✅ `h2h()` | - |

---

## Files Created During Testing

```
test_sofascore_wrapper.py      - Basic API test
test_sofascore_lineups.py      - Lineups exploration
test_sofascore_injuries.py     - Injury data extraction
test_sofascore_final.py        - Full extraction with reason codes
test_upcoming_availability.py  - Prematch data for upcoming fixtures
test_postmatch_stats.py        - Post-match data (xG, player stats)
test_correct_endpoints.py      - Correct League class usage
test_defensive_impact.py       - Defensive player impact data
test_player_impact.py          - Player valuation metrics
```

---

## Current Season IDs (2025-26)

| League | League ID | Season ID |
|--------|-----------|-----------|
| Premier League | 17 | 76986 |
| La Liga | 8 | TBD |
| Bundesliga | 35 | TBD |
| Serie A | 23 | TBD |
| Ligue 1 | 34 | TBD |

Note: Always verify season ID via `league.current_season()` as they change yearly.

---

## References

- **PyPI:** https://pypi.org/project/sofascore-wrapper/
- **GitHub:** https://github.com/tommhe14/sofascore-wrapper
- **Docs:** https://tommhe14.github.io/sofascore-wrapper/
- **Local clone:** `vendor/sofascore-wrapper/`