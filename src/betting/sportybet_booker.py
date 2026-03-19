"""
sportybet_booker.py - Automates booking bets on SportyBet Ghana.

Books bets to bet slip without placing them. User reviews and confirms manually.

Usage:
    python src/betting/sportybet_booker.py --accas artifacts/accas/today.json
    python src/betting/sportybet_booker.py --accas artifacts/accas/today.json --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from playwright.sync_api import Page, sync_playwright

# Load environment variables from .env file
load_dotenv()

# Constants
SPORTYBET_URL = "https://www.sportybet.com/gh/m/"
SIMILARITY_THRESHOLD = 0.70  # 70% match for team names
DEFAULT_TIMEOUT = 10000  # 10 seconds

# Market mapping: our code -> SportyBet UI
# SportyBet mobile UI uses specific tab names and line formats
MARKET_MAP: dict[str, dict[str, Any]] = {
    # Asian Handicap 2-way
    "ah2_home_p05": {"tab": "Handicap", "line": "+0.5", "selection": "Home"},
    "ah2_home_m05": {"tab": "Handicap", "line": "-0.5", "selection": "Home"},
    "ah2_home_p15": {"tab": "Handicap", "line": "+1.5", "selection": "Home"},
    "ah2_home_m15": {"tab": "Handicap", "line": "-1.5", "selection": "Home"},
    "ah2_away_p05": {"tab": "Handicap", "line": "+0.5", "selection": "Away"},
    "ah2_away_m05": {"tab": "Handicap", "line": "-0.5", "selection": "Away"},
    "ah2_away_p15": {"tab": "Handicap", "line": "+1.5", "selection": "Away"},
    "ah2_away_m15": {"tab": "Handicap", "line": "-1.5", "selection": "Away"},
    # 1X2 (Match Result)
    "1x2_h": {"tab": "1X2", "selection": "1"},
    "1x2_d": {"tab": "1X2", "selection": "X"},
    "1x2_a": {"tab": "1X2", "selection": "2"},
    # Over/Under 2.5
    "ou_2.5_over": {"tab": "O/U", "line": "2.5", "selection": "Over"},
    "ou_2.5_under": {"tab": "O/U", "line": "2.5", "selection": "Under"},
    # Over/Under 1.5
    "ou_1.5_over": {"tab": "O/U", "line": "1.5", "selection": "Over"},
    "ou_1.5_under": {"tab": "O/U", "line": "1.5", "selection": "Under"},
    # Over/Under 3.5
    "ou_3.5_over": {"tab": "O/U", "line": "3.5", "selection": "Over"},
    "ou_3.5_under": {"tab": "O/U", "line": "3.5", "selection": "Under"},
    # BTTS
    "btts_yes": {"tab": "GG/NG", "selection": "GG"},
    "btts_no": {"tab": "GG/NG", "selection": "NG"},
    # Double Chance
    "dc_1x": {"tab": "DC", "selection": "1X"},
    "dc_12": {"tab": "DC", "selection": "12"},
    "dc_x2": {"tab": "DC", "selection": "X2"},
}


@dataclass
class AccaLeg:
    """A single leg in an accumulator bet."""

    home_team: str
    away_team: str
    market_code: str
    odds: float | None = None

    def __str__(self) -> str:
        return f"{self.home_team} vs {self.away_team} -> {self.market_code}"


@dataclass
class Acca:
    """An accumulator bet with multiple legs."""

    legs: list[AccaLeg]
    name: str | None = None

    def __str__(self) -> str:
        leg_count = len(self.legs)
        return f"Acca ({leg_count} legs)"


def fuzzy_match_score(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings."""
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()
    return SequenceMatcher(None, s1_lower, s2_lower).ratio()


def find_match_by_teams(
    page: Page, home_team: str, away_team: str, threshold: float = SIMILARITY_THRESHOLD
) -> tuple[int, int] | None:
    """Find a match on the page by team names using fuzzy matching.

    Uses the reference bot's XPath structure:
    - Match container: //*[@id="importMatch"]/div[{fir_match}]/div/div[4]/div[{sec_match}]
    - Home team: .../div[1]/div/div[2]/div[1]
    - Away team: .../div[1]/div/div[2]/div[2]

    Returns tuple of (fir_match, sec_match) indices or None if not found.
    """
    print(f"  Finding match: {home_team} vs {away_team}")

    # Wait for matches to load
    try:
        page.wait_for_selector('#importMatch', timeout=DEFAULT_TIMEOUT)
    except Exception:
        print("    Match container not found")
        return None

    # Iterate through league containers (fir_match) and matches within (sec_match)
    # The reference bot iterates div[2] through div[100] for fir_match
    for fir_match in range(2, 101):
        # For each league container, iterate through matches
        # sec_match starts at 1 for each league
        for sec_match in range(1, 51):  # Up to 50 matches per league
            try:
                # Build XPath for team names
                home_xpath = f'//*[@id="importMatch"]/div[{fir_match}]/div/div[4]/div[{sec_match}]/div[1]/div/div[2]/div[1]'
                away_xpath = f'//*[@id="importMatch"]/div[{fir_match}]/div/div[4]/div[{sec_match}]/div[1]/div/div[2]/div[2]'

                home_elem = page.locator(f'xpath={home_xpath}')
                away_elem = page.locator(f'xpath={away_xpath}')

                # Check if both elements exist and are visible
                if home_elem.count() == 0 or away_elem.count() == 0:
                    # No more matches in this league, break to next league
                    break

                home_text = home_elem.first.text_content() or ""
                away_text = away_elem.first.text_content() or ""

                # Skip empty text
                if not home_text.strip() or not away_text.strip():
                    continue

                # Fuzzy match team names
                home_score = fuzzy_match_score(home_text, home_team)
                away_score = fuzzy_match_score(away_text, away_team)

                if home_score >= threshold and away_score >= threshold:
                    print(f"    Found match! (home: {home_score:.0%}, away: {away_score:.0%})")
                    print(f"      Home: {home_text}")
                    print(f"      Away: {away_text}")
                    return (fir_match, sec_match)

            except Exception:
                # Element not found, try next
                continue

    print("    No match found")
    return None


def click_market_tab(page: Page, market_info: dict[str, Any]) -> bool:
    """Click the correct market tab to show the odds grid.

    SportyBet mobile uses tabs like "1X2", "Handicap", "O/U", "GG/NG", "DC".
    """
    tab_name = market_info.get("tab", "1X2")
    print(f"  Clicking market tab: {tab_name}")

    # Common tab selectors on SportyBet mobile
    tab_selectors = [
        f"text={tab_name}",
        f"button:has-text('{tab_name}')",
        f"[class*='tab']:has-text('{tab_name}')",
        f".market-tabs button:has-text('{tab_name}')",
        f".m-tabs-item:has-text('{tab_name}')",
        f"div[role='tab']:has-text('{tab_name}')",
    ]

    for selector in tab_selectors:
        try:
            tab = page.locator(selector).first
            if tab.is_visible(timeout=2000):
                tab.click()
                print(f"    Clicked tab: {tab_name}")
                time.sleep(0.5)  # Wait for odds to load
                return True
        except Exception:
            continue

    print(f"    Tab not found: {tab_name}")
    return False


def click_odds(
    page: Page,
    fir_match: int,
    sec_match: int,
    selection: str,
) -> bool:
    """Click odds for a specific selection using reference bot XPath patterns.

    Uses the reference bot's XPath structure for 1X2 odds:
    - Home (1): //*[@id="importMatch"]/div[{fir_match}]/div/div[4]/div[{sec_match}]/div[2]/div[1]/div[1]
    - Draw (X): //*[@id="importMatch"]/div[{fir_match}]/div/div[4]/div[{sec_match}]/div[2]/div[1]/div[2]
    - Away (2): //*[@id="importMatch"]/div[{fir_match}]/div/div[4]/div[{sec_match}]/div[2]/div[1]/div[3]

    Args:
        page: Playwright page object
        fir_match: League container index
        sec_match: Match index within league
        selection: "1" (home), "X" (draw), or "2" (away)

    Returns True if odds clicked successfully, False otherwise.
    """
    # Map selection to XPath index
    selection_map = {
        "1": 1,  # Home win
        "X": 2,  # Draw
        "2": 3,  # Away win
    }

    div_index = selection_map.get(selection)
    if div_index is None:
        print(f"    Unknown selection: {selection}")
        return False

    # Build XPath for the odds element
    odds_xpath = f'//*[@id="importMatch"]/div[{fir_match}]/div/div[4]/div[{sec_match}]/div[2]/div[1]/div[{div_index}]'

    print(f"  Clicking odds for selection '{selection}' at XPath: {odds_xpath}")

    try:
        odds_elem = page.locator(f'xpath={odds_xpath}')
        if odds_elem.count() == 0:
            print(f"    Odds element not found")
            return False

        # Click the center of the element for reliability
        first_elem = odds_elem.first
        box = first_elem.bounding_box()
        if box:
            # Click at the center of the element
            center_x = box['x'] + box['width'] / 2
            center_y = box['y'] + box['height'] / 2
            page.mouse.click(center_x, center_y)
            print(f"    Clicked odds at ({center_x:.0f}, {center_y:.0f})")
            time.sleep(0.3)  # Wait for bet slip animation
            return True
        else:
            # Fallback to regular click
            first_elem.click()
            print(f"    Clicked odds (fallback)")
            return True

    except Exception as e:
        print(f"    Failed to click odds: {e}")
        return False


def click_market_tab(
    page: Page,
    fir_match: int,
    tab_name: str,
) -> bool:
    """Click a market tab for a specific match using reference bot XPath patterns.

    Uses the reference bot's XPath structure for market tabs:
    - 1X2: //*[@id="importMatch"]/div[{fir_match}]/div/div[3]/div[1]
    - BTS: //*[@id="importMatch"]/div[{fir_match}]/div/div[3]/div[3]

    Args:
        page: Playwright page object
        fir_match: League container index
        tab_name: "1X2" or "BTS"

    Returns True if tab clicked successfully, False otherwise.
    """
    # Map tab name to XPath index
    tab_map = {
        "1X2": 1,
        "BTS": 3,
        "GG/NG": 3,  # BTS is same as GG/NG
    }

    div_index = tab_map.get(tab_name)
    if div_index is None:
        print(f"    Unknown tab: {tab_name}")
        return False

    # Build XPath for the tab
    tab_xpath = f'//*[@id="importMatch"]/div[{fir_match}]/div/div[3]/div[{div_index}]'

    print(f"  Clicking market tab '{tab_name}' at XPath: {tab_xpath}")

    try:
        tab_elem = page.locator(f'xpath={tab_xpath}')
        if tab_elem.count() == 0:
            print(f"    Tab element not found")
            return False

        tab_elem.first.click()
        print(f"    Clicked tab: {tab_name}")
        time.sleep(0.5)  # Wait for odds to update
        return True

    except Exception as e:
        print(f"    Failed to click tab: {e}")
        return False


def find_and_click_odds(
    page: Page,
    market_info: dict[str, Any],
    fir_match: int,
    sec_match: int,
) -> bool:
    """Find and click the specific odds for a selection.

    Uses the reference bot's XPath patterns for reliability.

    Args:
        page: Playwright page object
        market_info: Market configuration dict with 'tab' and 'selection'
        fir_match: League container index from find_match_by_teams
        sec_match: Match index within league from find_match_by_teams

    Returns True if odds clicked successfully, False otherwise.
    """
    selection = market_info.get("selection", "")
    tab_name = market_info.get("tab", "1X2")

    print(f"  Looking for odds: {selection} (tab: {tab_name})")

    # For 1X2 market, use the direct XPath pattern
    if tab_name == "1X2":
        return click_odds(page, fir_match, sec_match, selection)

    # For other markets, we may need to click the market tab first
    # (This would require additional XPath patterns from the reference bot)
    # For now, fall back to generic selectors
    print(f"    Market '{tab_name}' not yet supported with XPath patterns")
    return False


def add_to_betslip(
    page: Page,
    market_info: dict[str, Any],
    fir_match: int,
    sec_match: int,
) -> bool:
    """Add a selection to the bet slip.

    Uses the match indices from find_match_by_teams to click odds.

    Args:
        page: Playwright page object
        market_info: Market configuration dict
        fir_match: League container index
        sec_match: Match index within league
    """
    # Find and click the odds
    if not find_and_click_odds(page, market_info, fir_match, sec_match):
        return False

    # Verify it was added to bet slip
    try:
        # Check for bet slip indicator
        betslip_selector = ".betslip-indicator, .m-betslip, [class*='betslip']"
        page.wait_for_selector(betslip_selector, timeout=3000)
        print("  Added to bet slip!")
        return True
    except Exception:
        print("  Warning: Could not verify bet slip addition")
        return True  # Assume success


def find_match_indices(page: Page, home_team: str, away_team: str) -> tuple[int, int] | None:
    """Find the match indices (fir_match, sec_match) for a specific match.

    This is a wrapper around find_match_by_teams for use in the booking flow.
    Returns the match indices or None if not found.
    """
    return find_match_by_teams(page, home_team, away_team)


def book_leg(page: Page, leg: AccaLeg) -> bool:
    """Book a single leg of an accumulator.

    Returns True if successfully added to bet slip.
    """
    print(f"\nBooking leg: {leg}")

    # Get market info
    market_info = MARKET_MAP.get(leg.market_code)
    if not market_info:
        print(f"  Unknown market code: {leg.market_code}")
        return False

    # Find the match on the current page
    match_indices = find_match_by_teams(page, leg.home_team, leg.away_team)
    if not match_indices:
        print(f"  Match not found: {leg.home_team} vs {leg.away_team}")
        return False

    fir_match, sec_match = match_indices

    # Add to bet slip
    success = add_to_betslip(page, market_info, fir_match, sec_match)

    return success


def book_acca(page: Page, acca: Acca) -> tuple[int, int]:
    """Book all legs of an accumulator.

    Returns tuple of (successful_legs, total_legs).
    """
    print(f"\n{'='*60}")
    print(f"Booking Acca: {acca}")
    print(f"{'='*60}")

    success_count = 0

    for i, leg in enumerate(acca.legs, 1):
        print(f"\n--- Leg {i}/{len(acca.legs)} ---")
        if book_leg(page, leg):
            success_count += 1
        else:
            print(f"  Failed to book leg {i}")

    return success_count, len(acca.legs)


def load_accas_from_json(path: str) -> list[Acca]:
    """Load accumulators from a JSON file.

    Expected format:
    {
        "accas": [
            {
                "legs": [
                    {
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "market_code": "ah2_home_m05",
                        "odds": 1.85
                    },
                    ...
                ]
            },
            ...
        ]
    }

    Also supports the Acca format from acca_builder.py.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Accas file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    accas = []

    # Handle different JSON formats
    accas_data = data.get("accas", [data] if "legs" in data else [])

    for acca_data in accas_data:
        legs = []
        for leg_data in acca_data.get("legs", []):
            # Handle both formats: teams as tuple or separate fields
            if "teams" in leg_data:
                home_team, away_team = leg_data["teams"]
            else:
                home_team = leg_data.get("home_team", "")
                away_team = leg_data.get("away_team", "")

            leg = AccaLeg(
                home_team=home_team,
                away_team=away_team,
                market_code=leg_data.get("market_code", ""),
                odds=leg_data.get("odds"),
            )
            legs.append(leg)

        if legs:
            acca = Acca(
                legs=legs,
                name=acca_data.get("name") or acca_data.get("narrative"),
            )
            accas.append(acca)

    return accas


def get_credentials() -> tuple[str, str] | None:
    """Get SportyBet credentials from environment variables.

    Returns tuple of (username, password) if both are set, None otherwise.
    """
    username = os.environ.get("SPORTY_USERNAME", "").strip()
    password = os.environ.get("SPORTY_PASSWORD", "").strip()

    if username and password:
        return (username, password)
    return None


def login_to_sportybet(page: Page, username: str, password: str) -> bool:
    """Login to SportyBet using credentials.

    Uses selectors from the reference bot's Login.py:
    - Balance check: span[id="j_balance"]
    - Phone input: div.m-phone input[name="phone"]
    - Password input: div.m-psd input[name="psd"]
    - Login button: button[name="logIn"]

    Returns True if login successful, False otherwise.
    """
    print("\nAttempting automatic login to SportyBet...")
    print(f"  Username: {username[:4]}****")  # Partially mask for privacy

    # First check if already logged in by looking for balance
    try:
        balance = page.locator('span[id="j_balance"]')
        if balance.count() > 0:
            balance_text = balance.text_content()
            print(f"Already logged in! Balance: {balance_text}")
            return True
    except Exception:
        pass

    print("Not logged in, attempting login...")

    # Wait for page to be ready
    page.wait_for_load_state("networkidle", timeout=DEFAULT_TIMEOUT)
    time.sleep(1.0)  # Extra wait for any dynamic content

    # Wait for phone input
    phone_input = page.locator('div.m-phone input[name="phone"]')
    if phone_input.count() == 0:
        print("Could not find phone input - might need to click login first")
        # Try to click login button to open login modal
        login_btn = page.locator('button:has-text("Login"), text="Login"')
        if login_btn.count() > 0:
            login_btn.first.click()
            time.sleep(2)

    # Enter phone number
    try:
        phone_input = page.wait_for_selector('div.m-phone input[name="phone"]', timeout=10000)
        phone_input.click()
        phone_input.fill(username)
        print("  Entered phone number")
    except Exception as e:
        print(f"  Could not find phone input: {e}")
        return False

    # Enter password
    try:
        password_input = page.wait_for_selector('div.m-psd input[name="psd"]', timeout=10000)
        password_input.click()
        password_input.fill(password)
        print("  Entered password")
    except Exception as e:
        print(f"  Could not find password input: {e}")
        return False

    # Click login button
    try:
        login_btn = page.wait_for_selector('button[name="logIn"]', timeout=10000)
        login_btn.click()
        print("  Clicked login button")
    except Exception as e:
        print(f"  Could not find login button: {e}")
        return False

    # Wait for balance to appear (login success)
    try:
        page.wait_for_selector('span[id="j_balance"]', timeout=15000)
        print("  Login successful!")
        return True
    except Exception:
        print("  Login failed - balance not found")
        return False


def wait_for_manual_login(page: Page, timeout: int = 120) -> bool:
    """Wait for manual login by checking for login indicators.

    Uses the balance selector from reference bot: span[id="j_balance"]

    Args:
        page: Playwright page object
        timeout: Maximum seconds to wait for login

    Returns True if login detected, False if timeout.
    """
    print(f"\nWaiting up to {timeout} seconds for you to log in...")
    print("Login URL: https://www.sportybet.com/gh/m/")

    start = time.time()
    while time.time() - start < timeout:
        # Check for logged-in indicators using reference bot's selector
        try:
            # Primary check: balance element from reference bot
            balance = page.locator('span[id="j_balance"]')
            if balance.count() > 0:
                print("Detected login (balance visible)")
                return True
            # Fallback checks
            if page.locator('text=Deposit').count() > 0:
                print("Detected login (deposit button visible)")
                return True
        except Exception:
            pass
        time.sleep(2)

    print("Timeout waiting for login")
    return False


def wait_for_login(page: Page, manual: bool = False, login_timeout: int = 120) -> bool:
    """Handle login to SportyBet.

    If credentials exist in environment and manual=False, try auto-login.
    If auto-login fails or credentials missing, fall back to manual login.

    Args:
        page: Playwright page object
        manual: If True, force manual login even if credentials exist
        login_timeout: Maximum seconds to wait for manual login

    Returns True if login successful, False otherwise.
    """
    if manual:
        print("Manual login requested (--manual-login flag)")
        return wait_for_manual_login(page, timeout=login_timeout)

    # Check for credentials
    credentials = get_credentials()
    if credentials:
        username, password = credentials
        if login_to_sportybet(page, username, password):
            return True
        else:
            print("\nAuto-login failed, falling back to manual login...")
            return wait_for_manual_login(page, timeout=login_timeout)
    else:
        print("\nNo credentials found in environment (SPORTY_USERNAME, SPORTY_PASSWORD)")
        print("Falling back to manual login...")
        return wait_for_manual_login(page, timeout=login_timeout)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automate booking bets on SportyBet Ghana"
    )
    parser.add_argument(
        "--accas",
        type=str,
        required=True,
        help="Path to JSON file containing accumulators",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no visible browser)",
    )
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="Skip login wait (assume already logged in)",
    )
    parser.add_argument(
        "--manual-login",
        action="store_true",
        help="Force manual login even if credentials exist in environment",
    )
    parser.add_argument(
        "--acca-index",
        type=int,
        default=0,
        help="Index of acca to book (default: 0, first)",
    )
    parser.add_argument(
        "--login-timeout",
        type=int,
        default=120,
        help="Seconds to wait for manual login (default: 120)",
    )

    args = parser.parse_args()

    # Load accas
    try:
        accas = load_accas_from_json(args.accas)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if not accas:
        print("No accumulators found in file")
        return 1

    print(f"Loaded {len(accas)} accumulator(s)")
    for i, acca in enumerate(accas):
        print(f"  {i}: {acca} - {len(acca.legs)} legs")

    # Select acca to book
    if args.acca_index >= len(accas):
        print(f"Error: Acca index {args.acca_index} out of range")
        return 1

    acca = accas[args.acca_index]
    print(f"\nSelected: {acca}")

    # Run browser automation
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=["--start-maximized"] if not args.headless else [],
        )

        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        )

        page = context.new_page()

        # Navigate to SportyBet
        print(f"\nNavigating to: {SPORTYBET_URL}")
        page.goto(SPORTYBET_URL)
        page.wait_for_load_state("networkidle", timeout=DEFAULT_TIMEOUT)

        # Handle login
        if not args.skip_login:
            if not wait_for_login(page, manual=args.manual_login, login_timeout=args.login_timeout):
                print("Login failed")
                browser.close()
                return 1

        # Book the acca
        success, total = book_acca(page, acca)

        # Summary
        print("\n" + "=" * 60)
        print("BOOKING SUMMARY")
        print("=" * 60)
        print(f"  Legs added: {success}/{total}")
        print(f"  Check bet slip in browser to confirm")
        print("=" * 60)

        if not args.headless:
            print("\nBrowser will stay open. Close manually or wait 30 seconds...")
            time.sleep(30)

        browser.close()

    return 0 if success == total else 1


if __name__ == "__main__":
    sys.exit(main())
