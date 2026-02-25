const { PlaywrightCrawler } = require('crawlee');
const fs = require('fs');
const path = require('path');

/**
 * FBref Player Stats Scraper
 * 
 * Scrapes player stats from FBref for all leagues in fbref_league_mapping.json
 * Output: data/v1/player_stats/<league_code>.json
 * 
 * Stats extracted:
 * - Player name, team, position, nationality
 * - Minutes, games played
 * - Goals, assists, xG, xA
 * - Per 90 stats
 */

const LEAGUE_MAPPING_PATH = path.join(__dirname, '..', 'data', 'v1', 'fbref_league_mapping.json');
const OUTPUT_DIR = path.join(__dirname, '..', 'data', 'v1', 'player_stats');

// Data-stat attributes we're interested in
const STAT_FIELDS = {
    player: 'player',
    nationality: 'nationality',
    position: 'pos',
    age: 'age',
    squad: 'squad',
    games: 'games',
    minutes: 'minutes',
    goals: 'goals',
    assists: 'assists',
    cards_yellow: 'cards_yellow',
    cards_red: 'cards_red',
    xg: 'xg',
    xa: 'xa',
    goals_per90: 'goals_per90',
    assists_per90: 'assists_per90',
    xg_per90: 'xg_per90',
    xa_per90: 'xa_per90',
    shots_on_target_pct: 'shots_on_target_pct',
    tackles: 'tackles',
    interceptions: 'interceptions',
    blocks: 'blocks',
    passes_completed: 'passes_completed',
    passes_attempted: 'passes_attempted',
    pass_accuracy_pct: 'pass_accuracy_pct',
    key_passes: 'key_passes',
    dribbles_completed: 'dribbles_completed',
    dribbles_attempted: 'dribbles_attempted',
    fouls_drawn: 'fouls_drawn',
    fouls_committed: 'fouls_committed',
    offsides: 'offsides',
    crosses: 'crosses',
    touches: 'touches',
    touches_att_pen: 'touches_att_pen',
    live: 'live'
};

async function scrapeLeague(leagueCode, leagueInfo) {
    console.log(`\n🏆 Scraping ${leagueCode}: ${leagueInfo.fbref_name}`);
    
    const outPath = path.join(OUTPUT_DIR, `${leagueCode}.json`);
    
    // Check if already scraped
    if (fs.existsSync(outPath)) {
        console.log(`   ⏭️  Already exists, skipping`);
        return;
    }

    const crawler = new PlaywrightCrawler({
        maxConcurrency: 1,
        async requestHandler({ page }) {
            await page.goto(leagueInfo.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
            
            // Wait for Cloudflare and page to settle
            await page.waitForTimeout(5000);
            
            // Check if we hit Cloudflare
            const title = await page.title();
            if (title.includes('Cloudflare') || title.includes('Access denied')) {
                throw new Error('Cloudflare block detected');
            }

            // Extract player data from the stats table
            const players = await page.evaluate(() => {
                const output = [];
                
                // Find the main stats table - look for standard stats
                const table = document.querySelector('#stats_standard');
                if (!table) {
                    // Try alternative selectors
                    const tables = document.querySelectorAll('table');
                    for (const t of tables) {
                        const headers = Array.from(t.querySelectorAll('th'));
                        const hasPlayer = headers.some(h => h.dataset?.stat === 'player');
                        if (hasPlayer) {
                            // Check if it's the main player stats (notgk)
                            const tbody = t.querySelector('tbody');
                            if (tbody) {
                                const firstRow = tbody.querySelector('tr');
                                if (firstRow && !firstRow.classList.contains('partial_table')) {
                                    // Skip goalkeepers for now
                                    const posCell = firstRow.querySelector('[data-stat="pos"]');
                                    if (posCell?.textContent?.includes('GK')) {
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // More robust: find all player rows
                const rows = document.querySelectorAll('#stats_standard tbody tr');
                
                rows.forEach(row => {
                    // Skip subheader rows
                    if (row.classList.contains('thead') || row.classList.contains('partial_table')) {
                        return;
                    }
                    
                    // Skip empty rows
                    const playerCell = row.querySelector('[data-stat="player"]');
                    if (!playerCell) return;
                    
                    const playerLink = playerCell.querySelector('a');
                    if (!playerLink) return;
                    
                    const playerName = playerLink.textContent?.trim();
                    if (!playerName || playerName.length < 2) return;
                    
                    // Build player object
                    const player = { name: playerName };
                    
                    // Extract all data-stat attributes
                    const cells = row.querySelectorAll('[data-stat]');
                    cells.forEach(cell => {
                        const stat = cell.dataset.stat;
                        const value = cell.textContent?.trim();
                        if (value && value !== '-' && value !== '') {
                            player[stat] = isNaN(value) ? value : parseFloat(value);
                        }
                    });
                    
                    // Skip if no meaningful data
                    if (!player.games && player.games !== 0) return;
                    if (player.games < 1) return; // Must have played at least 1 game
                    
                    output.push(player);
                });
                
                return output;
            });

            console.log(`   ✅ Found ${players.length} players`);
            
            // Save to file
            const result = {
                league_code: leagueCode,
                fbref_name: leagueInfo.fbref_name,
                fbref_id: leagueInfo.fbref_id,
                url: leagueInfo.url,
                scraped_at: new Date().toISOString(),
                player_count: players.length,
                players: players
            };
            
            fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
            console.log(`   💾 Saved to ${outPath}`);
        },
    });

    try {
        await crawler.run([{ url: leagueInfo.url, userData: { leagueCode } }]);
    } catch (e) {
        console.log(`   ❌ Error: ${e.message}`);
    }
}

async function main() {
    // Load league mapping
    const mapping = JSON.parse(fs.readFileSync(LEAGUE_MAPPING_PATH, 'utf-8'));
    
    // Ensure output directory exists
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    
    // Get league codes from CLI args, or all
    const args = process.argv.slice(2);
    const targetLeagues = args.length > 0 ? args : Object.keys(mapping);
    
    console.log(`🏃 FBref Player Stats Scraper`);
    console.log(`   Output: ${OUTPUT_DIR}`);
    console.log(`   Leagues: ${targetLeagues.join(', ')}`);
    
    for (const leagueCode of targetLeagues) {
        if (!mapping[leagueCode]) {
            console.log(`\n⚠️  Unknown league code: ${leagueCode}`);
            continue;
        }
        
        await scrapeLeague(leagueCode, mapping[leagueCode]);
    }
    
    console.log(`\n✅ Done!`);
}

main().catch(err => { console.error(err); process.exit(1); });
