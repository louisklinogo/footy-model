const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function enrichMatch(matchId) {
    console.log(`🚀 Enriching Match: ${matchId}`);
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    const data = { id: matchId, stats: {}, odds: {} };

    try {
        // 1. STATS ENRICHMENT
        await page.goto(`https://www.flashscore.com/match/${matchId}/#/match-statistics/0`, { waitUntil: 'networkidle' });
        // Wait for rows to hydrate (wcl-row_2oCpS)
        await page.waitForSelector('.wcl-row_2oCpS', { timeout: 10000 }).catch(() => null);
        
        data.stats = await page.evaluate(() => {
            const statsObj = {};
            const rows = document.querySelectorAll('div[class*="row_"]');
            rows.forEach(row => {
                const home = row.querySelector('[class*="homeValue_"]')?.innerText;
                const name = row.querySelector('[class*="categoryName_"]')?.innerText;
                const away = row.querySelector('[class*="awayValue_"]')?.innerText;
                if (name && home && away) {
                    statsObj[name] = { home, away };
                }
            });
            return statsObj;
        });

        // 2. ODDS ENRICHMENT (Over/Under)
        await page.goto(`https://www.flashscore.com/match/${matchId}/#/odds-comparison/over-under/0`, { waitUntil: 'networkidle' });
        await page.waitForSelector('.ui-table__row', { timeout: 10000 }).catch(() => null);

        data.odds = await page.evaluate(() => {
            const ou = {};
            const rows = document.querySelectorAll('.ui-table__row');
            rows.forEach(row => {
                const text = row.innerText;
                // Format: "1.5\n1.26\n3.75"
                const parts = text.split('\n');
                if (parts.length >= 3) {
                    const line = parts[0];
                    const over = parts[1];
                    const under = parts[2];
                    if (line && line.includes('.')) {
                        ou[line] = { over, under };
                    }
                }
            });
            return ou;
        });

    } catch (e) {
        console.error(`❌ Error enriching ${matchId}: ${e.message}`);
    }

    await browser.close();
    return data;
}

async function main() {
    const leagueCode = process.argv[2] || 'E0';
    const idPath = path.join(__dirname, 'data', `match_ids_${leagueCode}.json`);
    
    if (!fs.existsSync(idPath)) {
        console.error(`Missing ID discovery file: ${idPath}`);
        return;
    }

    const matches = JSON.parse(fs.readFileSync(idPath));
    const outputDir = path.join(__dirname, 'data', 'premium', leagueCode);
    fs.mkdirSync(outputDir, { recursive: true });

    // Process first 5 as a test
    for (const match of matches.slice(0, 5)) {
        const result = await enrichMatch(match.id);
        fs.writeFileSync(path.join(outputDir, `${match.id}.json`), JSON.stringify(result, null, 4));
        console.log(`✅ Saved: ${match.home} vs ${match.away}`);
    }
}

main();
