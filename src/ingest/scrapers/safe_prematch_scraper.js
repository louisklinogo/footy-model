const { PlaywrightCrawler } = require('crawlee');
const fs = require('fs');
const path = require('path');

/**
 * Safe Prematch Odds Scraper - v1
 * Focused only on grabbing CURRENT odds for upcoming fixtures.
 */

const FS_SIGN = 'SW9D1eZo';

function extractOdds(text) {
    const results = {};
    if (!text) return results;
    const tokens = text.split(/\s+/).filter(t => t.length > 0 && t !== '-');
    for (let i = 0; i < tokens.length - 2; i++) {
        const t1 = tokens[i];
        const t2 = tokens[i + 1];
        const t3 = tokens[i + 2];
        const isLine = t1.match(/^[+-]?\d(\.[0257]5?)?$/);
        const isOdds = t2.match(/^\d\.\d+$/) && t3.match(/^\d\.\d+$/);
        if (isLine && isOdds) {
            if (!results[t1]) results[t1] = { over: t2, under: t3 };
        }
    }
    return results;
}

function extract1X2(text) {
    if (!text) return {};
    const tokens = text.split(/\s+/).filter(t => t.length > 0 && t !== '-');
    for (let i = 0; i < tokens.length - 2; i++) {
        const t1 = tokens[i];
        const t2 = tokens[i + 1];
        const t3 = tokens[i + 2];
        if (t1.match(/^\d\.\d+$/) && t2.match(/^\d\.\d+$/) && t3.match(/^\d\.\d+$/)) {
            return { "1": t1, "X": t2, "2": t3 };
        }
    }
    return {};
}

async function main() {
    const outDir = path.join(__dirname, '..', 'data', 'v1', 'prematch_odds');
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

    // For this demonstration, we'll take a few upcoming match IDs from a league
    const leagueCode = process.argv[2] || 'E0';
    const idPath = path.join(__dirname, '..', 'data', 'v1', 'ids', `match_ids_${leagueCode}.json`);

    if (!fs.existsSync(idPath)) {
        console.error(`ID file not found: ${idPath}`);
        return;
    }

    const matches = JSON.parse(fs.readFileSync(idPath, 'utf-8')).slice(0, 10); // Limit for safety
    console.log(`🚀 Scraping fresh odds for ${matches.length} matches in ${leagueCode}...`);

    const crawler = new PlaywrightCrawler({
        async requestHandler({ page, request }) {
            const { id } = request.userData;
            const entry = { id, odds: { ou: {}, ah: {}, "1x2": {} }, scraped_at: new Date().toISOString() };

            try {
                await page.goto(`https://www.flashscore.com/match/${id}/#/odds-comparison/over-under/full-time`, { waitUntil: 'networkidle' });
                await page.waitForTimeout(3000);
                entry.odds.ou = extractOdds(await page.innerText('body'));

                await page.goto(`https://www.flashscore.com/match/${id}/#/odds-comparison/asian-handicap/full-time`, { waitUntil: 'networkidle' });
                await page.waitForTimeout(3000);
                entry.odds.ah = extractOdds(await page.innerText('body'));

                await page.goto(`https://www.flashscore.com/match/${id}/#/odds-comparison/1x2-odds/full-time`, { waitUntil: 'networkidle' });
                await page.waitForTimeout(3000);
                entry.odds["1x2"] = extract1X2(await page.innerText('body'));

                fs.writeFileSync(path.join(outDir, `${id}.json`), JSON.stringify(entry, null, 4));
                console.log(`✅ ${id}: Odds Captured`);
            } catch (e) {
                console.error(`❌ ${id}: ${e.message}`);
            }
        },
    });

    await crawler.run(matches.map(m => ({
        url: `https://www.flashscore.com/match/${m.id || m.flashscore_id}/`,
        userData: { id: m.id || m.flashscore_id }
    })));
}

main();
