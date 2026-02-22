const { PlaywrightCrawler } = require('crawlee');
const fs = require('fs');
const path = require('path');

/**
 * Safe Prematch Enricher - Inherits logic from premium_enricher_v4.js
 * Goal: Capture CURRENT odds for upcoming matches (OU, AH, 1X2).
 */

const FS_SIGN = 'SW9D1eZo';

// --- INHERITED LOGIC BEGINS ---

function parseStats(text) {
    const stats = { home: {}, away: {} };
    if (!text) return stats;
    const blocks = text.split('~SD' + String.fromCharCode(0xf7));
    blocks.forEach(block => {
        const rows = block.split(String.fromCharCode(0xac));
        let name, h, a;
        rows.forEach(row => {
            if (row.startsWith('SG' + String.fromCharCode(0xf7))) name = row.split(String.fromCharCode(0xf7))[1];
            if (row.startsWith('SH' + String.fromCharCode(0xf7))) h = row.split(String.fromCharCode(0xf7))[1];
            if (row.startsWith('SI' + String.fromCharCode(0xf7))) a = row.split(String.fromCharCode(0xf7))[1];
        });
        if (name && h && a) {
            if (!stats.home[name]) { stats.home[name] = h; stats.away[name] = a; }
        }
    });
    return stats;
}

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

// --- NEW LOGIC: 1X2 EXTRACTION ---

function extract1X2(text) {
    const results = {};
    if (!text) return results;
    const tokens = text.split(/\s+/).filter(t => t.length > 0 && t !== '-');
    // Pattern: We look for 3 consecutive decimals (Home, Draw, Away)
    for (let i = 0; i < tokens.length - 2; i++) {
        const t1 = tokens[i];
        const t2 = tokens[i + 1];
        const t3 = tokens[i + 2];
        const isOddsValue = t => t.match(/^\d+\.\d+$/);
        if (isOddsValue(t1) && isOddsValue(t2) && isOddsValue(t3)) {
            // First bookie top-line
            return { "1": t1, "X": t2, "2": t3 };
        }
    }
    return results;
}

// --- ORCHESTRATION ---

async function main() {
    const args = process.argv.slice(2);
    const leagueCode = args[0] || 'E0';
    const idListJson = args[1]; // Passed as JSON string of IDs

    if (!idListJson) {
        console.error("Missing ID list.");
        process.exit(1);
    }

    const ids = JSON.parse(idListJson);
    const outDir = path.join(__dirname, '..', 'data', 'v1', 'prematch_json', leagueCode);
    fs.mkdirSync(outDir, { recursive: true });

    console.log(`🚀 Safe Enricher working on ${ids.length} matches in ${leagueCode}...`);

    const crawler = new PlaywrightCrawler({
        maxConcurrency: 3,
        async requestHandler({ page, request }) {
            const { id } = request.userData;
            const entry = { id, stats: {}, odds: { ou: {}, ah: {}, "1x2": {} }, scraped_at: new Date().toISOString() };

            try {
                // 1. Visit Main summary for stats/metadata
                await page.goto(`https://www.flashscore.com/match/${id}/#/match-summary`, { waitUntil: 'domcontentloaded' });
                const baseUrl = page.url().split('?')[0];

                // Fetch stats (Using same handshake logic)
                const statsRaw = await page.evaluate(async (mid) => {
                    const sign = 'SW9D1eZo';
                    const resp = await fetch(`https://global.flashscore.ninja/2/x/feed/df_st_1_${mid}`, { headers: { 'x-fsign': sign } });
                    return resp.text();
                }, id);
                entry.stats = parseStats(statsRaw);

                // 2. Over/Under
                await page.goto(`${baseUrl}odds/over-under/full-time/`, { waitUntil: 'domcontentloaded' });
                await page.waitForTimeout(3000);
                entry.odds.ou = extractOdds(await page.innerText('body'));

                // 3. Asian Handicap
                await page.goto(`${baseUrl}odds/asian-handicap/full-time/`, { waitUntil: 'domcontentloaded' });
                await page.waitForTimeout(3000);
                entry.odds.ah = extractOdds(await page.innerText('body'));

                // 4. 1X2 (New Safely Added)
                await page.goto(`${baseUrl}odds/1x2-odds/full-time/`, { waitUntil: 'domcontentloaded' });
                await page.waitForTimeout(3000);
                entry.odds["1x2"] = extract1X2(await page.innerText('body'));

                fs.writeFileSync(path.join(outDir, `${id}.json`), JSON.stringify(entry, null, 4));
                console.log(`✅ Success: ${id} (OU:${Object.keys(entry.odds.ou).length} AH:${Object.keys(entry.odds.ah).length} 1X2:${Object.keys(entry.odds["1x2"]).length})`);

            } catch (e) {
                console.error(`❌ Failed: ${id} | ${e.message}`);
            }
        },
    });

    await crawler.run(ids.map(id => ({
        url: `https://www.flashscore.com/match/${id}/`,
        userData: { id }
    })));
}

main().catch(err => { console.error(err); process.exit(1); });
