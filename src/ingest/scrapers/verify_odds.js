const { PlaywrightCrawler } = require('crawlee');
const fs = require('fs');
const path = require('path');

/**
 * Premium Flashscore Enricher - Robust Verification Version v2
 * Focus: Verifying O1.5 and O2.5 Odds extraction via refined DOM interaction.
 */

const FS_SIGN = 'SW9D1eZo';
const DELIM_KV = String.fromCharCode(0xf7); // ÷
const DELIM_ROW = String.fromCharCode(0xac); // ¬

function parseStats(text) {
    const stats = { home: {}, away: {} };
    const matchSection = text.split('~SF' + DELIM_KV)[1] || '';
    const parts = matchSection.split('~SD' + DELIM_KV);
    
    parts.forEach(part => {
        const rows = part.split(DELIM_ROW);
        let name = null;
        let h = null;
        let a = null;
        
        rows.forEach(row => {
            if (row.startsWith('SG' + DELIM_KV)) name = row.split(DELIM_KV)[1];
            if (row.startsWith('SH' + DELIM_KV)) h = row.split(DELIM_KV)[1];
            if (row.startsWith('SI' + DELIM_KV)) a = row.split(DELIM_KV)[1];
        });
        
        if (name && h !== null && a !== null) {
            stats.home[name] = h;
            stats.away[name] = a;
        }
    });
    return stats;
}

async function main() {
    const leagueCode = 'E0';
    const idPath = path.join(__dirname, '..', 'data', 'scraper', `match_ids_${leagueCode}.json`);
    const outputDir = path.join(__dirname, '..', 'data', 'scraper', 'premium', 'test_verify');
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    const matches = JSON.parse(fs.readFileSync(idPath));
    const testMatches = matches.slice(0, 5);

    console.log(`🚀 Starting Robust Odds Verification for ${testMatches.length} matches...`);

    const crawler = new PlaywrightCrawler({
        async requestHandler({ page, request }) {
            const { id } = request.userData;
            const entry = { id, stats: {}, odds: {} };

            try {
                // 1. STATS (Direct Feed)
                const statsUrl = `https://global.flashscore.ninja/2/x/feed/df_st_1_${id}`;
                const statsRaw = await page.evaluate(async ({ url, sign }) => {
                    const resp = await fetch(url, { headers: { 'x-fsign': sign } });
                    return resp.text();
                }, { url: statsUrl, sign: FS_SIGN });
                entry.stats = parseStats(statsRaw);

                // 2. ODDS (Robust DOM Scrape)
                console.log(`  Processing Odds for ${id}...`);
                await page.goto(`https://www.flashscore.com/match/${id}/#/match-summary`, { waitUntil: 'networkidle' });
                
                // Click the Over/Under tab explicitly
                await page.evaluate(() => {
                    const tabs = Array.from(document.querySelectorAll('a, button, .tabs__tab, .wcl-tab_XJG99'));
                    const ouTab = tabs.find(t => t.innerText.includes('Over/Under'));
                    if (ouTab) ouTab.click();
                });

                await page.waitForTimeout(5000);
                
                entry.odds = await page.evaluate(() => {
                    const results = {};
                    // Find all odds values on the page using wcl class audited earlier
                    const allElements = Array.from(document.querySelectorAll('[class*="oddsValue"]'));
                    const allValues = allElements.map(el => el.innerText.trim());
                    
                    console.log('Detected values:', allValues.join(', '));

                    for (let i = 0; i < allValues.length - 2; i++) {
                        const v1 = allValues[i];
                        const v2 = allValues[i+1];
                        const v3 = allValues[i+2];
                        
                        if (v1.match(/^\d\.\d$/) && v2.match(/^\d\.\d+$/) && v3.match(/^\d\.\d+$/)) {
                            if (!results[v1]) {
                                results[v1] = { over: v2, under: v3 };
                            }
                        }
                    }
                    return results;
                });

                fs.writeFileSync(path.join(outputDir, `${id}.json`), JSON.stringify(entry, null, 4));
                const o15 = entry.odds['1.5'] ? entry.odds['1.5'].over : 'MISSING';
                console.log(`✅ ${id}: O1.5=${o15}`);

            } catch (e) {
                console.error(`❌ Error on ${id}: ${e.message}`);
            }
        },
        maxConcurrency: 1,
    });

    const requests = testMatches.map(m => ({
        url: `https://www.flashscore.com/match/${m.id}/#/match-summary`,
        userData: { id: m.id }
    }));

    await crawler.run(requests);
}

main();
