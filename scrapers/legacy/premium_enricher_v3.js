const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

/**
 * Premium Flashscore Enricher - v3.1 (Hybrid Direct Feed)
 * Stats: Fast Direct Feed Ingestion
 * Odds: Browser-Handshake Feed Ingestion
 */

const FS_SIGN = 'SW9D1eZo';
const DELIM_KV = String.fromCharCode(0xf7); // ÷
const DELIM_ROW = String.fromCharCode(0xac); // ¬

function parseStats(text) {
    const stats = { home: {}, away: {} };
    if (!text) return stats;
    
    // Pattern: SG÷[Name]¬SH÷[Home]¬SI÷[Away]¬
    // We split by ~SD÷ blocks
    const blocks = text.split('~SD' + DELIM_KV);
    
    blocks.forEach(block => {
        const rows = block.split(DELIM_ROW);
        let name = null;
        let h = null;
        let a = null;
        
        rows.forEach(row => {
            if (row.startsWith('SG' + DELIM_KV)) name = row.split(DELIM_KV)[1];
            if (row.startsWith('SH' + DELIM_KV)) h = row.split(DELIM_KV)[1];
            if (row.startsWith('SI' + DELIM_KV)) a = row.split(DELIM_KV)[1];
        });
        
        if (name && h !== null && a !== null) {
            if (!stats.home[name]) {
                stats.home[name] = h;
                stats.away[name] = a;
            }
        }
    });
    return stats;
}

function parseOdds(text) {
    const results = {};
    if (!text || text === '0') return results;
    
    // Format: MI÷[Line]¬...OD÷[Over]¬...OD÷[Under]
    const parts = text.split('MI' + DELIM_KV);
    parts.forEach(part => {
        const lines = part.split(DELIM_ROW);
        const label = lines[0]; 
        const odds = lines.filter(l => l.startsWith('OD' + DELIM_KV)).map(l => l.split(DELIM_KV)[1]);
        if (label && odds.length >= 2) {
            results[label] = { over: odds[0], under: odds[1] };
        }
    });
    return results;
}

async function main() {
    const ids = ['lKNJm8ak', 'W2uAbPD7']; 
    const outputDir = path.join(__dirname, 'data', 'premium', 'v3_test');
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    for (const id of ids) {
        console.log(`🚀 Processing: ${id}`);
        const entry = { id, stats: {}, odds: { ou: {}, ah: {}, corners: {}, btts: {} } };

        try {
            // 1. Establish Handshake
            await page.goto(`https://www.flashscore.com/match/${id}/#/match-summary`, { waitUntil: 'networkidle' });

            // 2. Fetch all feeds using browser's session
            const feeds = await page.evaluate(async ({ mid, sign }) => {
                const fetchFeed = async (url) => {
                    try {
                        const resp = await fetch(url, { headers: { 'x-fsign': sign } });
                        return await resp.text();
                    } catch (e) { return ''; }
                };

                return {
                    stats: await fetchFeed(`https://global.flashscore.ninja/2/x/feed/df_st_1_${mid}`),
                    ou: await fetchFeed(`https://global.flashscore.ninja/2/x/feed/df_od_1_${mid}_2`),
                    ah: await fetchFeed(`https://global.flashscore.ninja/2/x/feed/df_od_1_${mid}_1`),
                    corners: await fetchFeed(`https://global.flashscore.ninja/2/x/feed/df_od_1_${mid}_13`),
                    btts: await fetchFeed(`https://global.flashscore.ninja/2/x/feed/df_od_1_${mid}_5`)
                };
            }, { mid: id, sign: FS_SIGN });

            // 3. Parse
            entry.stats = parseStats(feeds.stats);
            entry.odds.ou = parseOdds(feeds.ou);
            entry.odds.ah = parseOdds(feeds.ah);
            entry.odds.corners = parseOdds(feeds.corners);
            entry.odds.btts = parseOdds(feeds.btts);

            fs.writeFileSync(path.join(outputDir, `${id}.json`), JSON.stringify(entry, null, 4));
            
            console.log(`✅ Success: ${id}`);
            console.log(`   Stats: ${Object.keys(entry.stats.home).length}`);
            console.log(`   OU Lines: ${Object.keys(entry.odds.ou).length}`);
            console.log(`   O1.5: ${entry.odds.ou['1.5'] ? entry.odds.ou['1.5'].over : 'MISSING'}`);

        } catch (e) {
            console.error(`❌ Failed ${id}: ${e.message}`);
        }
    }

    await browser.close();
}

main();
