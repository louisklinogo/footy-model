const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const FS_SIGN = 'SW9D1eZo';

function parseStats(text) {
    const stats = { home: {}, away: {} };
    if (!text) return stats;
    const regex = /SG\xf7(.*?)\xac.*?SH\xf7(.*?)(?=SI\xf7)SI\xf7(.*?)\xac/g;
    let m;
    while ((m = regex.exec(text)) !== null) {
        const name = m[1].replace(/\xac/g, '').trim();
        const home = m[2].replace(/\xac/g, '').trim();
        const away = m[3].replace(/\xac/g, '').trim();
        if (!stats.home[name]) { stats.home[name] = home; stats.away[name] = away; }
    }
    return stats;
}

function extractOdds(text, type) {
    const results = {};
    if (!text) return results;
    const tokens = text.split(/\s+/).filter(t => t.length > 0 && t !== '-');
    
    for (let i = 0; i < tokens.length - 2; i++) {
        const t1 = tokens[i];   
        const t2 = tokens[i+1]; 
        const t3 = tokens[i+2]; 
        
        // Match lines like 1.5, 8.5, -1.5, +0.75
        const isLine = t1.match(/^[+-]?\d(\.[0257]5?)?$/);
        const isOdds = t2.match(/^\d\.\d+$/) && t3.match(/^\d\.\d+$/);
        
        if (isLine && isOdds) {
            if (!results[t1]) results[t1] = { val1: t2, val2: t3 };
        }
    }
    return results;
}

async function main() {
    const leagueCode = process.argv[2] || 'E0';
    const idPath = path.join(__dirname, 'data', `match_ids_${leagueCode}.json`);
    const matches = JSON.parse(fs.readFileSync(idPath));
    const outputDir = path.join(__dirname, 'data', 'premium', leagueCode);
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    const browser = await chromium.launch({ headless: true });
    
    // Filter out already processed matches
    const remainingMatches = matches.filter(m => !fs.existsSync(path.join(outputDir, `${m.id}.json`)));
    
    if (remainingMatches.length === 0) {
        console.log(`✅ All matches for ${leagueCode} already enriched.`);
        await browser.close();
        return;
    }

    console.log(`🚀 Processing ${remainingMatches.length} remaining matches...`);
    
    for (const match of remainingMatches) {
        console.log(`📡 Processing: ${match.id}`);
        const page = await browser.newPage();
        const entry = { id: match.id, stats: {}, odds: { ou: {}, ah: {} } };

        try {
            // 1. STATS (Direct Feed Handshake)
            await page.goto(`https://www.flashscore.com/match/${match.id}/`, { waitUntil: 'networkidle', timeout: 60000 });
            const baseUrl = page.url().split('?')[0]; // Full slug URL
            
            const statsRaw = await page.evaluate(async ({ mid, sign }) => {
                const resp = await fetch(`https://global.flashscore.ninja/2/x/feed/df_st_1_${mid}`, { headers: { 'x-fsign': sign } });
                return resp.text();
            }, { mid: match.id, sign: FS_SIGN });
            entry.stats = parseStats(statsRaw);

            // 2. OU ODDS
            await page.goto(`${baseUrl}odds/over-under/full-time/?mid=${match.id}`, { waitUntil: 'networkidle', timeout: 60000 });
            await page.waitForTimeout(4000);
            entry.odds.ou = extractOdds(await page.innerText('body'), 'ou');

            // 3. AH ODDS
            await page.goto(`${baseUrl}odds/asian-handicap/full-time/?mid=${match.id}`, { waitUntil: 'networkidle', timeout: 60000 });
            await page.waitForTimeout(4000);
            entry.odds.ah = extractOdds(await page.innerText('body'), 'ah');

            fs.writeFileSync(path.join(outputDir, `${match.id}.json`), JSON.stringify(entry, null, 4));
            console.log(`✅ Success: ${match.id} | Stats: ${Object.keys(entry.stats.home || {}).length} | OU: ${Object.keys(entry.odds.ou).length} | AH: ${Object.keys(entry.odds.ah).length}`);

        } catch (e) {
            console.error(`❌ Failed ${match.id}: ${e.message}`);
        }
        await page.close();
        // Delay to be nice to servers
        await new Promise(r => setTimeout(r, 2000));
    }
    await browser.close();
}

main();
