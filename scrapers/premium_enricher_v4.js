const { PlaywrightCrawler } = require('crawlee');
const fs = require('fs');
const path = require('path');

/**
 * Premium Flashscore Enricher - v4 (Playwright Production Version)
 * Uses the technical blueprint found in the autopsy.
 */

const FS_SIGN = 'SW9D1eZo';
const DELIM_KV = String.fromCharCode(0xf7); 
const DELIM_ROW = String.fromCharCode(0xac); 

function parseStats(text) {
    const stats = { home: {}, away: {} };
    if (!text) return stats;
    const blocks = text.split('~SD' + DELIM_KV);
    blocks.forEach(block => {
        const rows = block.split(DELIM_ROW);
        let name, h, a;
        rows.forEach(row => {
            if (row.startsWith('SG' + DELIM_KV)) name = row.split(DELIM_KV)[1];
            if (row.startsWith('SH' + DELIM_KV)) h = row.split(DELIM_KV)[1];
            if (row.startsWith('SI' + DELIM_KV)) a = row.split(DELIM_KV)[1];
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
    // Autopsy found pattern: [Line] \n [Over] \n [Under]
    const tokens = text.split(/\s+/).filter(t => t.length > 0 && t !== '-');
    
    for (let i = 0; i < tokens.length - 2; i++) {
        const t1 = tokens[i];   
        const t2 = tokens[i+1]; 
        const t3 = tokens[i+2]; 
        
        // Match lines: 1.5, 8.5, -1.5, +0.75
        const isLine = t1.match(/^[+-]?\d(\.[0257]5?)?$/);
        const isOdds = t2.match(/^\d\.\d+$/) && t3.match(/^\d\.\d+$/);
        
        if (isLine && isOdds) {
            if (!results[t1]) results[t1] = { over: t2, under: t3 };
        }
    }
    return results;
}

function normalizeResultStatus(statusText) {
    if (!statusText) return null;
    const normalized = String(statusText).trim().toLowerCase();
    if (!normalized) return null;
    if (normalized.includes('finished') || normalized === 'ft') return 'ft';
    if (normalized.includes('live') || normalized.includes('in progress') || normalized.includes('1st half') || normalized.includes('2nd half') || normalized.includes('halftime')) return 'live';
    if (normalized.includes('scheduled') || normalized.includes('not started') || normalized.includes('today') || normalized.includes('tomorrow')) return 'scheduled';
    if (normalized.includes('postponed')) return 'postponed';
    if (normalized.includes('cancelled') || normalized.includes('canceled')) return 'cancelled';
    if (normalized.includes('abandoned')) return 'abandoned';
    return null;
}

function parseFinalScore(scoreText) {
    if (!scoreText) return null;
    const match = String(scoreText).match(/(\d+)\s*[-:]\s*(\d+)/);
    if (!match) return null;
    return {
        home_goals: Number.parseInt(match[1], 10),
        away_goals: Number.parseInt(match[2], 10),
    };
}

async function extractResult(page) {
    const rawStatus = await page.locator('.detailScore__status').first().textContent().catch(() => null);
    const scoreText = await page.locator('.detailScore__wrapper').first().textContent().catch(() => null);

    const status = normalizeResultStatus(rawStatus && rawStatus.trim());
    const parsedScore = parseFinalScore(scoreText);
    if (!status && !parsedScore) return null;

    const result = {};
    if (status) {
        result.fixture_status = status;
        if (status === 'ft') {
            result.result_status = 'ft';
        }
    }
    if (parsedScore) {
        result.home_goals = parsedScore.home_goals;
        result.away_goals = parsedScore.away_goals;
    }
    result.scraped_at_utc = new Date().toISOString();
    return result;
}

function parseArgs(argv) {
    const args = {
        leagueCode: 'E0',
        idsRoot: path.join(__dirname, '..', 'data', 'v1', 'ids'),
        outRoot: path.join(__dirname, '..', 'data', 'v1', 'premium'),
    };

    for (let i = 2; i < argv.length; i += 1) {
        const token = argv[i];
        if (token === '--ids-root') {
            args.idsRoot = argv[i + 1] || args.idsRoot;
            i += 1;
            continue;
        }
        if (token === '--out-root') {
            args.outRoot = argv[i + 1] || args.outRoot;
            i += 1;
            continue;
        }
        if (!token.startsWith('-') && args.leagueCode === 'E0') {
            args.leagueCode = token;
        }
    }

    return args;
}

async function main() {
    const { leagueCode, idsRoot, outRoot } = parseArgs(process.argv);
    const idPath = path.join(idsRoot, `match_ids_${leagueCode}.json`);
    
    if (!fs.existsSync(idPath)) {
        console.error(`ID file not found: ${idPath}`);
        return;
    }

    const matchesRaw = JSON.parse(fs.readFileSync(idPath, 'utf-8'));
    const matches = Array.isArray(matchesRaw) ? matchesRaw : [];
    const outputDir = path.join(outRoot, leagueCode);
    fs.mkdirSync(outputDir, { recursive: true });

    // Filter out already processed matches for idempotency
    const normalizedMatches = matches
        .map(m => {
            const id = String((m && (m.id || m.flashscore_id)) || '').trim();
            if (!id) return null;
            return { id };
        })
        .filter(Boolean);
    const remainingMatches = normalizedMatches.filter(m => !fs.existsSync(path.join(outputDir, `${m.id}.json`)));
    
    if (remainingMatches.length === 0) {
        console.log(`ALL matches for ${leagueCode} already enriched in ${path.relative(process.cwd(), outputDir)}.`);
        return;
    }

    console.log(`Starting Playwright Enrichment (v4) for ${remainingMatches.length} matches in ${leagueCode}...`);

    const crawler = new PlaywrightCrawler({
        requestHandlerTimeoutSecs: 120,
        maxConcurrency: 5, 
        async requestHandler({ page, request }) {
            const { id } = request.userData;
            const entry = { id, stats: {}, odds: { ou: {}, ah: {} } };

            try {
                // 1. Handshake & Stats
                await page.goto(`https://www.flashscore.com/match/${id}/`, { waitUntil: 'networkidle', timeout: 60000 });
                const result = await extractResult(page);
                if (result) {
                    entry.result = result;
                }
                const baseUrl = page.url().split('?')[0];
                
                const statsRaw = await page.evaluate(async ({ mid, sign }) => {
                    const resp = await fetch(`https://global.flashscore.ninja/2/x/feed/df_st_1_${mid}`, { headers: { 'x-fsign': sign } });
                    return resp.text();
                }, { mid: id, sign: FS_SIGN });
                entry.stats = parseStats(statsRaw);

                // 2. Over/Under Odds
                await page.goto(`${baseUrl}odds/over-under/full-time/?mid=${id}`, { waitUntil: 'networkidle', timeout: 60000 });
                await page.waitForTimeout(4000);
                entry.odds.ou = extractOdds(await page.innerText('body'));

                // 3. Asian Handicap Odds
                await page.goto(`${baseUrl}odds/asian-handicap/full-time/?mid=${id}`, { waitUntil: 'networkidle', timeout: 60000 });
                await page.waitForTimeout(4000);
                entry.odds.ah = extractOdds(await page.innerText('body'));

                fs.writeFileSync(path.join(outputDir, `${id}.json`), JSON.stringify(entry, null, 4));
                console.log(`Success: ${id} | Stats: ${Object.keys(entry.stats.home || {}).length} | OU: ${Object.keys(entry.odds.ou).length} | AH: ${Object.keys(entry.odds.ah).length}`);

            } catch (e) {
                console.error(`Failed ${id}: ${e.message}`);
            }
            // Close the page to free up resources
            await page.close();
            // Be nice to the servers
            await new Promise(r => setTimeout(r, 1000));
        },
    });

    await crawler.run(remainingMatches.map(m => ({
        url: `https://www.flashscore.com/match/${m.id}/`,
        userData: { id: m.id }
    })));
}

main();
