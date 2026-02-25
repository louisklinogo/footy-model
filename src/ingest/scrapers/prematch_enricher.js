const { PlaywrightCrawler } = require('crawlee');
const fs = require('fs');
const path = require('path');

/**
 * Prematch Enricher - Captures odds, stats, AND injuries/lineups
 * Renamed from safe_prematch_enricher.js (2026-02-22)
 * Added: Injuries, suspensions, questionable players, predicted lineups
 * 
 * IMPORTANT: Odds extraction logic is UNCHANGED - do not modify
 */

const FS_SIGN = 'SW9D1eZo';

// --- INHERITED LOGIC: STATS PARSING ---

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

// --- INHERITED LOGIC: ODDS EXTRACTION (DO NOT MODIFY) ---

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
    const results = {};
    if (!text) return results;
    const tokens = text.split(/\s+/).filter(t => t.length > 0 && t !== '-');
    for (let i = 0; i < tokens.length - 2; i++) {
        const t1 = tokens[i];
        const t2 = tokens[i + 1];
        const t3 = tokens[i + 2];
        const isOddsValue = t => t.match(/^\d+\.\d+$/);
        if (isOddsValue(t1) && isOddsValue(t2) && isOddsValue(t3)) {
            return { "1": t1, "X": t2, "2": t3 };
        }
    }
    return results;
}

// --- NEW LOGIC: INJURIES & LINEUPS (DYNAMIC - WORKS FOR ANY TEAM) ---

const INJURY_REASONS = [
    'Knee Injury', 'Groin Injury', 'Hamstring Injury', 'Ankle Injury',
    'Shoulder Injury', 'Back Injury', 'Broken Leg', 'Suspended',
    'Calf Injury', 'Thigh Injury', 'Illness', 'International Duty', 'Injury'
];

function extractReasonFromText(text) {
    for (const reason of INJURY_REASONS) {
        if (text.includes(reason)) return reason;
    }
    // Check for partial matches
    if (text.includes('Hamstring')) return 'Hamstring Injury';
    if (text.includes('Calf')) return 'Calf Injury';
    if (text.includes('Thigh')) return 'Thigh Injury';
    if (text.includes('Ankle')) return 'Ankle Injury';
    if (text.includes('Back')) return 'Back Injury';
    if (text.includes('Shoulder')) return 'Shoulder Injury';
    if (text.includes('Broken')) return 'Broken Leg';
    return 'Unknown';
}

async function extractInjuriesAndLineups(page) {
    const result = {
        home: { missing: [], questionable: [], lineup: [] },
        away: { missing: [], questionable: [], lineup: [] }
    };

    try {
        // Go to lineups page - clean up any odds paths from URL
        const currentUrl = page.url();
        // Extract base match URL (remove /odds/ paths)
        let baseUrl = currentUrl.split('#')[0].split('?')[0].replace(/\/odds\/.*/, '');
        // Ensure baseUrl ends with /
        if (!baseUrl.endsWith('/')) baseUrl += '/';
        const lineupsUrl = baseUrl + 'summary/lineups/';
        console.log(`    → Navigating to lineups: ${lineupsUrl}`);
        await page.goto(lineupsUrl, { waitUntil: 'domcontentloaded' });
        await page.waitForTimeout(3000);

        // Extract injuries using dynamic team detection
        // The page structure has players grouped by team in order (home first, away second)
        const data = await page.evaluate(() => {
            const output = { home: { missing: [], questionable: [], lineup: [] }, away: { missing: [], questionable: [], lineup: [] } };
            
            const INJURY_REASONS = [
                'Knee Injury', 'Groin Injury', 'Hamstring Injury', 'Ankle Injury',
                'Shoulder Injury', 'Back Injury', 'Broken Leg', 'Suspended',
                'Calf Injury', 'Thigh Injury', 'Illness', 'International Duty', 'Injury'
            ];
            
            const extractReason = (text) => {
                for (const reason of INJURY_REASONS) {
                    if (text.includes(reason)) return reason;
                }
                if (text.includes('Hamstring')) return 'Hamstring Injury';
                if (text.includes('Calf')) return 'Calf Injury';
                if (text.includes('Thigh')) return 'Thigh Injury';
                if (text.includes('Ankle')) return 'Ankle Injury';
                if (text.includes('Back')) return 'Back Injury';
                if (text.includes('Shoulder')) return 'Shoulder Injury';
                if (text.includes('Broken')) return 'Broken Leg';
                return 'Unknown';
            };
            
            // Find "Will not play" header
            const allElements = Array.from(document.querySelectorAll('*'));
            const willNotPlayHeader = allElements.find(el => el.textContent?.trim() === 'Will not play');
            
            if (willNotPlayHeader) {
                // Structure: .section > .wcl-headerSection (header) + .lf__sidesBox > .lf__sides > .lf__side (teams)
                // Find the section container
                const section = willNotPlayHeader.closest('.section');
                
                if (section) {
                    // Find the sidesBox within this section
                    const sidesBox = section.querySelector('.lf__sidesBox');
                    
                    if (sidesBox) {
                        // Get the lf__sides container
                        const sides = sidesBox.querySelector('.lf__sides');
                        
                        if (sides) {
                            // Each .lf__side contains one team's players
                            const teamSides = sides.querySelectorAll(':scope > .lf__side');
                            
                            teamSides.forEach((side, sideIdx) => {
                                const team = sideIdx === 0 ? 'home' : 'away';
                                
                                // Get all direct children (player entries)
                                const entries = side.querySelectorAll(':scope > *');
                                
                                entries.forEach(entry => {
                                    const link = entry.querySelector('a[href*="/player/"]');
                                    if (!link) return;
                                    
                                    const name = link.textContent?.trim();
                                    if (!name || name.length < 2 || name.length > 30) return;
                                    
                                    const entryText = entry.textContent || '';
                                    const reason = extractReason(entryText);
                                    
                                    if (!output[team].missing.find(p => p.name === name)) {
                                        output[team].missing.push({ name, reason });
                                    }
                                });
                            });
                        }
                    }
                }
            }
            
            // Same for "Questionable"
            const questionableHeader = allElements.find(el => el.textContent?.trim() === 'Questionable');
            
            if (questionableHeader) {
                const section = questionableHeader.closest('.section');
                
                if (section) {
                    const sidesBox = section.querySelector('.lf__sidesBox');
                    
                    if (sidesBox) {
                        const sides = sidesBox.querySelector('.lf__sides');
                        
                        if (sides) {
                            const teamSides = sides.querySelectorAll(':scope > .lf__side');
                            
                            teamSides.forEach((side, sideIdx) => {
                                const team = sideIdx === 0 ? 'home' : 'away';
                                
                                const entries = side.querySelectorAll(':scope > *');
                                
                                entries.forEach(entry => {
                                    const link = entry.querySelector('a[href*="/player/"]');
                                    if (!link) return;
                                    
                                    const name = link.textContent?.trim();
                                    if (!name || name.length < 2 || name.length > 30) return;
                                    
                                    const entryText = entry.textContent || '';
                                    const reason = extractReason(entryText);
                                    
                                    if (!output[team].questionable.find(p => p.name === name)) {
                                        output[team].questionable.push({ name, reason });
                                    }
                                });
                            });
                        }
                    }
                }
            }
            
            return output;
        });

        // Assign extracted data
        Object.assign(result.home, data.home);
        Object.assign(result.away, data.away);
        
        console.log(`    → Found: ${result.home.missing.length} home injuries, ${result.away.missing.length} away injuries`);

    } catch (e) {
        console.log(`  ⚠️ Injury/lineup scrape error: ${e.message}`);
    }

    return result;
}

// --- ORCHESTRATION ---

async function main() {
    const args = process.argv.slice(2);
    const leagueCode = args[0] || 'E0';
    const idListJson = args[1];

    if (!idListJson) {
        console.error("Missing ID list.");
        process.exit(1);
    }

    const ids = JSON.parse(idListJson);
    const outDir = path.join(__dirname, '..', 'data', 'v1', 'prematch_json', leagueCode);
    fs.mkdirSync(outDir, { recursive: true });

    console.log(`🚀 Prematch Enricher working on ${ids.length} matches in ${leagueCode}...`);
    console.log(`   Includes: Odds + Stats + Injuries/Lineups`);

    const crawler = new PlaywrightCrawler({
        maxConcurrency: 3,
        async requestHandler({ page, request }) {
            const { id } = request.userData;
            
            // Initialize entry with all sections
            const entry = { 
                id, 
                stats: {}, 
                odds: { ou: {}, ah: {}, "1x2": {} }, 
                availability: { home: {}, away: {} },  // NEW
                scraped_at: new Date().toISOString() 
            };

            try {
                // 1. Visit Main summary for stats/metadata
                await page.goto(`https://www.flashscore.com/match/${id}/#/match-summary`, { waitUntil: 'domcontentloaded' });
                // Extract base match URL BEFORE navigating to other pages
                const baseUrl = page.url().split('#')[0].split('?')[0].replace(/\/odds\/.*/, '').replace(/\/summary\/.*/, '').replace(/\/lineups\/.*/, '');

                // Fetch stats (Using same handshake logic)
                const statsRaw = await page.evaluate(async (mid) => {
                    const sign = 'SW9D1eZo';
                    const resp = await fetch(`https://global.flashscore.ninja/2/x/feed/df_st_1_${mid}`, { headers: { 'x-fsign': sign } });
                    return resp.text();
                }, id);
                entry.stats = parseStats(statsRaw);

                // 2. Over/Under (UNCHANGED)
                await page.goto(`${baseUrl}odds/over-under/full-time/`, { waitUntil: 'domcontentloaded' });
                await page.waitForTimeout(3000);
                entry.odds.ou = extractOdds(await page.innerText('body'));

                // 3. Asian Handicap (UNCHANGED)
                await page.goto(`${baseUrl}odds/asian-handicap/full-time/`, { waitUntil: 'domcontentloaded' });
                await page.waitForTimeout(3000);
                entry.odds.ah = extractOdds(await page.innerText('body'));

                // 4. 1X2 (UNCHANGED)
                await page.goto(`${baseUrl}odds/1x2-odds/full-time/`, { waitUntil: 'domcontentloaded' });
                await page.waitForTimeout(3000);
                entry.odds["1x2"] = extract1X2(await page.innerText('body'));

                // 5. Injuries & Lineups (NEW!)
                console.log(`  📋 Fetching injuries/lineups for ${id}...`);
                entry.availability = await extractInjuriesAndLineups(page);

                const missingCount = entry.availability.home.missing.length + entry.availability.away.missing.length;
                console.log(`  ✅ ${id}: Odds + ${missingCount} injuries/missing players`);

                fs.writeFileSync(path.join(outDir, `${id}.json`), JSON.stringify(entry, null, 4));

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
