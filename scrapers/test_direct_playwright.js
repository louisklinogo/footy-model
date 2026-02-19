const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function scrapeOdds(matchId) {
    console.log(`🔍 Scoping Match: ${matchId}`);
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    try {
        await page.goto(`https://www.flashscore.com/match/${matchId}/#/match-summary`, { waitUntil: 'networkidle' });
        
        // Take a screenshot to debug
        const screenshotPath = path.join(__dirname, '..', 'data', 'scraper', 'premium', `debug_${matchId}.png`);
        await page.screenshot({ path: screenshotPath });
        console.log(`  📸 Screenshot saved: ${screenshotPath}`);

        // Try clicking Odds then Over/Under
        try {
            await page.click('text="Odds"', { timeout: 5000 });
            console.log('  Clicked Odds');
            await page.click('text="Over/Under"', { timeout: 5000 });
            console.log('  Clicked Over/Under');
        } catch (e) {
            console.log('  ⚠️ Click failed, trying direct navigation...');
            await page.goto(`https://www.flashscore.com/match/${matchId}/#/odds-comparison/over-under/0`, { waitUntil: 'networkidle' });
        }

        await page.waitForTimeout(5000);

        const odds = await page.evaluate(() => {
            const results = {};
            // Look for any element containing decimal odds
            const allSpans = Array.from(document.querySelectorAll('span, div')).filter(el => el.innerText.trim().match(/^\d\.\d+$/));
            const allValues = allSpans.map(el => el.innerText.trim());
            
            // Look for lines (0.5, 1.5, 2.5)
            const lines = Array.from(document.querySelectorAll('div, span')).filter(el => el.innerText.trim().match(/^\d\.\d$/)).map(el => el.innerText.trim());
            
            return {
                allValues: allValues.slice(0, 20),
                lines: lines.slice(0, 10)
            };
        });

        console.log(`  ✅ Data Sample for ${matchId}:`, JSON.stringify(odds));
        await browser.close();
        return odds;

    } catch (e) {
        console.error(`  ❌ Error for ${matchId}: ${e.message}`);
        await browser.close();
        return null;
    }
}

async function main() {
    const ids = ['lKNJm8ak', 'W2uAbPD7', '4nnmgUyS'];
    for (const id of ids) {
        await scrapeOdds(id);
    }
}

main();
