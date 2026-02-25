const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function scrapeMatchDetails(matchId) {
    console.log(`Scraping full details for Match: ${matchId}`);
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    const baseUrl = `https://www.flashscore.com/match/${matchId}/#/`;
    const data = { id: matchId, stats: {}, odds: {} };

    try {
        // 1. Scrape Statistics
        await page.goto(`${baseUrl}match-statistics/0`, { waitUntil: 'networkidle' });
        await page.waitForTimeout(2000);
        
        data.stats = await page.evaluate(() => {
            const result = {};
            const rows = document.querySelectorAll('.stat__row');
            rows.forEach(row => {
                const name = row.querySelector('.stat__categoryName')?.innerText;
                const home = row.querySelector('.stat__homeValue')?.innerText;
                const away = row.querySelector('.stat__awayValue')?.innerText;
                if (name) result[name] = { home, away };
            });
            return result;
        });

        // 2. Scrape Over/Under Odds
        await page.goto(`${baseUrl}odds-comparison/over-under/0`, { waitUntil: 'networkidle' });
        await page.waitForTimeout(2000);
        
        data.odds = await page.evaluate(() => {
            const ou = {};
            const rows = document.querySelectorAll('.ui-table__row');
            rows.forEach(row => {
                const line = row.querySelector('.oddsCell__cell--fixed')?.innerText;
                const over = row.querySelectorAll('.oddsCell__cell')[1]?.innerText;
                if (line && line.includes('1.5')) ou['o15'] = over;
                if (line && line.includes('2.5')) ou['o25'] = over;
            });
            return ou;
        });

    } catch (e) {
        console.error(`Failed to scrape ${matchId}: ${e.message}`);
    }

    await browser.close();
    return data;
}

async function main() {
    const matchId = process.argv[2] || 'lKNJm8ak';
    const result = await scrapeMatchDetails(matchId);
    
    const outputPath = path.join(__dirname, '..', 'data', 'scraper', 'premium', `${matchId}.json`);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, JSON.stringify(result, null, 4));
    console.log(`Success! Saved to ${outputPath}`);
    console.log(JSON.stringify(result, null, 2));
}

main();
