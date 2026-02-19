const { PlaywrightCrawler } = require("crawlee");
const path = require("path");

const { getEnabledLeagues } = require("./lib/league_registry");
const {
  applyWindow,
  buildDiscoveryWindow,
  mergeAndSortRecords,
  normalizeRows,
  writeCanonicalOutput,
  writeLegacyOutput,
  writeV1IdsOutput,
} = require("./lib/discovery_utils");

function parseArgs(argv) {
  const args = { mode: "incremental", leagueCode: null };
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--mode") {
      args.mode = argv[i + 1] || args.mode;
      i += 1;
      continue;
    }
    if (token === "--league") {
      args.leagueCode = argv[i + 1] || null;
      i += 1;
      continue;
    }
    if (!token.startsWith("-") && !args.leagueCode) {
      args.leagueCode = token;
    }
  }
  if (!["seed", "incremental"].includes(args.mode)) {
    throw new Error(`Unsupported mode: ${args.mode}. Use seed or incremental.`);
  }
  return args;
}

function getMaxClicks(mode, source) {
  if (source === "results") {
    return mode === "seed" ? 120 : 10;
  }
  return mode === "seed" ? 80 : 12;
}

async function expandRows(page, maxClicks) {
  let clicks = 0;
  while (clicks < maxClicks) {
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(1500);

    const beforeCount = await page
      .locator('div[id^="g_1_"]')
      .count()
      .catch(() => 0);

    const primary = page.locator('a:has-text("Show more matches")').first();
    const fallback = page.locator("a.event__more, .event__more").first();

    let target = null;
    if (await primary.isVisible().catch(() => false)) {
      target = primary;
    } else if (await fallback.isVisible().catch(() => false)) {
      target = fallback;
    }

    if (!target) {
      break;
    }

    await target.scrollIntoViewIfNeeded().catch(() => null);
    await target.click().catch(() => null);
    await page.waitForTimeout(2500);
    clicks += 1;

    const afterCount = await page
      .locator('div[id^="g_1_"]')
      .count()
      .catch(() => 0);
    if (afterCount <= beforeCount) {
      break;
    }
  }
}

async function extractRows(page) {
  return page.evaluate(() => {
    const out = [];

    const pickTeamName = (root) => {
      if (!root) return null;
      const span = root.querySelector('[data-testid^="wcl-scores-simple-text"]');
      if (span && span.textContent) {
        return span.textContent.trim();
      }
      return (root.innerText || root.textContent || "").trim();
    };

    document.querySelectorAll('div[id^="g_1_"]').forEach((el) => {
      const id = el.id.split("_").pop();
      const date = el.querySelector(".event__time")?.innerText?.trim();
      const home = pickTeamName(el.querySelector(".event__homeParticipant"));
      const away = pickTeamName(el.querySelector(".event__awayParticipant"));
      if (id && home && away) {
        out.push({ id, date, home, away });
      }
    });
    return out;
  });
}

async function main() {
  const { mode, leagueCode } = parseArgs(process.argv);
  const leagues = getEnabledLeagues(leagueCode);

  if (!leagues.length) {
    throw new Error(leagueCode ? `Unknown or disabled league code: ${leagueCode}` : "No enabled leagues found.");
  }

  const failed = new Set();
  const buckets = new Map(
    leagues.map((league) => [league.league_code, { league, results: [], fixtures: [] }]),
  );

  const crawler = new PlaywrightCrawler({
    browserPoolOptions: { useFingerprints: true },
    navigationTimeoutSecs: 120,
    maxConcurrency: 3,
    maxRequestsPerCrawl: leagues.length * 2,
    async requestHandler({ page, request, log }) {
      const { league, source, mode: requestMode } = request.userData;
      const leagueCodeValue = league.league_code;

      log.info(`Processing ${source} for ${leagueCodeValue} (${requestMode})`);
      await page.waitForSelector('div[id^="g_1_"]', { timeout: 30000 });

      const maxClicks = getMaxClicks(requestMode, source);
      await expandRows(page, maxClicks);
      const rawRows = await extractRows(page);

      const window = buildDiscoveryWindow({ mode: requestMode, source, league });
      const normalized = normalizeRows(rawRows, league, source, { window });
      const windowed = applyWindow(normalized, window);

      const bucket = buckets.get(leagueCodeValue);
      bucket[source] = windowed;

      log.info(`${leagueCodeValue}/${source}: kept ${windowed.length} rows after window filter`);
    },
    async failedRequestHandler({ request, log }) {
      const { league, source } = request.userData;
      const leagueCodeValue = league?.league_code || "unknown";
      failed.add(leagueCodeValue);
      log.error(`Failed ${source} request for ${leagueCodeValue}: ${request.url}`);
    },
  });

  const requests = [];
  for (const league of leagues) {
    requests.push({
      url: `https://www.flashscore.com/football/${league.flashscore_slug}/results/`,
      userData: { league, source: "results", mode },
    });
    requests.push({
      url: `https://www.flashscore.com/football/${league.flashscore_slug}/fixtures/`,
      userData: { league, source: "fixtures", mode },
    });
  }

  await crawler.run(requests);

  const failedLeagues = [];
  for (const { league, results, fixtures } of buckets.values()) {
    const leagueCodeValue = league.league_code;
    if (failed.has(leagueCodeValue)) {
      failedLeagues.push(leagueCodeValue);
      continue;
    }

    const merged = mergeAndSortRecords(results, fixtures);
    const canonicalPath = writeCanonicalOutput(leagueCodeValue, merged);
    const v1MatchIdsPath = writeV1IdsOutput(leagueCodeValue, "match_ids", merged);
    const v1UpcomingIdsPath = writeV1IdsOutput(leagueCodeValue, "upcoming_ids", fixtures);
    const matchIdsPath = writeLegacyOutput(leagueCodeValue, "match_ids", merged);
    const upcomingIdsPath = writeLegacyOutput(leagueCodeValue, "upcoming_ids", fixtures);

    console.log(
      `${leagueCodeValue}: results=${results.length}, fixtures=${fixtures.length}, merged=${merged.length} -> ${path.relative(process.cwd(), canonicalPath)}; ${path.relative(process.cwd(), v1MatchIdsPath)}; ${path.relative(process.cwd(), v1UpcomingIdsPath)}; ${path.relative(process.cwd(), matchIdsPath)}; ${path.relative(process.cwd(), upcomingIdsPath)}`,
    );
  }

  if (failedLeagues.length) {
    throw new Error(`Discovery failed for league(s): ${failedLeagues.sort().join(", ")}`);
  }
}

main().catch((error) => {
  console.error(`discover_upcoming_ids failed: ${error.message}`);
  process.exit(1);
});
