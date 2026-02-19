const https = require("https");

const { getEnabledLeagues } = require("./lib/league_registry");

const FLASHSCORE_HOST = "www.flashscore.com";
const REQUEST_TIMEOUT_MS = 20000;
const MAX_REDIRECTS = 10;

function requestPath(pathname, redirectCount = 0) {
  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        host: FLASHSCORE_HOST,
        path: pathname,
        method: "GET",
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
          Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
      },
      (res) => {
        const statusCode = res.statusCode || 0;
        const location = res.headers.location;
        res.resume();

        if (statusCode >= 300 && statusCode < 400 && location) {
          if (redirectCount >= MAX_REDIRECTS) {
            reject(new Error(`Too many redirects for ${pathname}`));
            return;
          }
          const nextPath = location.startsWith("http")
            ? new URL(location).pathname
            : location;
          resolve(requestPath(nextPath, redirectCount + 1));
          return;
        }

        resolve({
          statusCode,
          finalPath: pathname,
          redirected: redirectCount > 0,
        });
      },
    );

    req.setTimeout(REQUEST_TIMEOUT_MS, () => {
      req.destroy(new Error(`Timeout for ${pathname}`));
    });
    req.on("error", reject);
    req.end();
  });
}

async function checkLeague(league) {
  const checks = [
    { source: "results", path: `/football/${league.flashscore_slug}/results/` },
    { source: "fixtures", path: `/football/${league.flashscore_slug}/fixtures/` },
  ];

  for (const check of checks) {
    const response = await requestPath(check.path);
    if (response.statusCode >= 400) {
      throw new Error(`${league.league_code} ${check.source} returned HTTP ${response.statusCode}`);
    }
    if (!response.finalPath.includes(`/football/${league.flashscore_slug}/`)) {
      throw new Error(
        `${league.league_code} ${check.source} canonical mismatch: got ${response.finalPath}`,
      );
    }
  }
}

async function main() {
  const leagues = getEnabledLeagues();
  if (!leagues.length) {
    throw new Error("No enabled leagues found in registry.");
  }

  const failures = [];
  for (const league of leagues) {
    try {
      await checkLeague(league);
      console.log(`OK  ${league.league_code} ${league.flashscore_slug}`);
    } catch (error) {
      failures.push({ league: league.league_code, error: error.message });
      console.error(`FAIL ${league.league_code} ${error.message}`);
    }
  }

  if (failures.length) {
    const codes = failures.map((item) => item.league).join(", ");
    throw new Error(`League slug smoke check failed for: ${codes}`);
  }
}

main().catch((error) => {
  console.error(`check_league_slugs failed: ${error.message}`);
  process.exit(1);
});
