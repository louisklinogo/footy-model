const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const REGISTRY_PATH = path.join(__dirname, "config", "league_registry.json");
const LEGACY_DATA_DIR = path.join(__dirname, "..", "data", "scraper");
const V1_DISCOVERY_DIR = path.join(__dirname, "..", "data", "v1", "discovery");
const V1_REPORTS_DIR = path.join(__dirname, "..", "data", "v1", "reports");

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function readRegistry() {
  const raw = fs.readFileSync(REGISTRY_PATH, "utf-8");
  const leagues = JSON.parse(raw);
  if (!Array.isArray(leagues)) {
    throw new Error("league_registry.json must be a JSON array");
  }
  return leagues;
}

function parseArgs(argv) {
  const args = { mode: "seed", leagues: null };
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--mode") {
      args.mode = argv[i + 1] || args.mode;
      i += 1;
      continue;
    }
    if (token === "--leagues") {
      args.leagues = (argv[i + 1] || "").split(",").map((s) => s.trim()).filter(Boolean);
      i += 1;
      continue;
    }
  }
  return args;
}

function summarizeDiscovery(leagueCode) {
  const v1Path = path.join(V1_DISCOVERY_DIR, `discovery_fixtures_${leagueCode}.json`);
  const legacyPath = path.join(LEGACY_DATA_DIR, `discovery_fixtures_${leagueCode}.json`);
  const p = fs.existsSync(v1Path) ? v1Path : legacyPath;
  if (!fs.existsSync(p)) {
    return { league: leagueCode, ok: false, error: "missing discovery output" };
  }
  const rows = JSON.parse(fs.readFileSync(p, "utf-8"));
  const kickoffs = rows
    .map((r) => r && r.kickoff_datetime_utc)
    .filter(Boolean)
    .sort();
  return {
    league: leagueCode,
    ok: true,
    total: rows.length,
    min_kickoff: kickoffs[0] || null,
    max_kickoff: kickoffs[kickoffs.length - 1] || null,
  };
}

function main() {
  const { mode, leagues: onlyLeagues } = parseArgs(process.argv);
  if (!['seed', 'incremental'].includes(mode)) {
    throw new Error(`Unsupported mode: ${mode}`);
  }

  const enabled = readRegistry().filter((l) => l.enabled);
  const selected = onlyLeagues
    ? enabled.filter((l) => onlyLeagues.includes(l.league_code))
    : enabled;

  if (!selected.length) {
    throw new Error("No enabled leagues selected");
  }

  const summary = [];
  for (const league of selected) {
    const leagueCode = league.league_code;
    const target = path.join(__dirname, "discover_upcoming_ids.js");
    const result = spawnSync(process.execPath, [target, "--mode", mode, "--league", leagueCode], {
      stdio: "inherit",
    });
    if (result.status !== 0) {
      summary.push({ league: leagueCode, ok: false, error: `discover exited ${result.status}` });
      continue;
    }
    summary.push(summarizeDiscovery(leagueCode));
  }

  ensureDir(V1_REPORTS_DIR);
  const outPath = path.join(V1_REPORTS_DIR, `seed_summary_${mode}.json`);
  fs.writeFileSync(outPath, JSON.stringify(summary, null, 2));
  console.log(`Wrote ${path.relative(process.cwd(), outPath)}`);

  const failed = summary.filter((s) => !s.ok);
  if (failed.length) {
    const codes = failed.map((f) => f.league).join(", ");
    throw new Error(`Seed failed for league(s): ${codes}`);
  }
}

main();
