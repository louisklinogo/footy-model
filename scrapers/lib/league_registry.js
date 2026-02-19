const fs = require("fs");
const path = require("path");

const REGISTRY_PATH = path.join(__dirname, "..", "config", "league_registry.json");

function readLeagueRegistry() {
  const raw = fs.readFileSync(REGISTRY_PATH, "utf-8");
  const leagues = JSON.parse(raw);
  if (!Array.isArray(leagues)) {
    throw new Error("League registry must be a JSON array.");
  }
  return leagues;
}

function getEnabledLeagues(leagueCode) {
  const leagues = readLeagueRegistry().filter((league) => league.enabled);
  if (!leagueCode) {
    return leagues;
  }
  return leagues.filter((league) => league.league_code === leagueCode);
}

module.exports = {
  REGISTRY_PATH,
  readLeagueRegistry,
  getEnabledLeagues,
};
