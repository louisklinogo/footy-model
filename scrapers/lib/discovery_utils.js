const fs = require("fs");
const path = require("path");

const LEGACY_DATA_DIR = path.join(__dirname, "..", "..", "data", "scraper");
const V1_ROOT_DIR = path.join(__dirname, "..", "..", "data", "v1");
const V1_DISCOVERY_DIR = path.join(V1_ROOT_DIR, "discovery");
const V1_IDS_DIR = path.join(V1_ROOT_DIR, "ids");

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function parseKickoffRaw(rawDateText) {
  if (!rawDateText) {
    return null;
  }
  const normalized = String(rawDateText)
    .replace(/\s+/g, " ")
    .trim();
  const match = normalized.match(/(\d{1,2})\.(\d{1,2})\.(?:\s*(\d{4})\.)?\s*(\d{1,2}):(\d{2})/);
  if (!match) {
    return null;
  }
  return {
    day: Number(match[1]),
    month: Number(match[2]),
    year: match[3] ? Number(match[3]) : null,
    hour: Number(match[4]),
    minute: Number(match[5]),
  };
}

function getOffsetMinutesAt(timeZone, date) {
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

  const parts = Object.fromEntries(
    formatter
      .formatToParts(date)
      .filter((part) => part.type !== "literal")
      .map((part) => [part.type, part.value]),
  );

  const utcFromParts = Date.UTC(
    Number(parts.year),
    Number(parts.month) - 1,
    Number(parts.day),
    Number(parts.hour),
    Number(parts.minute),
    Number(parts.second),
  );
  return (utcFromParts - date.getTime()) / 60000;
}

function zonedDateTimeToUtcIso({ year, month, day, hour, minute, timeZone }) {
  const initialUtc = Date.UTC(year, month - 1, day, hour, minute, 0);
  const initialOffset = getOffsetMinutesAt(timeZone, new Date(initialUtc));
  let adjustedUtc = initialUtc - initialOffset * 60 * 1000;
  const adjustedOffset = getOffsetMinutesAt(timeZone, new Date(adjustedUtc));
  if (adjustedOffset !== initialOffset) {
    adjustedUtc = initialUtc - adjustedOffset * 60 * 1000;
  }
  return new Date(adjustedUtc).toISOString();
}

function buildDiscoveryWindow({ mode, source, league, now = new Date() }) {
  const nowDate = now instanceof Date ? now : new Date(now);
  const nowTs = nowDate.getTime();
  const lookbackDays = Number(league.results_lookback_days || 7);
  const lookaheadDays = Number(league.fixtures_lookahead_days || 30);

  if (mode === "incremental") {
    if (source === "results") {
      return {
        start: new Date(nowTs - lookbackDays * 24 * 60 * 60 * 1000),
        end: nowDate,
      };
    }
    return {
      start: nowDate,
      end: new Date(nowTs + lookaheadDays * 24 * 60 * 60 * 1000),
    };
  }

  const seasonStartMonth = Number(league.season_start_month || 8);
  const nowYear = nowDate.getUTCFullYear();
  const nowMonth = nowDate.getUTCMonth() + 1;
  const seasonStartYear = nowMonth >= seasonStartMonth ? nowYear : nowYear - 1;

  if (source === "fixtures") {
    if (league.season_type === "calendar_year") {
      const seasonYear = nowYear;
      return {
        start: nowDate,
        end: new Date(Date.UTC(seasonYear, 11, 31, 23, 59, 59)),
      };
    }

    let endYear = seasonStartYear + 1;
    let endMonth = seasonStartMonth - 1;
    if (endMonth <= 0) {
      endMonth = 12;
      endYear -= 1;
    }
    const end = new Date(Date.UTC(endYear, endMonth, 0, 23, 59, 59));
    return {
      start: nowDate,
      end,
    };
  }

  return {
    start: new Date(Date.UTC(seasonStartYear, seasonStartMonth - 1, 1, 0, 0, 0)),
    end: nowDate,
  };
}

function chooseBestYear(parsed, window, league) {
  if (parsed.year) {
    return parsed.year;
  }

  const timeZone = league.timezone || "UTC";
  const center = Math.round((window.start.getTime() + window.end.getTime()) / 2);
  const startYear = window.start.getUTCFullYear();
  const endYear = window.end.getUTCFullYear();
  const candidates = new Set([startYear - 1, startYear, endYear, endYear + 1]);

  let bestYear = startYear;
  let bestScore = Number.POSITIVE_INFINITY;

  for (const year of candidates) {
    const iso = zonedDateTimeToUtcIso({
      year,
      month: parsed.month,
      day: parsed.day,
      hour: parsed.hour,
      minute: parsed.minute,
      timeZone,
    });
    const ts = Date.parse(iso);

    let score;
    if (ts >= window.start.getTime() && ts <= window.end.getTime()) {
      score = Math.abs(ts - center);
    } else {
      const dist = ts < window.start.getTime() ? window.start.getTime() - ts : ts - window.end.getTime();
      score = 1_000_000_000_000_000 + dist;
    }

    if (score < bestScore) {
      bestScore = score;
      bestYear = year;
    }
  }

  return bestYear;
}

function toKickoffUtc(rawDateText, league, window) {
  const parsed = parseKickoffRaw(rawDateText);
  if (!parsed) {
    return null;
  }
  const year = chooseBestYear(parsed, window, league);

  return zonedDateTimeToUtcIso({
    year,
    month: parsed.month,
    day: parsed.day,
    hour: parsed.hour,
    minute: parsed.minute,
    timeZone: league.timezone || "UTC",
  });
}

function normalizeRows(rawRows, league, source, options = {}) {
  const window = options.window || buildDiscoveryWindow({ mode: "incremental", source, league });
  return rawRows
    .filter((row) => row && row.id && row.home && row.away)
    .map((row) => {
      const flashscoreId = String(row.id).trim();
      const kickoffUtc = toKickoffUtc(row.date, league, window);
      return {
        flashscore_id: flashscoreId,
        league_code: league.league_code,
        source,
        kickoff_raw: row.date ? String(row.date).trim() : null,
        kickoff_datetime_utc: kickoffUtc,
        home_team: String(row.home).trim(),
        away_team: String(row.away).trim(),
      };
    });
}

function applyWindow(records, window) {
  const minTs = window.start.getTime();
  const maxTs = window.end.getTime();

  return records.filter((record) => {
    if (!record.kickoff_datetime_utc) {
      return true;
    }
    const ts = Date.parse(record.kickoff_datetime_utc);
    if (Number.isNaN(ts)) {
      return true;
    }
    return ts >= minTs && ts <= maxTs;
  });
}

function mergeAndSortRecords(existing, incoming) {
  const byId = new Map();
  for (const record of existing || []) {
    if (record && record.flashscore_id) {
      byId.set(record.flashscore_id, record);
    }
  }

  for (const record of incoming || []) {
    const prev = byId.get(record.flashscore_id);
    if (!prev) {
      byId.set(record.flashscore_id, record);
      continue;
    }
    const mergedSources = Array.from(new Set([].concat(prev.source || [], record.source || []))).sort();
    byId.set(record.flashscore_id, {
      ...prev,
      ...record,
      source: mergedSources,
      kickoff_datetime_utc: record.kickoff_datetime_utc || prev.kickoff_datetime_utc,
      kickoff_raw: record.kickoff_raw || prev.kickoff_raw,
    });
  }

  return Array.from(byId.values()).sort((a, b) => {
    if (a.kickoff_datetime_utc && b.kickoff_datetime_utc) {
      const byDate = a.kickoff_datetime_utc.localeCompare(b.kickoff_datetime_utc);
      if (byDate !== 0) {
        return byDate;
      }
    } else if (a.kickoff_datetime_utc) {
      return -1;
    } else if (b.kickoff_datetime_utc) {
      return 1;
    }
    return String(a.flashscore_id).localeCompare(String(b.flashscore_id));
  });
}

function writeJsonAtomic(filePath, payload) {
  const tmpPath = `${filePath}.tmp`;
  fs.writeFileSync(tmpPath, payload);
  fs.renameSync(tmpPath, filePath);
}

function writeLegacyOutput(leagueCode, filenamePrefix, records) {
  ensureDir(LEGACY_DATA_DIR);
  const legacyRecords = records.map((record) => ({
    id: record.flashscore_id,
    flashscore_id: record.flashscore_id,
    date: record.kickoff_raw,
    kickoff_raw: record.kickoff_raw,
    home: record.home_team,
    home_team: record.home_team,
    away: record.away_team,
    away_team: record.away_team,
    league_code: record.league_code,
    kickoff_datetime_utc: record.kickoff_datetime_utc,
  }));
  const outPath = path.join(LEGACY_DATA_DIR, `${filenamePrefix}_${leagueCode}.json`);
  writeJsonAtomic(outPath, JSON.stringify(legacyRecords, null, 2));
  return outPath;
}

function writeV1IdsOutput(leagueCode, filenamePrefix, records) {
  ensureDir(V1_IDS_DIR);
  const legacyCompatibleRecords = records.map((record) => ({
    id: record.flashscore_id,
    flashscore_id: record.flashscore_id,
    date: record.kickoff_raw,
    kickoff_raw: record.kickoff_raw,
    home: record.home_team,
    home_team: record.home_team,
    away: record.away_team,
    away_team: record.away_team,
    league_code: record.league_code,
    kickoff_datetime_utc: record.kickoff_datetime_utc,
  }));
  const outPath = path.join(V1_IDS_DIR, `${filenamePrefix}_${leagueCode}.json`);
  writeJsonAtomic(outPath, JSON.stringify(legacyCompatibleRecords, null, 2));
  return outPath;
}

function writeCanonicalOutput(leagueCode, records) {
  ensureDir(V1_DISCOVERY_DIR);
  const canonicalPath = path.join(V1_DISCOVERY_DIR, `discovery_fixtures_${leagueCode}.json`);
  const merged = mergeAndSortRecords([], records);
  writeJsonAtomic(canonicalPath, JSON.stringify(merged, null, 2));
  return canonicalPath;
}

module.exports = {
  applyWindow,
  buildDiscoveryWindow,
  mergeAndSortRecords,
  normalizeRows,
  writeCanonicalOutput,
  writeLegacyOutput,
  writeV1IdsOutput,
};
