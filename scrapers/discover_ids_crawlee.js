const { spawnSync } = require("child_process");
const path = require("path");

const args = process.argv.slice(2);
const target = path.join(__dirname, "discover_upcoming_ids.js");

const cmdArgs = [target, "--mode", "incremental"];
if (args.length && !args[0].startsWith("-")) {
  cmdArgs.push("--league", args[0]);
}

const result = spawnSync(process.execPath, cmdArgs, { stdio: "inherit" });
process.exit(result.status || 0);
