const { spawnSync } = require("child_process");
const path = require("path");

const target = path.join(__dirname, "discover_upcoming_ids.js");
const result = spawnSync(process.execPath, [target, "--mode", "seed"], { stdio: "inherit" });
process.exit(result.status || 0);
