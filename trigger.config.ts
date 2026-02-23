import { defineConfig } from "@trigger.dev/sdk";

export default defineConfig({
  // Project ID from Trigger.dev dashboard
  // Get this from: https://cloud.trigger.dev → Your Project → Settings
  project: process.env.TRIGGER_PROJECT_ID || "",
  
  // Directories containing task files
  dirs: ["./trigger"],
  
  // Build extensions for Python tasks
  buildExtensions: [
    {
      name: "python",
      package: "@trigger.dev/python",
    },
  ],
  
  // Machine preset for compute-heavy tasks
  machine: "medium-1x",
  
  // Maximum task duration (30 minutes)
  maxDuration: 1800,
});
