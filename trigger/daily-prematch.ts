import { schedules, task } from "@trigger.dev/sdk";
import { python } from "@trigger.dev/python";

/**
 * Daily Prematch Pipeline
 * 
 * Runs every morning at 6:00 UTC to:
 * 1. Scrape prematch data (odds + injuries) for upcoming fixtures
 * 2. Ingest data to database
 * 3. Run AI research enrichment for fixtures with injury data
 */

export const dailyPrematchPipeline = schedules.task({
  id: "daily-prematch-pipeline",
  
  // Run daily at 6:00 AM UTC
  cron: {
    pattern: "0 6 * * *",
    timezone: "UTC",
  },
  
  // Retry configuration
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 60_000,
    maxTimeoutInMs: 600_000,
  },
  
  // Run the Python pipeline script
  run: async (payload, { ctx }) => {
    console.log(`[${new Date().toISOString()}] Starting daily prematch pipeline`);
    console.log(`Schedule ID: ${payload.scheduleId}`);
    console.log(`Scheduled time: ${payload.timestamp}`);
    
    // Execute Python pipeline
    const result = await python.runScript({
      script: "src/pipelines/daily_prematch_pipeline.py",
      args: [
        "--days", "3",
        "--research-limit", "10",
      ],
      env: {
        DATABASE_URL: process.env.DATABASE_URL!,
        TAVILY_API_KEY: process.env.TAVILY_API_KEY!,
      },
    });
    
    if (result.exitCode !== 0) {
      throw new Error(`Pipeline failed with exit code ${result.exitCode}: ${result.stderr}`);
    }
    
    console.log("Pipeline output:", result.stdout);
    
    return {
      success: true,
      exitCode: result.exitCode,
      output: result.stdout,
    };
  },
});

/**
 * Hourly Tick Job
 * 
 * Runs every hour to:
 * 1. Settle fixtures that have finished
 * 2. Run predictions for upcoming fixtures
 * 3. Score past predictions
 */

export const hourlyTickJob = schedules.task({
  id: "hourly-tick-job",
  
  // Run every hour
  cron: {
    pattern: "0 * * * *",
    timezone: "UTC",
  },
  
  retry: {
    maxAttempts: 2,
    factor: 2,
    minTimeoutInMs: 30_000,
    maxTimeoutInMs: 300_000,
  },
  
  run: async (payload, { ctx }) => {
    console.log(`[${new Date().toISOString()}] Starting hourly tick job`);
    
    const result = await python.runScript({
      script: "src/jobs/tick_due_fixtures_v1.py",
      args: [
        "--max-settle", "25",
        "--max-predict", "50",
      ],
      env: {
        DATABASE_URL: process.env.DATABASE_URL!,
      },
    });
    
    if (result.exitCode !== 0) {
      throw new Error(`Tick job failed with exit code ${result.exitCode}: ${result.stderr}`);
    }
    
    return {
      success: true,
      exitCode: result.exitCode,
      output: result.stdout,
    };
  },
});

/**
 * Manual Trigger: Run Pipeline for Specific Leagues
 * 
 * Trigger manually via API or dashboard for specific leagues
 */

export const runPipelineForLeagues = task({
  id: "run-pipeline-for-leagues",
  
  run: async (payload: { leagues: string[]; days?: number }) => {
    console.log(`Running pipeline for leagues: ${payload.leagues.join(", ")}`);
    
    const args = [
      "--leagues", payload.leagues.join(","),
      "--days", String(payload.days || 3),
    ];
    
    const result = await python.runScript({
      script: "src/pipelines/daily_prematch_pipeline.py",
      args,
      env: {
        DATABASE_URL: process.env.DATABASE_URL!,
        TAVILY_API_KEY: process.env.TAVILY_API_KEY!,
      },
    });
    
    return {
      success: result.exitCode === 0,
      exitCode: result.exitCode,
      output: result.stdout,
    };
  },
});
