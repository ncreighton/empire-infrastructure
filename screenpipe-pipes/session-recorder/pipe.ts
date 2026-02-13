/**
 * Session Recorder Pipe for Screenpipe
 * =======================================
 * Runs every 60 seconds. Correlates Screenpipe screen captures with
 * GeeLark automation job IDs from the automation API for post-mortem debugging.
 *
 * Creates session logs that link:
 * - Screenpipe timestamps + OCR text
 * - Active automation job IDs from automation_api.py
 * - App names and window context
 *
 * Logs are written to JSON files in the session_log_dir.
 */

import { pipe } from "@screenpipe/js";

interface AutomationJob {
  job_id: string;
  profile_id: string;
  platform: string;
  status: string;
  started_at: string;
}

interface SessionEntry {
  timestamp: string;
  screenpipe_text: string;
  app_name: string;
  window_name: string;
  active_jobs: AutomationJob[];
}

interface SessionLog {
  session_id: string;
  started_at: string;
  entries: SessionEntry[];
}

function getConfig(): { automationApiUrl: string; sessionLogDir: string } {
  return {
    automationApiUrl:
      (pipe as any).config?.automation_api_url || "http://localhost:8001",
    sessionLogDir:
      (pipe as any).config?.session_log_dir ||
      "D:\\Claude Code Projects\\geelark-automation\\logs\\sessions",
  };
}

async function getActiveJobs(apiUrl: string): Promise<AutomationJob[]> {
  try {
    const resp = await fetch(`${apiUrl}/run/status`);
    if (resp.ok) {
      const data = await resp.json();
      // Extract active jobs from the status response
      const jobs: AutomationJob[] = [];
      if (data.active_jobs) {
        for (const job of data.active_jobs) {
          jobs.push({
            job_id: job.job_id || job.id || "unknown",
            profile_id: job.profile_id || "",
            platform: job.platform || "",
            status: job.status || "running",
            started_at: job.started_at || "",
          });
        }
      }
      // Also check last_run for recently completed
      if (data.last_run) {
        jobs.push({
          job_id: data.last_run.job_id || "last",
          profile_id: data.last_run.profile_id || "",
          platform: data.last_run.platform || "",
          status: data.last_run.status || "completed",
          started_at: data.last_run.started_at || "",
        });
      }
      return jobs;
    }
  } catch {
    // Automation API might not be running
  }
  return [];
}

async function recordSession(): Promise<void> {
  const { automationApiUrl, sessionLogDir } = getConfig();

  // Query last 60 seconds of screen content
  const oneMinAgo = new Date(Date.now() - 60_000).toISOString();

  let screenResults;
  try {
    screenResults = await pipe.queryScreenpipe({
      contentType: "ocr",
      limit: 30,
      startTime: oneMinAgo,
    });
  } catch (e) {
    console.error(`Screenpipe query failed: ${e}`);
    return;
  }

  if (!screenResults?.data?.length) return;

  // Get active automation jobs
  const activeJobs = await getActiveJobs(automationApiUrl);

  // Only record when there are active jobs (to avoid noise)
  if (activeJobs.length === 0) return;

  const entries: SessionEntry[] = [];
  for (const item of screenResults.data) {
    const content = item.content || {};
    entries.push({
      timestamp: content.timestamp || new Date().toISOString(),
      screenpipe_text: (content.text || "").substring(0, 500),
      app_name: content.app_name || "unknown",
      window_name: content.window_name || "",
      active_jobs: activeJobs,
    });
  }

  // Generate session ID from timestamp
  const now = new Date();
  const sessionId = `session-${now.toISOString().replace(/[:.]/g, "-").substring(0, 19)}`;

  const sessionLog: SessionLog = {
    session_id: sessionId,
    started_at: now.toISOString(),
    entries,
  };

  // Write session log
  // Note: In Screenpipe pipes, file I/O may need to go through the host API
  // For now, send to the automation API for storage
  try {
    await fetch(`${automationApiUrl}/run/session-log`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sessionLog),
    });
    console.log(
      `[session-recorder] Logged ${entries.length} entries for ${activeJobs.length} active jobs`
    );
  } catch {
    // If automation API doesn't have this endpoint yet, that's OK
    console.log(
      `[session-recorder] ${entries.length} entries captured (API storage unavailable)`
    );
  }
}

// Main execution — Screenpipe calls this on the cron schedule
recordSession().catch(console.error);
