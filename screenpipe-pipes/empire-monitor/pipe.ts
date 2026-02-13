/**
 * Empire Monitor Pipe for Screenpipe
 * ====================================
 * Runs every 30 seconds, scanning recent screen captures for:
 * - WordPress errors (500, database error, fatal error)
 * - GeeLark failures (profile failed, rate limit, captcha)
 * - n8n workflow failures (execution error, timeout)
 *
 * Sends alerts to empire-dashboard at localhost:8000/api/alerts.
 */

import { pipe } from "@screenpipe/js";

interface AlertPattern {
  category: string;
  severity: "critical" | "warning" | "info";
  keywords: string[];
  /** Only match when OCR comes from these apps (empty = any app) */
  apps: string[];
}

// Apps captured by screenpipe on this machine
const BROWSER_APPS = ["Google Chrome", "Microsoft Edge", "Firefox"];
const TERMINAL_APPS = ["Windows Terminal Host", "WindowsTerminal"];
const ALL_APPS: string[] = []; // empty = match any app

const ALERT_PATTERNS: AlertPattern[] = [
  // WordPress site errors (visible in browser or terminal curl output)
  {
    category: "wordpress_error",
    severity: "critical",
    keywords: [
      "500 Internal Server Error",
      "Error establishing a database connection",
      "PHP Fatal error",
      "white screen of death",
      "503 Service Temporarily Unavailable",
    ],
    apps: [...BROWSER_APPS, ...TERMINAL_APPS],
  },
  {
    category: "wordpress_warning",
    severity: "warning",
    keywords: [
      "PHP Warning",
      "WP_Error",
      "cURL error 28",   // timeout
      "cURL error 7",    // connection refused
      "REST API error",
    ],
    apps: [...BROWSER_APPS, ...TERMINAL_APPS],
  },
  // GeeLark automation failures (visible in terminal running automation scripts)
  {
    category: "geelark_error",
    severity: "critical",
    keywords: [
      "profile failed to start",
      "cloud phone error",
      "device disconnected",
      "ADB connection lost",
      "GeeLarkError",
      "profile_start_failed",
    ],
    apps: TERMINAL_APPS,
  },
  {
    category: "geelark_rate_limit",
    severity: "warning",
    keywords: [
      "rate limit exceeded",
      "too many requests",
      "HTTP 429",
      "captcha detected",
      "RateLimitExceeded",
    ],
    apps: TERMINAL_APPS,
  },
  {
    category: "geelark_ban",
    severity: "critical",
    keywords: [
      "account suspended",
      "account banned",
      "permanently banned",
      "violated community guidelines",
      "your account has been locked",
    ],
    apps: ALL_APPS, // Could appear in browser or terminal
  },
  // n8n workflow failures (visible in browser dashboard or terminal)
  {
    category: "n8n_error",
    severity: "critical",
    keywords: [
      "Workflow execution failed",
      "execution error",
      "ERROR: Workflow",
    ],
    apps: [...BROWSER_APPS, ...TERMINAL_APPS],
  },
  {
    category: "n8n_warning",
    severity: "warning",
    keywords: [
      "execution timed out",
      "retry limit reached",
      "ECONNREFUSED",
    ],
    apps: [...BROWSER_APPS, ...TERMINAL_APPS],
  },
  // Automation pipeline errors (Python tracebacks in terminal)
  {
    category: "automation_crash",
    severity: "critical",
    keywords: [
      "Traceback (most recent call last)",
      "UnhandledPromiseRejection",
      "FATAL ERROR",
    ],
    apps: TERMINAL_APPS,
  },
  // Service health (screenpipe itself, vision service)
  {
    category: "service_down",
    severity: "warning",
    keywords: [
      "Connection refused localhost:8002",
      "Connection refused localhost:3030",
      "Connection refused localhost:8000",
      "Connection refused localhost:8001",
    ],
    apps: TERMINAL_APPS,
  },
];

// Track recent alerts to avoid duplicates
const recentAlerts: Map<string, number> = new Map();

function getConfig(): { dashboardUrl: string; cooldownMs: number } {
  return {
    dashboardUrl:
      (pipe as any).config?.dashboard_url || "http://localhost:8000",
    cooldownMs:
      ((pipe as any).config?.alert_cooldown_minutes || 5) * 60 * 1000,
  };
}

function shouldAlert(alertKey: string, cooldownMs: number): boolean {
  const lastSent = recentAlerts.get(alertKey);
  if (lastSent && Date.now() - lastSent < cooldownMs) {
    return false;
  }
  recentAlerts.set(alertKey, Date.now());
  // Clean old entries
  for (const [key, ts] of recentAlerts.entries()) {
    if (Date.now() - ts > cooldownMs * 2) {
      recentAlerts.delete(key);
    }
  }
  return true;
}

async function sendAlert(
  dashboardUrl: string,
  category: string,
  severity: string,
  message: string,
  appName: string,
  timestamp: string
): Promise<void> {
  try {
    await fetch(`${dashboardUrl}/api/alerts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: "screenpipe-empire-monitor",
        category,
        severity,
        message,
        app_name: appName,
        timestamp,
        detected_at: new Date().toISOString(),
      }),
    });
  } catch (e) {
    // Dashboard might not be running; log but don't crash
    console.error(`Failed to send alert to dashboard: ${e}`);
  }

  // Also send desktop notification for critical alerts
  if (severity === "critical") {
    try {
      await pipe.sendDesktopNotification({
        title: `Empire Alert: ${category}`,
        body: message.substring(0, 200),
      });
    } catch {
      // Notification API may not be available
    }
  }
}

async function checkRecentScreenContent(): Promise<void> {
  const { dashboardUrl, cooldownMs } = getConfig();

  // Query last 30 seconds of screen content
  const thirtySecsAgo = new Date(Date.now() - 30_000).toISOString();

  let results;
  try {
    results = await pipe.queryScreenpipe({
      contentType: "ocr",
      limit: 20,
      startTime: thirtySecsAgo,
    });
  } catch (e) {
    console.error(`Screenpipe query failed: ${e}`);
    return;
  }

  if (!results?.data?.length) return;

  for (const item of results.data) {
    const text = item.content?.text || "";
    const appName = item.content?.app_name || "unknown";
    const timestamp = item.content?.timestamp || new Date().toISOString();
    const textLower = text.toLowerCase();

    for (const pattern of ALERT_PATTERNS) {
      // Skip if pattern requires specific apps and this isn't one of them
      if (pattern.apps.length > 0 && !pattern.apps.some(a => a.toLowerCase() === appName.toLowerCase())) {
        continue;
      }

      for (const keyword of pattern.keywords) {
        if (textLower.includes(keyword.toLowerCase())) {
          const alertKey = `${pattern.category}:${keyword}`;
          if (shouldAlert(alertKey, cooldownMs)) {
            // Extract context around the keyword
            const idx = textLower.indexOf(keyword.toLowerCase());
            const start = Math.max(0, idx - 50);
            const end = Math.min(text.length, idx + keyword.length + 100);
            const context = text.substring(start, end).trim();

            await sendAlert(
              dashboardUrl,
              pattern.category,
              pattern.severity,
              `[${appName}] ${context}`,
              appName,
              timestamp
            );

            console.log(
              `[empire-monitor] ${pattern.severity.toUpperCase()}: ${pattern.category} in ${appName} - ${keyword}`
            );
          }
          break; // One alert per pattern per item
        }
      }
    }
  }
}

// Main execution — Screenpipe calls this on the cron schedule
checkRecentScreenContent().catch(console.error);
