"""
GSC + Bing + Visual Audit Pipeline
Pulls live data from Google Search Console and Bing Webmaster APIs for all 14 sites.
Identifies top pages by traffic, runs article + ZIMM visual audits, applies safe fixes.

Usage:
    python gsc_top_pages_audit.py                          # Full pipeline
    python gsc_top_pages_audit.py --audit-only             # No fixes applied
    python gsc_top_pages_audit.py --site smarthomewizards  # Single site
    python gsc_top_pages_audit.py --skip-visual            # Skip ZIMM visual audits
    python gsc_top_pages_audit.py --top 10                 # Top 10 pages per site
"""
import argparse
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"D:\Claude Code Projects")
CREDENTIALS_DIR = BASE_DIR / "credentials"
CONFIG_PATH = BASE_DIR / "config" / "sites.json"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

AUDIT_API_BASE = "http://217.216.84.245:8001"

# GSC property URLs for each site
GSC_PROPERTIES = {
    "smarthomewizards": "https://smarthomewizards.com/",
    "mythicalarchives": "https://mythicalarchives.com/",
    "bulletjournals": "https://bulletjournals.net/",
    "witchcraftforbeginners": "sc-domain:witchcraftforbeginners.com",
    "wealthfromai": "https://wealthfromai.com/",
    "aidiscoverydigest": "https://aidiscoverydigest.com/",
    "aiinactionhub": "https://aiinactionhub.com/",
    "pulsegearreviews": "https://pulsegearreviews.com/",
    "wearablegearreviews": "https://wearablegearreviews.com/",
    "smarthomegearreviews": "https://smarthomegearreviews.com/",
    "clearainews": "https://clearainews.com/",
    "theconnectedhaven": "https://theconnectedhaven.com/",
    "manifestandalign": "https://manifestandalign.com/",
    "familyflourish": "https://family-flourish.com/",
}

# Traffic level thresholds (combined GSC + Bing clicks)
TRAFFIC_HIGH = 50
TRAFFIC_MEDIUM = 10

# Fixes safe for ALL traffic levels (additive only)
SAFE_ALL = {
    "add_image_alt",
    "set_og_title",
    "set_og_description",
    "set_twitter_card",
    "suggest_focus_keyword",
    "add_link_rel_noopener",
    "fix_html_entities",
    "set_featured_image_alt",
    "improve_image_alt",
}

# Fixes safe for MEDIUM and LOW traffic only
SAFE_MEDIUM_LOW = {
    "expand_meta_title",
    "generate_meta_description",
    "expand_meta_description",
    "shorten_meta_title",
    "shorten_meta_description",
    "remove_noindex",
}

# Never auto-apply on high-traffic pages
NEVER_AUTO_HIGH = {
    "add_internal_links",
    "remove_self_link",
    "generate_and_set_featured_image",
}


# =============================================================================
# Credential Loading
# =============================================================================

def load_gsc_credentials():
    """Load Google OAuth credentials and refresh the access token."""
    creds_path = CREDENTIALS_DIR / "google" / "oauth_credentials.json"
    with open(creds_path, "r") as f:
        creds = json.load(f)

    # Refresh the access token
    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token",
    })
    resp.raise_for_status()
    token_data = resp.json()
    return token_data["access_token"]


def load_bing_api_key():
    """Load Bing Webmaster API key."""
    key_path = CREDENTIALS_DIR / "bing" / "api_key.json"
    with open(key_path, "r") as f:
        data = json.load(f)
    return data["api_key"]


def load_sites_config():
    """Load WordPress site configurations."""
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    return data.get("sites", data)


# =============================================================================
# Phase 1a: GSC Data
# =============================================================================

def pull_gsc_data(access_token, site_slug, gsc_property, top_n=20):
    """Pull top pages and queries from Google Search Console."""
    headers = {"Authorization": f"Bearer {access_token}"}
    base_url = "https://searchconsole.googleapis.com/webmasters/v3"

    end_date = datetime.now() - timedelta(days=3)
    start_date = end_date - timedelta(days=28)
    date_range = {
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
    }

    result = {
        "site_slug": site_slug,
        "gsc_property": gsc_property,
        "pages": [],
        "page_queries": {},
        "totals": {"clicks": 0, "impressions": 0, "ctr": 0, "position": 0},
        "error": None,
    }

    # Query 1: Top pages by clicks
    try:
        resp = requests.post(
            f"{base_url}/sites/{requests.utils.quote(gsc_property, safe='')}/searchAnalytics/query",
            headers=headers,
            json={
                **date_range,
                "dimensions": ["page"],
                "rowLimit": top_n,
                "dataState": "final",
            },
        )
        resp.raise_for_status()
        rows = resp.json().get("rows", [])

        for row in rows:
            page_url = row["keys"][0]
            result["pages"].append({
                "url": page_url,
                "gsc_clicks": row.get("clicks", 0),
                "gsc_impressions": row.get("impressions", 0),
                "gsc_ctr": round(row.get("ctr", 0), 4),
                "gsc_position": round(row.get("position", 0), 1),
            })

        # Calculate totals
        result["totals"]["clicks"] = sum(r.get("clicks", 0) for r in rows)
        result["totals"]["impressions"] = sum(r.get("impressions", 0) for r in rows)
        if result["totals"]["impressions"] > 0:
            result["totals"]["ctr"] = round(
                result["totals"]["clicks"] / result["totals"]["impressions"], 4
            )
            result["totals"]["position"] = round(
                sum(r.get("position", 0) * r.get("impressions", 0) for r in rows)
                / result["totals"]["impressions"],
                1,
            )

        time.sleep(1)  # Rate limit

    except Exception as e:
        result["error"] = f"GSC pages query failed: {e}"
        return result

    # Query 2: Top queries per page
    try:
        resp = requests.post(
            f"{base_url}/sites/{requests.utils.quote(gsc_property, safe='')}/searchAnalytics/query",
            headers=headers,
            json={
                **date_range,
                "dimensions": ["page", "query"],
                "rowLimit": 500,
                "dataState": "final",
            },
        )
        resp.raise_for_status()
        rows = resp.json().get("rows", [])

        for row in rows:
            page_url = row["keys"][0]
            query = row["keys"][1]
            if page_url not in result["page_queries"]:
                result["page_queries"][page_url] = []
            result["page_queries"][page_url].append({
                "query": query,
                "clicks": row.get("clicks", 0),
                "impressions": row.get("impressions", 0),
                "position": round(row.get("position", 0), 1),
            })

        time.sleep(1)  # Rate limit

    except Exception as e:
        # Non-fatal: page data is still useful without per-page queries
        print(f"    [WARN] Page-query data failed: {e}")

    return result


# =============================================================================
# Phase 1b: Bing Webmaster Data
# =============================================================================

def pull_bing_data(api_key, site_slug, domain):
    """Pull search queries, traffic, and crawl issues from Bing Webmaster."""
    base_url = "https://ssl.bing.com/webmaster/api.svc/json"
    site_url = f"https://{domain}/"

    result = {
        "site_slug": site_slug,
        "queries": [],
        "traffic": None,
        "crawl_issues": [],
        "totals": {"clicks": 0, "impressions": 0},
        "error": None,
    }

    # Query stats
    try:
        resp = requests.get(
            f"{base_url}/GetQueryStats",
            params={"apikey": api_key, "siteUrl": site_url},
            timeout=30,
        )
        if resp.status_code == 200:
            queries = resp.json().get("d", [])
            if queries:
                for q in queries:
                    result["queries"].append({
                        "query": q.get("Query", ""),
                        "clicks": q.get("Clicks", 0),
                        "impressions": q.get("Impressions", 0),
                        "position": q.get("AvgClickPosition", 0),
                    })
                result["totals"]["clicks"] = sum(q.get("Clicks", 0) for q in queries)
                result["totals"]["impressions"] = sum(
                    q.get("Impressions", 0) for q in queries
                )
        else:
            result["error"] = f"Bing query stats: HTTP {resp.status_code}"
    except Exception as e:
        result["error"] = f"Bing query stats: {e}"

    time.sleep(0.5)

    # Traffic stats
    try:
        resp = requests.get(
            f"{base_url}/GetRankAndTrafficStats",
            params={"apikey": api_key, "siteUrl": site_url},
            timeout=30,
        )
        if resp.status_code == 200:
            result["traffic"] = resp.json().get("d", None)
    except Exception as e:
        print(f"    [WARN] Bing traffic stats failed: {e}")

    time.sleep(0.5)

    # Crawl issues
    try:
        resp = requests.get(
            f"{base_url}/GetCrawlIssues",
            params={"apikey": api_key, "siteUrl": site_url},
            timeout=30,
        )
        if resp.status_code == 200:
            issues = resp.json().get("d", [])
            if issues:
                result["crawl_issues"] = issues
    except Exception as e:
        print(f"    [WARN] Bing crawl issues failed: {e}")

    return result


# =============================================================================
# Phase 2: Map URLs to WordPress Post IDs
# =============================================================================

def extract_slug(url):
    """Extract the slug from a URL path."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    # Take the last segment as slug
    segments = [s for s in path.split("/") if s]
    return segments[-1] if segments else None


def map_urls_to_posts(pages, site_config, domain):
    """Map page URLs to WordPress post IDs via REST API."""
    wp_user = site_config.get("wordpress", {}).get("user", "")
    wp_pass = site_config.get("wordpress", {}).get("app_password", "")
    base_api = f"https://{domain}/wp-json/wp/v2"
    auth = (wp_user, wp_pass)

    mapped = []
    skipped = []

    for page in pages:
        url = page["url"]
        slug = extract_slug(url)

        if not slug:
            skipped.append({"url": url, "reason": "no slug"})
            continue

        # Skip non-content URLs
        skip_patterns = ["/category/", "/tag/", "/author/", "/page/", "/feed/",
                         "/wp-", "/archive", "/sitemap"]
        if any(p in url.lower() for p in skip_patterns):
            skipped.append({"url": url, "reason": "non-content URL"})
            continue

        post_id = None
        post_type = None

        # Try posts first
        try:
            resp = requests.get(
                f"{base_api}/posts",
                params={"slug": slug, "_fields": "id,title,status"},
                auth=auth,
                timeout=15,
            )
            if resp.status_code == 200:
                posts = resp.json()
                if posts:
                    post_id = posts[0]["id"]
                    post_type = "post"
        except Exception as e:
            print(f"    [WARN] WP posts lookup failed for {slug}: {e}")

        # Try pages if not found as post
        if not post_id:
            try:
                resp = requests.get(
                    f"{base_api}/pages",
                    params={"slug": slug, "_fields": "id,title,status"},
                    auth=auth,
                    timeout=15,
                )
                if resp.status_code == 200:
                    wp_pages = resp.json()
                    if wp_pages:
                        post_id = wp_pages[0]["id"]
                        post_type = "page"
            except Exception as e:
                print(f"    [WARN] WP pages lookup failed for {slug}: {e}")

        if post_id:
            mapped.append({
                **page,
                "slug": slug,
                "post_id": post_id,
                "post_type": post_type,
            })
        else:
            skipped.append({"url": url, "reason": "not found in WordPress"})

        time.sleep(0.3)  # Don't hammer the WP API

    return mapped, skipped


def classify_traffic(page):
    """Classify a page's traffic level based on combined GSC + Bing clicks."""
    combined = page.get("combined_clicks", 0)
    if combined >= TRAFFIC_HIGH:
        return "HIGH"
    elif combined >= TRAFFIC_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"


# =============================================================================
# Phase 3a: Article Audits
# =============================================================================

def run_article_audits(pages_by_site):
    """Run article audits via the audit API for all mapped pages."""
    results = {}
    total = sum(len(pages) for pages in pages_by_site.values())
    done = 0

    for site_slug, pages in pages_by_site.items():
        results[site_slug] = []
        batch = []

        for page in pages:
            batch.append(page)

            if len(batch) >= 5:
                for p in batch:
                    done += 1
                    audit = _run_single_audit(site_slug, p, done, total)
                    results[site_slug].append(audit)
                batch = []
                time.sleep(2)

        # Process remaining
        for p in batch:
            done += 1
            audit = _run_single_audit(site_slug, p, done, total)
            results[site_slug].append(audit)

    return results


def _run_single_audit(site_slug, page, current, total):
    """Run a single article audit."""
    post_id = page["post_id"]
    print(f"  [{current}/{total}] Auditing {site_slug} post {post_id}...", end=" ")

    try:
        resp = requests.post(
            f"{AUDIT_API_BASE}/audit",
            json={
                "site_id": site_slug,
                "post_id": post_id,
                "workflow_type": "zimm_standard",
            },
            timeout=120,
        )
        if resp.status_code == 200:
            raw = resp.json()
            # API returns {audit_id, summary, report} — flatten for our use
            report = raw.get("report", {})
            audit_id = raw.get("audit_id", "")
            score = report.get("overall_score", 0)
            issues = report.get("issues", [])
            print(f"Score: {score}, Issues: {len(issues)}")
            return {
                **page,
                "audit": {
                    "audit_id": audit_id,
                    "overall_score": score,
                    "status": report.get("status", ""),
                    "issues": issues,
                    "auto_fixable_count": report.get("auto_fixable_count", 0),
                    "checks_total": report.get("checks_total", 0),
                },
            }
        else:
            print(f"HTTP {resp.status_code}")
            return {**page, "audit": {"error": f"HTTP {resp.status_code}"}}
    except Exception as e:
        print(f"Error: {e}")
        return {**page, "audit": {"error": str(e)}}


# =============================================================================
# Phase 3b: ZIMM Visual Audits
# =============================================================================

def run_visual_audits(pages_by_site, sites_config):
    """Run ZIMM visual audits for all mapped pages."""
    results = {}
    total = sum(len(pages) for pages in pages_by_site.values())
    done = 0

    for site_slug, pages in pages_by_site.items():
        domain = sites_config.get(site_slug, {}).get("domain", "")
        results[site_slug] = []
        batch = []

        for page in pages:
            batch.append(page)

            if len(batch) >= 3:
                for p in batch:
                    done += 1
                    visual = _run_single_visual_audit(
                        site_slug, p, domain, done, total
                    )
                    results[site_slug].append(visual)
                batch = []
                time.sleep(3)

        # Process remaining
        for p in batch:
            done += 1
            visual = _run_single_visual_audit(site_slug, p, domain, done, total)
            results[site_slug].append(visual)

    return results


def _run_single_visual_audit(site_slug, page, domain, current, total):
    """Run a single ZIMM visual audit."""
    post_id = page["post_id"]
    post_url = page["url"]

    # Ensure URL has trailing slash
    if not post_url.endswith("/"):
        post_url += "/"

    print(f"  [{current}/{total}] Visual audit {site_slug} post {post_id}...", end=" ")

    try:
        resp = requests.post(
            f"{AUDIT_API_BASE}/zimm/audit-sync",
            json={
                "site_slug": site_slug,
                "post_id": post_id,
                "post_url": post_url,
                "workflow_type": "zimm_standard",
            },
            timeout=120,
        )
        if resp.status_code == 200:
            result = resp.json()
            score = result.get("score", "?")
            status = result.get("status", "?")
            print(f"Score: {score}/100 [{status}]")
            return {**page, "visual_audit": result}
        else:
            print(f"HTTP {resp.status_code}")
            return {**page, "visual_audit": {"error": f"HTTP {resp.status_code}"}}
    except Exception as e:
        print(f"Error: {e}")
        return {**page, "visual_audit": {"error": str(e)}}


# =============================================================================
# Phase 4: Classify Fixes by Safety Level
# =============================================================================

def classify_fixes(audit_result, traffic_level):
    """Classify which fixes are safe to apply given the traffic level."""
    issues = audit_result.get("issues", [])
    safe_fixes = []
    needs_review = []

    for issue in issues:
        if not issue.get("auto_fixable"):
            needs_review.append({**issue, "reason": "not auto-fixable"})
            continue

        fix_action = issue.get("fix_action", "")

        if fix_action in SAFE_ALL:
            safe_fixes.append(issue)
        elif fix_action in SAFE_MEDIUM_LOW and traffic_level in ("MEDIUM", "LOW"):
            safe_fixes.append(issue)
        elif fix_action in NEVER_AUTO_HIGH and traffic_level == "HIGH":
            needs_review.append({**issue, "reason": f"blocked for {traffic_level} traffic"})
        elif traffic_level == "LOW":
            safe_fixes.append(issue)
        else:
            needs_review.append({
                **issue,
                "reason": f"not whitelisted for {traffic_level} traffic",
            })

    return safe_fixes, needs_review


# =============================================================================
# Phase 5: Generate Report
# =============================================================================

def generate_report(
    gsc_data, bing_data, mapped_pages, audit_results, visual_results,
    fix_classifications, timestamp
):
    """Generate comprehensive JSON report and print summary."""
    report = {
        "generated_at": timestamp,
        "pipeline": "gsc_bing_visual_audit",
        "search_performance": {},
        "bing_crawl_issues": {},
        "site_breakdowns": {},
        "top_opportunities": [],
        "low_visual_scores": [],
        "fix_summary": {
            "auto_fixable": 0,
            "needs_review": 0,
            "by_category": {},
        },
    }

    # Search performance summary
    total_gsc_clicks = 0
    total_gsc_impressions = 0
    total_bing_clicks = 0
    total_bing_impressions = 0

    for site_slug in gsc_data:
        gsc = gsc_data[site_slug]
        bing = bing_data.get(site_slug, {})

        gsc_clicks = gsc.get("totals", {}).get("clicks", 0)
        gsc_impr = gsc.get("totals", {}).get("impressions", 0)
        bing_clicks = bing.get("totals", {}).get("clicks", 0)
        bing_impr = bing.get("totals", {}).get("impressions", 0)

        total_gsc_clicks += gsc_clicks
        total_gsc_impressions += gsc_impr
        total_bing_clicks += bing_clicks
        total_bing_impressions += bing_impr

        report["search_performance"][site_slug] = {
            "gsc_clicks": gsc_clicks,
            "gsc_impressions": gsc_impr,
            "bing_clicks": bing_clicks,
            "bing_impressions": bing_impr,
            "combined_clicks": gsc_clicks + bing_clicks,
        }

        # Bing crawl issues
        if bing.get("crawl_issues"):
            report["bing_crawl_issues"][site_slug] = bing["crawl_issues"]

    report["search_performance"]["_totals"] = {
        "gsc_clicks": total_gsc_clicks,
        "gsc_impressions": total_gsc_impressions,
        "bing_clicks": total_bing_clicks,
        "bing_impressions": total_bing_impressions,
        "combined_clicks": total_gsc_clicks + total_bing_clicks,
    }

    # Site breakdowns
    all_opportunities = []

    for site_slug, pages in mapped_pages.items():
        audits = audit_results.get(site_slug, [])
        visuals = visual_results.get(site_slug, [])

        # Build lookup dicts by post_id
        audit_by_id = {a["post_id"]: a.get("audit", {}) for a in audits}
        visual_by_id = {v["post_id"]: v.get("visual_audit", {}) for v in visuals}

        site_issues = 0
        site_visual_scores = []
        page_details = []

        for page in pages:
            pid = page["post_id"]
            audit = audit_by_id.get(pid, {})
            visual = visual_by_id.get(pid, {})
            traffic_level = page.get("traffic_level", "LOW")

            issue_count = len(audit.get("issues", []))
            site_issues += issue_count

            visual_score = visual.get("score", None)
            if visual_score is not None:
                site_visual_scores.append(visual_score)

            safe, review = fix_classifications.get(pid, ([], []))

            page_detail = {
                "url": page["url"],
                "post_id": pid,
                "traffic_level": traffic_level,
                "combined_clicks": page.get("combined_clicks", 0),
                "audit_score": audit.get("overall_score", audit.get("score")),
                "audit_issues": issue_count,
                "visual_score": visual_score,
                "visual_status": visual.get("status"),
                "safe_fixes": len(safe),
                "needs_review": len(review),
            }
            page_details.append(page_detail)

            # Track opportunities
            if issue_count > 0 or (visual_score is not None and visual_score < 70):
                all_opportunities.append({
                    "site": site_slug,
                    "url": page["url"],
                    "post_id": pid,
                    "combined_clicks": page.get("combined_clicks", 0),
                    "traffic_level": traffic_level,
                    "audit_issues": issue_count,
                    "visual_score": visual_score,
                    "safe_fixes": len(safe),
                    "impact_score": page.get("combined_clicks", 0) * (issue_count + 1),
                })

        avg_visual = (
            round(sum(site_visual_scores) / len(site_visual_scores), 1)
            if site_visual_scores
            else None
        )

        report["site_breakdowns"][site_slug] = {
            "pages_analyzed": len(pages),
            "total_issues": site_issues,
            "avg_visual_score": avg_visual,
            "pages": page_details,
        }

    # Top 20 highest-impact opportunities
    all_opportunities.sort(key=lambda x: x.get("impact_score", 0), reverse=True)
    report["top_opportunities"] = all_opportunities[:20]

    # Pages with visual score < 70
    for site_slug, pages in mapped_pages.items():
        visuals = visual_results.get(site_slug, [])
        visual_by_id = {v["post_id"]: v.get("visual_audit", {}) for v in visuals}
        for page in pages:
            visual = visual_by_id.get(page["post_id"], {})
            score = visual.get("score")
            if score is not None and score < 70:
                report["low_visual_scores"].append({
                    "site": site_slug,
                    "url": page["url"],
                    "post_id": page["post_id"],
                    "visual_score": score,
                    "visual_status": visual.get("status"),
                    "issues": [
                        c.get("message", c.get("check_name", ""))
                        for c in visual.get("checks", [])
                        if not c.get("passed")
                    ],
                })

    # Fix summary
    total_auto = 0
    total_review = 0
    by_category = {}

    for pid, (safe, review) in fix_classifications.items():
        total_auto += len(safe)
        total_review += len(review)
        for fix in safe + review:
            cat = fix.get("category", "OTHER")
            by_category[cat] = by_category.get(cat, 0) + 1

    report["fix_summary"] = {
        "auto_fixable": total_auto,
        "needs_review": total_review,
        "by_category": by_category,
    }

    return report


def print_report_summary(report):
    """Print a human-readable summary of the report."""
    print("\n" + "=" * 80)
    print("GSC + BING + VISUAL AUDIT PIPELINE - REPORT")
    print(f"Generated: {report['generated_at']}")
    print("=" * 80)

    # Search Performance
    print("\n--- SEARCH PERFORMANCE (Last 28 Days) ---")
    totals = report["search_performance"].get("_totals", {})
    print(f"{'Site':<25} {'GSC Clicks':>12} {'GSC Impr':>12} {'Bing Clicks':>12} {'Bing Impr':>12} {'Combined':>10}")
    print("-" * 85)

    for site, data in sorted(report["search_performance"].items()):
        if site == "_totals":
            continue
        print(
            f"{site:<25} {data['gsc_clicks']:>12,} {data['gsc_impressions']:>12,} "
            f"{data['bing_clicks']:>12,} {data['bing_impressions']:>12,} "
            f"{data['combined_clicks']:>10,}"
        )

    print("-" * 85)
    print(
        f"{'TOTAL':<25} {totals.get('gsc_clicks', 0):>12,} "
        f"{totals.get('gsc_impressions', 0):>12,} "
        f"{totals.get('bing_clicks', 0):>12,} "
        f"{totals.get('bing_impressions', 0):>12,} "
        f"{totals.get('combined_clicks', 0):>10,}"
    )

    # Bing Crawl Issues
    if report.get("bing_crawl_issues"):
        print("\n--- BING CRAWL ISSUES ---")
        for site, issues in report["bing_crawl_issues"].items():
            print(f"  {site}: {len(issues) if isinstance(issues, list) else issues}")

    # Site Breakdowns
    print("\n--- SITE BREAKDOWN ---")
    print(f"{'Site':<25} {'Pages':>8} {'Issues':>8} {'Avg Visual':>12}")
    print("-" * 55)
    for site, data in sorted(report["site_breakdowns"].items()):
        avg_v = f"{data['avg_visual_score']}/100" if data["avg_visual_score"] else "N/A"
        print(
            f"{site:<25} {data['pages_analyzed']:>8} "
            f"{data['total_issues']:>8} {avg_v:>12}"
        )

    # Top Opportunities
    if report["top_opportunities"]:
        print("\n--- TOP 20 HIGHEST-IMPACT OPPORTUNITIES ---")
        print(f"{'Site':<22} {'URL':<35} {'Traffic':>8} {'Level':<7} {'Issues':>7} {'Visual':>7}")
        print("-" * 90)
        for opp in report["top_opportunities"][:20]:
            url_short = opp["url"].split("/")[-2] if opp["url"].endswith("/") else opp["url"].split("/")[-1]
            url_short = url_short[:33]
            vs = f"{opp['visual_score']}" if opp.get("visual_score") is not None else "N/A"
            print(
                f"{opp['site']:<22} {url_short:<35} "
                f"{opp['combined_clicks']:>8} {opp['traffic_level']:<7} "
                f"{opp['audit_issues']:>7} {vs:>7}"
            )

    # Low Visual Scores
    if report["low_visual_scores"]:
        print(f"\n--- PAGES WITH VISUAL SCORE < 70 ({len(report['low_visual_scores'])} pages) ---")
        for p in report["low_visual_scores"][:15]:
            url_short = p["url"].split("/")[-2] if p["url"].endswith("/") else p["url"].split("/")[-1]
            print(f"  {p['site']:<22} {url_short:<35} Score: {p['visual_score']}/100")
            for issue in p.get("issues", [])[:3]:
                print(f"    - {issue}")

    # Fix Summary
    fs = report["fix_summary"]
    print(f"\n--- FIX SUMMARY ---")
    print(f"  Auto-fixable: {fs['auto_fixable']}")
    print(f"  Needs review: {fs['needs_review']}")
    if fs["by_category"]:
        print("  By category:")
        for cat, count in sorted(fs["by_category"].items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")


# =============================================================================
# Phase 6: Execute Safe Fixes
# =============================================================================

def execute_safe_fixes(audit_results, fix_classifications):
    """Execute safe fixes via the fix API."""
    fix_log = {
        "timestamp": datetime.now().isoformat(),
        "fixes_applied": [],
        "fixes_skipped": [],
        "failures": [],
    }

    for site_slug, audits in audit_results.items():
        for audited_page in audits:
            pid = audited_page["post_id"]
            audit = audited_page.get("audit", {})
            audit_id = audit.get("audit_id", audit.get("id"))

            if not audit_id:
                continue

            safe_fixes, review_fixes = fix_classifications.get(pid, ([], []))

            if not safe_fixes:
                for rf in review_fixes:
                    fix_log["fixes_skipped"].append({
                        "site": site_slug,
                        "post_id": pid,
                        "fix": rf.get("fix_action", rf.get("title", "unknown")),
                        "reason": rf.get("reason", "needs review"),
                    })
                continue

            traffic_level = audited_page.get("traffic_level", "LOW")
            print(
                f"  Fixing {site_slug} post {pid} "
                f"({traffic_level} traffic, {len(safe_fixes)} safe fixes)..."
            )

            # Create fix plan
            try:
                resp = requests.post(
                    f"{AUDIT_API_BASE}/fix/plan/{audit_id}",
                    timeout=30,
                )
                if resp.status_code != 200:
                    fix_log["failures"].append({
                        "site": site_slug,
                        "post_id": pid,
                        "step": "create_plan",
                        "error": f"HTTP {resp.status_code}",
                    })
                    continue

                plan = resp.json()
                plan_id = plan.get("plan_id", plan.get("id", plan.get("report_id")))

                if not plan_id:
                    fix_log["failures"].append({
                        "site": site_slug,
                        "post_id": pid,
                        "step": "create_plan",
                        "error": "no plan_id returned",
                    })
                    continue

            except Exception as e:
                fix_log["failures"].append({
                    "site": site_slug,
                    "post_id": pid,
                    "step": "create_plan",
                    "error": str(e),
                })
                continue

            # Execute plan (auto-approved only)
            try:
                resp = requests.post(
                    f"{AUDIT_API_BASE}/fix/execute/{plan_id}",
                    params={"auto_only": "true"},
                    timeout=120,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    fixes_list = result.get("fixes", [])
                    completed = sum(1 for f in fixes_list if f.get("status") == "completed")
                    failed = sum(1 for f in fixes_list if f.get("status") == "failed")
                    skipped = sum(1 for f in fixes_list if f.get("status") == "skipped")

                    fix_log["fixes_applied"].append({
                        "site": site_slug,
                        "post_id": pid,
                        "traffic_level": traffic_level,
                        "completed": completed,
                        "failed": failed,
                        "skipped": skipped,
                        "plan_id": plan_id,
                        "details": [
                            {"action": f.get("action"), "status": f.get("status"), "result": f.get("result")}
                            for f in fixes_list if f.get("status") == "completed"
                        ],
                    })
                    print(f"    Applied: {completed}, Failed: {failed}, Skipped: {skipped}")
                else:
                    fix_log["failures"].append({
                        "site": site_slug,
                        "post_id": pid,
                        "step": "execute",
                        "error": f"HTTP {resp.status_code}",
                    })
            except Exception as e:
                fix_log["failures"].append({
                    "site": site_slug,
                    "post_id": pid,
                    "step": "execute",
                    "error": str(e),
                })

            time.sleep(1)

    # Log skipped review-only fixes
    for pid, (safe, review) in fix_classifications.items():
        for rf in review:
            fix_log["fixes_skipped"].append({
                "post_id": pid,
                "fix": rf.get("fix_action", rf.get("title", "unknown")),
                "reason": rf.get("reason", "needs review"),
            })

    return fix_log


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GSC + Bing + Visual Audit Pipeline"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run phases 1-5 only, no fixes applied",
    )
    parser.add_argument(
        "--site",
        type=str,
        help="Run for a single site only (e.g., smarthomewizards)",
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="Skip ZIMM visual audits (faster, article audit only)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top pages per site (default: 20)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 80)
    print("GSC + BING + VISUAL AUDIT PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'audit-only' if args.audit_only else 'full (audit + fix)'}")
    if args.site:
        print(f"Site filter: {args.site}")
    print(f"Top pages per site: {args.top}")
    print(f"Visual audits: {'skipped' if args.skip_visual else 'enabled'}")
    print("=" * 80)

    # Load configs
    sites_config = load_sites_config()

    # Determine which sites to process
    if args.site:
        if args.site not in sites_config:
            print(f"[ERROR] Site '{args.site}' not found in config/sites.json")
            return
        target_sites = {args.site: sites_config[args.site]}
    else:
        target_sites = sites_config

    # =========================================================================
    # Phase 1a: Pull GSC Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1a: GOOGLE SEARCH CONSOLE DATA")
    print("=" * 80)

    try:
        gsc_token = load_gsc_credentials()
        print("[OK] GSC access token refreshed")
    except Exception as e:
        print(f"[ERROR] Failed to refresh GSC token: {e}")
        gsc_token = None

    gsc_data = {}
    for site_slug in target_sites:
        gsc_property = GSC_PROPERTIES.get(site_slug)
        if not gsc_property:
            print(f"  [{site_slug}] No GSC property configured, skipping")
            gsc_data[site_slug] = {
                "site_slug": site_slug,
                "pages": [],
                "page_queries": {},
                "totals": {"clicks": 0, "impressions": 0},
                "error": "no GSC property",
            }
            continue

        if not gsc_token:
            gsc_data[site_slug] = {
                "site_slug": site_slug,
                "pages": [],
                "page_queries": {},
                "totals": {"clicks": 0, "impressions": 0},
                "error": "no access token",
            }
            continue

        print(f"  [{site_slug}] Fetching top {args.top} pages...", end=" ")
        data = pull_gsc_data(gsc_token, site_slug, gsc_property, top_n=args.top)
        gsc_data[site_slug] = data

        if data["error"]:
            print(f"ERROR: {data['error']}")
        else:
            clicks = data["totals"]["clicks"]
            pages = len(data["pages"])
            print(f"{clicks:,} clicks, {pages} pages")

    # =========================================================================
    # Phase 1b: Pull Bing Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1b: BING WEBMASTER DATA")
    print("=" * 80)

    try:
        bing_key = load_bing_api_key()
        print("[OK] Bing API key loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load Bing API key: {e}")
        bing_key = None

    bing_data = {}
    for site_slug, site_config in target_sites.items():
        domain = site_config.get("domain", "")
        if not domain or not bing_key:
            bing_data[site_slug] = {
                "site_slug": site_slug,
                "queries": [],
                "totals": {"clicks": 0, "impressions": 0},
            }
            continue

        print(f"  [{site_slug}] Fetching Bing data...", end=" ")
        data = pull_bing_data(bing_key, site_slug, domain)
        bing_data[site_slug] = data

        if data.get("error"):
            print(f"WARN: {data['error']}")
        else:
            clicks = data["totals"]["clicks"]
            queries = len(data["queries"])
            crawl = len(data.get("crawl_issues", []))
            print(f"{clicks:,} clicks, {queries} queries, {crawl} crawl issues")

    # =========================================================================
    # Phase 2: Map URLs to WordPress Post IDs
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: MAP URLs TO WORDPRESS POSTS")
    print("=" * 80)

    mapped_pages = {}
    all_skipped = {}

    for site_slug, site_config in target_sites.items():
        domain = site_config.get("domain", "")
        gsc = gsc_data.get(site_slug, {})
        pages = gsc.get("pages", [])

        if not pages:
            print(f"  [{site_slug}] No pages to map")
            mapped_pages[site_slug] = []
            continue

        print(f"  [{site_slug}] Mapping {len(pages)} URLs to WP posts...", end=" ")

        # Merge Bing click data into pages
        bing = bing_data.get(site_slug, {})
        bing_query_clicks = {}
        for q in bing.get("queries", []):
            # Bing doesn't give per-page data in query stats, but we add site-level
            pass

        # Add combined traffic score
        bing_total = bing.get("totals", {}).get("clicks", 0)
        gsc_total = gsc.get("totals", {}).get("clicks", 0)
        page_count = len(pages)

        # Distribute bing clicks proportionally by GSC clicks
        for page in pages:
            gsc_page_clicks = page.get("gsc_clicks", 0)
            if gsc_total > 0 and bing_total > 0:
                bing_share = round(bing_total * (gsc_page_clicks / gsc_total))
            else:
                bing_share = 0
            page["bing_clicks"] = bing_share
            page["combined_clicks"] = gsc_page_clicks + bing_share

        mapped, skipped = map_urls_to_posts(pages, site_config, domain)

        # Classify traffic levels
        for page in mapped:
            page["traffic_level"] = classify_traffic(page)

        mapped_pages[site_slug] = mapped
        all_skipped[site_slug] = skipped
        print(f"{len(mapped)} mapped, {len(skipped)} skipped")

    total_mapped = sum(len(p) for p in mapped_pages.values())
    print(f"\n  Total pages mapped: {total_mapped}")

    if total_mapped == 0:
        print("[WARN] No pages mapped. Check GSC data and WP connectivity.")
        # Still generate report with search data
        report = generate_report(
            gsc_data, bing_data, mapped_pages, {}, {}, {}, timestamp
        )
        report_path = REPORTS_DIR / f"gsc_bing_audit_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[Saved] {report_path}")
        print_report_summary(report)
        return

    # =========================================================================
    # Phase 3a: Article Audits
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3a: ARTICLE AUDITS")
    print("=" * 80)

    audit_results = run_article_audits(mapped_pages)

    # =========================================================================
    # Phase 3b: ZIMM Visual Audits (optional)
    # =========================================================================
    visual_results = {}

    if not args.skip_visual:
        print("\n" + "=" * 80)
        print("PHASE 3b: ZIMM VISUAL AUDITS")
        print("=" * 80)
        visual_results = run_visual_audits(mapped_pages, sites_config)
    else:
        print("\n[SKIP] Visual audits skipped (--skip-visual)")

    # =========================================================================
    # Phase 4: Classify Fixes
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: CLASSIFY FIXES BY SAFETY LEVEL")
    print("=" * 80)

    fix_classifications = {}  # post_id -> (safe_fixes, needs_review)

    for site_slug, audits in audit_results.items():
        for audited_page in audits:
            pid = audited_page["post_id"]
            audit = audited_page.get("audit", {})
            traffic_level = audited_page.get("traffic_level", "LOW")

            if "error" in audit:
                continue

            safe, review = classify_fixes(audit, traffic_level)
            fix_classifications[pid] = (safe, review)

            if safe or review:
                print(
                    f"  {site_slug} post {pid} ({traffic_level}): "
                    f"{len(safe)} safe, {len(review)} need review"
                )

    # =========================================================================
    # Phase 5: Generate Report
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: GENERATE REPORT")
    print("=" * 80)

    report = generate_report(
        gsc_data, bing_data, mapped_pages, audit_results, visual_results,
        fix_classifications, timestamp
    )

    report_path = REPORTS_DIR / f"gsc_bing_audit_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[Saved] {report_path}")

    print_report_summary(report)

    # =========================================================================
    # Phase 6: Execute Safe Fixes (unless --audit-only)
    # =========================================================================
    if args.audit_only:
        print("\n[SKIP] Fix execution skipped (--audit-only mode)")
        print("=" * 80)
        print("PIPELINE COMPLETE (audit only)")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("PHASE 6: EXECUTE SAFE FIXES")
    print("=" * 80)

    fix_log = execute_safe_fixes(audit_results, fix_classifications)

    fix_log_path = REPORTS_DIR / f"fix_log_{timestamp}.json"
    with open(fix_log_path, "w") as f:
        json.dump(fix_log, f, indent=2, default=str)
    print(f"\n[Saved] {fix_log_path}")

    # Print fix summary
    applied = len(fix_log["fixes_applied"])
    skipped = len(fix_log["fixes_skipped"])
    failed = len(fix_log["failures"])
    print(f"\n--- FIX EXECUTION SUMMARY ---")
    print(f"  Fixes applied: {applied}")
    print(f"  Fixes skipped: {skipped}")
    print(f"  Failures:      {failed}")

    if failed:
        print("\n  Failures:")
        for f_item in fix_log["failures"][:10]:
            print(f"    {f_item.get('site', '?')} post {f_item.get('post_id', '?')}: {f_item.get('error', '?')}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
