"""
Daily Analytics Sync to Supabase
Pulls from GSC, GA4, and Bing and syncs to Supabase
Designed to run as a scheduled task
"""

import json
import sys
import requests
import logging
from datetime import datetime
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Setup logging
LOG_FILE = Path(__file__).parent / 'logs' / 'analytics_sync.log'
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Supabase config
SUPABASE_URL = "https://pkiwwdrzsbfqhbmnmfnl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBraXd3ZHJ6c2JmcWhibW5tZm5sIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NjA4NDQ3NiwiZXhwIjoyMDgxNjYwNDc2fQ.MFdyX2mDK9YLesQJY4vWRoCNhYzj_oEAmuj2hDj6mWs"

# Credentials paths
GSC_SERVICE_ACCOUNT = Path(r"D:\Claude Code Projects\credentials\google\gsc-service-account.json")
GA4_OAUTH_CREDS = Path(r"D:\Claude Code Projects\credentials\google\oauth_credentials.json")
BING_API_KEY_FILE = Path(r"D:\Claude Code Projects\credentials\bing\api_key.json")

# Site mappings
SITE_IDS = {
    "smarthomewizards": "3effc9ba-7ebd-4716-b2c6-61fb77dbaab6",
    "witchcraftforbeginners": "b5819ecd-7124-4a39-96ad-427bd8a7b2c6",
    "mythicalarchives": "6c81db16-465f-4a27-8e1e-9a30b0f89589",
    "wealthfromai": "3c6799a0-726f-4633-9a60-609367a7ccde",
    "aidiscoverydigest": "128e74ec-564b-4691-998f-91e5ed7f0a2c",
    "aiinactionhub": "2b45f571-1d60-40db-938d-a7a3e96ba82c",
    "bulletjournals": "6650efda-870c-454d-8673-4699f2a7394c",
    "familyflourish": "30fa9833-ebd2-4100-b550-9a58911b7ae6",
    "pulsegearreviews": "e7861a71-a3b1-41cc-b6fc-8729b9dc95a1",
    "smarthomegearreviews": "b7927698-0832-4fc2-bdf6-4ebac06b9f58",
    "clearainews": "9ae3be4a-a5c0-4ffe-9d93-37ad841f210c",
    "celebrationseason": "3b045b90-36cf-4536-a009-eb4de397b208",
    "theconnectedhaven": "2838e7b5-425f-4a00-b328-a507a55290b3",
    "manifestandalign": "ae336d86-86d9-421a-98b7-4e62143ce71f",
    "wearablegearreviews": "ef610ae3-979f-45e0-a42a-fcae20f340c1",
}

GSC_SITES = {
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
    "familyflourish": "https://family-flourish.com/",
    "celebrationseason": "https://celebrationseason.net/",
}

GA4_PROPERTIES = {
    "508350677": "smarthomewizards",
    "435719684": "witchcraftforbeginners",
    "481551320": "mythicalarchives",
    "508216120": "wealthfromai",
    "515500773": "clearainews",
    "515483127": "aiinactionhub",
    "482136982": "bulletjournals",
    "482108306": "familyflourish",
    "515477643": "pulsegearreviews",
    "515497050": "smarthomegearreviews",
    "484742581": "manifestandalign",
    "484735168": "celebrationseason",
    "515476855": "wearablegearreviews",
}

DOMAIN_TO_SLUG = {
    "smarthomewizards.com": "smarthomewizards",
    "witchcraftforbeginners.com": "witchcraftforbeginners",
    "mythicalarchives.com": "mythicalarchives",
    "wealthfromai.com": "wealthfromai",
    "aidiscoverydigest.com": "aidiscoverydigest",
    "aiinactionhub.com": "aiinactionhub",
    "bulletjournals.net": "bulletjournals",
    "family-flourish.com": "familyflourish",
    "pulsegearreviews.com": "pulsegearreviews",
    "smarthomegearreviews.com": "smarthomegearreviews",
    "clearainews.com": "clearainews",
    "celebrationseason.net": "celebrationseason",
    "manifestandalign.com": "manifestandalign",
    "wearablegearreviews.com": "wearablegearreviews",
}


def supabase_upsert(table: str, data: list):
    """Upsert data to Supabase"""
    if not data:
        return 0

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal"
    }

    today = datetime.now().strftime("%Y-%m-%d")

    # For GSC, delete today's data first then insert (different constraint structure)
    if table == "gsc_performance":
        # Delete existing records for today
        delete_url = f"{SUPABASE_URL}/rest/v1/{table}?date=eq.{today}"
        requests.delete(delete_url, headers=headers)
        # Insert fresh
        url = f"{SUPABASE_URL}/rest/v1/{table}"
    else:
        url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict=site_id,date"

    r = requests.post(url, headers=headers, json=data)

    if r.status_code in [200, 201, 204]:
        return len(data)
    else:
        log.error(f"Supabase {table} error: {r.status_code} - {r.text[:200]}")
        return 0


def sync_gsc():
    """Sync GSC data using OAuth (same as GA4)"""
    log.info("Starting GSC sync...")

    if not GA4_OAUTH_CREDS.exists():
        log.error(f"OAuth creds not found: {GA4_OAUTH_CREDS}")
        return 0

    try:
        with open(GA4_OAUTH_CREDS) as f:
            creds = json.load(f)

        # Refresh OAuth token
        token_response = requests.post('https://oauth2.googleapis.com/token', data={
            'client_id': creds['client_id'],
            'client_secret': creds['client_secret'],
            'refresh_token': creds['refresh_token'],
            'grant_type': 'refresh_token'
        })

        if token_response.status_code != 200:
            log.error(f"GSC token refresh failed: {token_response.text[:100]}")
            return 0

        access_token = token_response.json()['access_token']
    except Exception as e:
        log.error(f"GSC auth failed: {e}")
        return 0

    from datetime import timedelta
    end_date = datetime.now() - timedelta(days=3)
    start_date = end_date - timedelta(days=28)
    today = datetime.now().strftime("%Y-%m-%d")

    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}

    records = []
    for slug, site_url in GSC_SITES.items():
        site_id = SITE_IDS.get(slug)
        if not site_id:
            continue

        try:
            import urllib.parse
            encoded_url = urllib.parse.quote(site_url, safe='')
            api_url = f'https://searchconsole.googleapis.com/webmasters/v3/sites/{encoded_url}/searchAnalytics/query'

            response = requests.post(api_url, headers=headers, json={
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'dimensions': ['query'],
                'rowLimit': 1000
            })

            if response.status_code != 200:
                log.warning(f"GSC {slug} failed: {response.status_code}")
                continue

            data = response.json()
            rows = data.get('rows', [])
            total_clicks = sum(r.get('clicks', 0) for r in rows)
            total_impressions = sum(r.get('impressions', 0) for r in rows)
            avg_position = sum(r.get('position', 0) for r in rows) / len(rows) if rows else 0
            avg_ctr = sum(r.get('ctr', 0) for r in rows) / len(rows) if rows else 0

            records.append({
                "site_id": site_id,
                "date": today,
                "clicks": total_clicks,
                "impressions": total_impressions,
                "ctr": round(avg_ctr, 4),
                "position": round(avg_position, 1),
                "country": "ALL",
                "device": "ALL",
                "source": "daily_sync"
            })
            log.info(f"GSC {slug}: {total_clicks} clicks, {total_impressions} impr")

        except Exception as e:
            log.warning(f"GSC {slug} failed: {str(e)[:50]}")

    synced = supabase_upsert("gsc_performance", records)
    log.info(f"GSC synced: {synced} sites")
    return synced


def sync_ga4():
    """Sync GA4 data using OAuth"""
    log.info("Starting GA4 sync...")

    if not GA4_OAUTH_CREDS.exists():
        log.error(f"GA4 OAuth creds not found: {GA4_OAUTH_CREDS}")
        return 0

    with open(GA4_OAUTH_CREDS) as f:
        creds = json.load(f)

    # Refresh token
    token_response = requests.post('https://oauth2.googleapis.com/token', data={
        'client_id': creds['client_id'],
        'client_secret': creds['client_secret'],
        'refresh_token': creds['refresh_token'],
        'grant_type': 'refresh_token'
    })

    if token_response.status_code != 200:
        log.error(f"GA4 token refresh failed: {token_response.text[:100]}")
        return 0

    access_token = token_response.json()['access_token']
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}

    # Get accounts and properties
    accounts_resp = requests.get(
        'https://analyticsadmin.googleapis.com/v1beta/accounts',
        headers=headers
    )

    if accounts_resp.status_code != 200:
        log.error(f"GA4 accounts failed: {accounts_resp.text[:100]}")
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    from datetime import timedelta
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    records = []
    for prop_id, slug in GA4_PROPERTIES.items():
        site_id = SITE_IDS.get(slug)
        if not site_id:
            continue

        try:
            report_resp = requests.post(
                f'https://analyticsdata.googleapis.com/v1beta/properties/{prop_id}:runReport',
                headers=headers,
                json={
                    'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
                    'metrics': [
                        {'name': 'sessions'},
                        {'name': 'totalUsers'},
                        {'name': 'screenPageViews'},
                        {'name': 'bounceRate'}
                    ]
                }
            )

            if report_resp.status_code == 200:
                data = report_resp.json()
                if 'rows' in data and data['rows']:
                    values = data['rows'][0]['metricValues']
                    records.append({
                        "site_id": site_id,
                        "date": today,
                        "sessions": int(float(values[0]['value'])),
                        "users": int(float(values[1]['value'])),
                        "pageviews": int(float(values[2]['value'])),
                        "bounce_rate": round(float(values[3]['value']), 4),
                        "source": "daily_sync"
                    })
                    log.info(f"GA4 {slug}: {values[0]['value']} sessions")
        except Exception as e:
            log.warning(f"GA4 {slug} failed: {str(e)[:50]}")

    synced = supabase_upsert("ga4_performance", records)
    log.info(f"GA4 synced: {synced} sites")
    return synced


def sync_bing():
    """Sync Bing Webmaster data"""
    log.info("Starting Bing sync...")

    if not BING_API_KEY_FILE.exists():
        log.error(f"Bing API key not found: {BING_API_KEY_FILE}")
        return 0

    with open(BING_API_KEY_FILE) as f:
        api_key = json.load(f)['api_key']

    # Get sites
    sites_resp = requests.get(
        f'https://ssl.bing.com/webmaster/api.svc/json/GetUserSites?apikey={api_key}'
    )

    if sites_resp.status_code != 200:
        log.error(f"Bing sites failed: {sites_resp.text[:100]}")
        return 0

    sites = sites_resp.json().get('d', [])
    today = datetime.now().strftime("%Y-%m-%d")
    records = []

    for site in sites:
        site_url = site.get('Url', '')
        domain = site_url.replace('https://', '').replace('http://', '').rstrip('/')
        slug = DOMAIN_TO_SLUG.get(domain)
        site_id = SITE_IDS.get(slug) if slug else None

        if not site_id:
            continue

        try:
            stats_resp = requests.get(
                f'https://ssl.bing.com/webmaster/api.svc/json/GetQueryStats?apikey={api_key}&siteUrl={site_url}'
            )

            if stats_resp.status_code == 200:
                stats = stats_resp.json().get('d', [])
                total_clicks = sum(s.get('Clicks', 0) for s in stats)
                total_impressions = sum(s.get('Impressions', 0) for s in stats)

                records.append({
                    "site_id": site_id,
                    "date": today,
                    "clicks": total_clicks,
                    "impressions": total_impressions,
                    "source": "daily_sync"
                })
                log.info(f"Bing {slug}: {total_clicks} clicks")
        except Exception as e:
            log.warning(f"Bing {domain} failed: {str(e)[:50]}")

    synced = supabase_upsert("bing_performance", records)
    log.info(f"Bing synced: {synced} sites")
    return synced


def main():
    log.info("=" * 50)
    log.info("DAILY ANALYTICS SYNC STARTED")
    log.info("=" * 50)

    results = {
        'gsc': sync_gsc(),
        'ga4': sync_ga4(),
        'bing': sync_bing()
    }

    log.info("=" * 50)
    log.info(f"SYNC COMPLETE - GSC: {results['gsc']}, GA4: {results['ga4']}, Bing: {results['bing']}")
    log.info("=" * 50)

    return results


if __name__ == "__main__":
    main()
