"""
Seed FORGE CODEX with real audit data by auditing recent posts across all sites.

Usage:
    python scripts/seed-forge.py
    python scripts/seed-forge.py --count 3 --vps-ip 217.216.84.245
    python scripts/seed-forge.py --dry-run
"""

import argparse
import sys
import time

import requests

DEFAULT_VPS_IP = "217.216.84.245"
AUDIT_PORT = 8001

SITES = [
    "smarthomewizards",
    "mythicalarchives",
    "bulletjournals",
    "witchcraftforbeginners",
    "wealthfromai",
    "aidiscoverydigest",
    "aiinactionhub",
    "pulsegearreviews",
    "wearablegearreviews",
    "smarthomegearreviews",
    "clearainews",
    "theconnectedhaven",
    "manifestandalign",
    "familyflourish",
]


def main():
    parser = argparse.ArgumentParser(description="Seed FORGE CODEX with real audit data")
    parser.add_argument("--count", type=int, default=2, help="Posts to audit per site (default: 2)")
    parser.add_argument("--vps-ip", default=DEFAULT_VPS_IP, help=f"VPS IP address (default: {DEFAULT_VPS_IP})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--sites", nargs="+", help="Only seed specific site(s)")
    args = parser.parse_args()

    base = f"http://{args.vps_ip}:{AUDIT_PORT}"
    sites = args.sites if args.sites else SITES
    total_audits = len(sites) * args.count

    print(f"FORGE Seed: {len(sites)} sites x {args.count} posts = {total_audits} audits")
    print(f"API: {base}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    # Check article-audit is reachable
    if not args.dry_run:
        try:
            resp = requests.get(f"{base}/", timeout=10)
            if resp.status_code != 200:
                print(f"ERROR: Article-audit returned {resp.status_code}")
                sys.exit(1)
            print(f"Article-audit: healthy\n")
        except requests.ConnectionError:
            print(f"ERROR: Cannot reach article-audit at {base}")
            sys.exit(1)

    audited = 0
    learned = 0
    errors = []
    site_scores = {}

    for i, site in enumerate(sites, 1):
        print(f"[{i}/{len(sites)}] {site}...")

        if args.dry_run:
            print(f"  POST {base}/audit/recent/{site}?count={args.count}&workflow_type=zimm_standard")
            print(f"  POST {base}/forge/learn")
            continue

        # Audit recent posts
        try:
            resp = requests.post(
                f"{base}/audit/recent/{site}",
                params={"count": args.count, "workflow_type": "zimm_standard"},
                timeout=120,  # Audits can take a while
            )
            resp.raise_for_status()
            audit_data = resp.json()

            summary = audit_data.get("summary", {})
            batch_size = summary.get("batch_size", audit_data.get("count", 0))
            avg_score = summary.get("average_score")
            audited += batch_size

            if avg_score is not None:
                site_scores[site] = {
                    "avg": avg_score,
                    "count": batch_size,
                    "issues": summary.get("total_issues", 0),
                }
                print(f"  Audited {batch_size} posts (avg score: {avg_score:.1f}, issues: {summary.get('total_issues', '?')})")
            else:
                print(f"  Audited {batch_size} posts")

        except requests.HTTPError as e:
            error_msg = ""
            if e.response is not None:
                try:
                    error_msg = e.response.json().get("detail", e.response.text[:200])
                except Exception:
                    error_msg = e.response.text[:200]
            errors.append(f"{site} audit: {e} — {error_msg}")
            print(f"  ERROR auditing: {e}")
            continue
        except requests.ConnectionError as e:
            errors.append(f"{site} audit: connection lost")
            print(f"  ERROR: connection lost")
            continue

        # Feed results into FORGE
        try:
            learn_resp = requests.post(
                f"{base}/forge/learn",
                json={"site": site, "results": audit_data},
                timeout=30,
            )
            learn_resp.raise_for_status()
            learn_data = learn_resp.json()
            total = learn_data.get("total_audits", "?")
            print(f"  FORGE learned (total: {total})")
            learned += 1
        except requests.HTTPError as e:
            errors.append(f"{site} forge/learn: {e}")
            print(f"  ERROR feeding FORGE: {e}")
        except requests.ConnectionError:
            errors.append(f"{site} forge/learn: connection lost")
            print(f"  ERROR: FORGE connection lost")

        # Brief pause between sites to avoid overwhelming the API
        if i < len(sites):
            time.sleep(2)

    # Summary
    print(f"\n{'='*50}")
    print(f"FORGE Seed Complete")
    print(f"{'='*50}")
    print(f"  Sites processed: {len(sites)}")
    print(f"  Posts audited:   {audited}")
    print(f"  FORGE learned:   {learned}")

    if site_scores:
        print(f"\nScores by site:")
        for site, data in sorted(site_scores.items(), key=lambda x: x[1]["avg"]):
            print(f"  {site:30s} avg={data['avg']:5.1f}  ({data['count']} posts)")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

    # Check FORGE health
    if not args.dry_run:
        print(f"\nFORGE health check:")
        try:
            resp = requests.get(f"{base}/forge/health", timeout=10)
            health = resp.json()
            print(f"  {health}")
        except Exception as e:
            print(f"  Could not check: {e}")


if __name__ == "__main__":
    main()
