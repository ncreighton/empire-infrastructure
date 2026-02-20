"""Re-enable XML-RPC blocking on all WordPress sites after ZimmWriter setup.
Reactivates the 'Disable XML-RPC' plugin and 'Block XML-RPC Completely' code snippets."""
import json
import requests
from requests.auth import HTTPBasicAuth

SITES_JSON = r"D:\Claude Code Projects\config\sites.json"

with open(SITES_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

sites = []
for site_id, config in data["sites"].items():
    wp = config.get("wordpress", {})
    domain = config.get("domain", "")
    if wp.get("user") and wp.get("app_password") and domain:
        sites.append({
            "domain": domain,
            "user": wp["user"],
            "app_password": wp["app_password"],
        })

print(f"Re-enabling XML-RPC security on {len(sites)} sites\n")

plugin_ok = 0
snippet_ok = 0

for site in sites:
    domain = site["domain"]
    auth = HTTPBasicAuth(site["user"], site["app_password"])
    base = f"https://{domain}/wp-json"
    print(f"[{domain}]", flush=True)

    # 1. Reactivate Disable XML-RPC plugin
    try:
        r = requests.put(
            f"{base}/wp/v2/plugins/disable-xml-rpc/disable-xml-rpc",
            json={"status": "active"},
            auth=auth,
            timeout=15
        )
        if r.status_code == 200:
            print(f"  Plugin: ACTIVATED", flush=True)
            plugin_ok += 1
        else:
            print(f"  Plugin: {r.status_code} {r.text[:80]}", flush=True)
    except Exception as e:
        print(f"  Plugin error: {e}", flush=True)

    # 2. Reactivate Block XML-RPC code snippets
    try:
        r = requests.get(f"{base}/code-snippets/v1/snippets", auth=auth, timeout=15)
        if r.status_code == 200:
            snippets = r.json()
            for s in snippets:
                name = s.get("name", "").lower()
                if "xml" in name and "rpc" in name and "block" in name:
                    sid = s["id"]
                    r2 = requests.put(
                        f"{base}/code-snippets/v1/snippets/{sid}",
                        json={"active": True},
                        auth=auth,
                        timeout=15
                    )
                    if r2.status_code == 200:
                        print(f"  Snippet '{s['name']}': ACTIVATED", flush=True)
                        snippet_ok += 1
                    else:
                        print(f"  Snippet error: {r2.status_code}", flush=True)
        else:
            print(f"  Snippets API: {r.status_code}", flush=True)
    except Exception as e:
        print(f"  Snippet error: {e}", flush=True)

print(f"\n{'='*50}")
print(f"Plugins reactivated: {plugin_ok}/{len(sites)}")
print(f"Snippets reactivated: {snippet_ok}")
print(f"{'='*50}")
