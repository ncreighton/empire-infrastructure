"""
ZimmWriter Controller API Server Test

Tests all REST API endpoints via requests library.
Requires the API server running: python -m uvicorn src.api:app --host 0.0.0.0 --port 8765

Tests:
1. Health & status endpoints
2. Connection endpoints
3. Configuration endpoints
4. Config window endpoints
5. Profile endpoints
6. Preset endpoints
7. Simulated n8n workflow

Usage:
    # First, start the server in another terminal:
    python -m uvicorn src.api:app --host 0.0.0.0 --port 8765

    # Then run tests:
    python scripts/test_api_server.py
    python scripts/test_api_server.py --base-url http://localhost:8765
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class APITester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.results = []

    def test(self, method: str, path: str, json_data: dict = None,
             expected_status: int = 200, description: str = "") -> dict:
        """Run a single API test."""
        url = f"{self.base_url}{path}"
        result = {
            "method": method.upper(),
            "path": path,
            "description": description,
            "status": "unknown",
            "http_status": None,
            "response": None,
            "error": None,
            "elapsed_ms": 0,
        }

        try:
            start = time.time()
            if method.upper() == "GET":
                resp = requests.get(url, timeout=30)
            elif method.upper() == "POST":
                resp = requests.post(url, json=json_data or {}, timeout=30)
            else:
                result["error"] = f"Unsupported method: {method}"
                result["status"] = "error"
                self.results.append(result)
                return result

            result["elapsed_ms"] = int((time.time() - start) * 1000)
            result["http_status"] = resp.status_code

            try:
                result["response"] = resp.json()
            except Exception:
                result["response"] = resp.text[:500]

            if resp.status_code == expected_status:
                result["status"] = "pass"
            else:
                result["status"] = "fail"
                result["error"] = f"Expected {expected_status}, got {resp.status_code}"

        except requests.ConnectionError:
            result["status"] = "error"
            result["error"] = "Connection refused - is the server running?"
        except requests.Timeout:
            result["status"] = "error"
            result["error"] = "Request timed out"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        self.results.append(result)
        return result

    def print_result(self, result: dict):
        """Print a single test result."""
        icons = {"pass": "OK", "fail": "XX", "error": "!!"}
        icon = icons.get(result["status"], "??")
        method = result["method"]
        path = result["path"]
        ms = result["elapsed_ms"]
        desc = result.get("description", "")

        print(f"  [{icon}] {method:4s} {path:35s} {ms:5d}ms  {desc}")
        if result["status"] != "pass":
            print(f"         {result.get('error', '')}")

    def summary(self) -> dict:
        """Print and return test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "pass")
        failed = sum(1 for r in self.results if r["status"] == "fail")
        errors = sum(1 for r in self.results if r["status"] == "error")

        print(f"\n  Total: {total}  Pass: {passed}  Fail: {failed}  Error: {errors}")
        return {"total": total, "passed": passed, "failed": failed, "errors": errors}


def run_all_tests(base_url: str):
    """Run the complete API test suite."""
    t = APITester(base_url)

    print("=" * 70)
    print("  ZIMMWRITER CONTROLLER API TEST SUITE")
    print(f"  Server: {base_url}")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── Health & Info ──
    print("\n  HEALTH & INFO")
    print("  " + "-" * 60)

    r = t.test("GET", "/", description="Root info")
    t.print_result(r)

    r = t.test("GET", "/health", description="Health check")
    t.print_result(r)

    r = t.test("GET", "/is-running", description="ZimmWriter running?")
    t.print_result(r)

    # ── Connection ──
    print("\n  CONNECTION")
    print("  " + "-" * 60)

    r = t.test("POST", "/connect", description="Connect to ZimmWriter")
    t.print_result(r)

    if r["status"] != "pass":
        print("\n  STOPPING: Cannot connect to ZimmWriter")
        print("  Make sure ZimmWriter is running and visible")
        return t

    r = t.test("GET", "/status", description="Full status dump")
    t.print_result(r)

    r = t.test("POST", "/bring-to-front", description="Bring to front")
    t.print_result(r)

    r = t.test("POST", "/screenshot", description="Take screenshot")
    t.print_result(r)

    # ── Discovery ──
    print("\n  ELEMENT DISCOVERY")
    print("  " + "-" * 60)

    r = t.test("GET", "/controls/buttons", description="List buttons")
    t.print_result(r)

    r = t.test("GET", "/controls/checkboxes", description="List checkboxes")
    t.print_result(r)

    r = t.test("GET", "/controls/dropdowns", description="List dropdowns")
    t.print_result(r)

    r = t.test("GET", "/controls/text-fields", description="List text fields")
    t.print_result(r)

    # ── Configuration ──
    print("\n  BULK WRITER CONFIGURATION")
    print("  " + "-" * 60)

    r = t.test("POST", "/configure", json_data={
        "section_length": "Medium",
        "voice": "Second Person",
    }, description="Set dropdowns")
    t.print_result(r)

    r = t.test("POST", "/checkboxes", json_data={
        "literary_devices": True,
        "lists": True,
        "nuke_ai_words": True,
    }, description="Set checkboxes")
    t.print_result(r)

    r = t.test("POST", "/feature-toggle", json_data={
        "feature": "serp_scraping",
        "enable": False,
    }, description="Toggle feature")
    t.print_result(r)

    # ── Titles ──
    print("\n  TITLES & CONTENT")
    print("  " + "-" * 60)

    r = t.test("POST", "/titles", json_data={
        "titles": ["API Test Article 1", "API Test Article 2"],
    }, description="Set titles")
    t.print_result(r)

    r = t.test("POST", "/clear", description="Clear all data")
    t.print_result(r)

    # ── Config Windows ──
    print("\n  CONFIG WINDOWS")
    print("  " + "-" * 60)

    r = t.test("POST", "/config/wordpress", json_data={
        "site_url": "https://smarthomewizards.com",
        "user_name": "SmartHomeGuru",
        "article_status": "draft",
    }, description="WordPress config")
    t.print_result(r)

    # Close any dialog that may have opened
    time.sleep(1)

    r = t.test("POST", "/config/serp-scraping", json_data={
        "enable": True,
    }, description="SERP config")
    t.print_result(r)
    time.sleep(1)

    # ── Presets ──
    print("\n  SITE PRESETS")
    print("  " + "-" * 60)

    r = t.test("GET", "/presets", description="List presets")
    t.print_result(r)
    if r["status"] == "pass" and r.get("response"):
        presets = r["response"].get("presets", {})
        print(f"         {len(presets)} presets available")

    r = t.test("GET", "/presets/smarthomewizards.com", description="Get preset")
    t.print_result(r)

    r = t.test("GET", "/presets/nonexistent.com", expected_status=404,
               description="Get missing preset (404)")
    t.print_result(r)

    r = t.test("POST", "/presets/smarthomewizards.com/apply",
               description="Apply preset")
    t.print_result(r)

    # ── Profiles ──
    print("\n  PROFILES")
    print("  " + "-" * 60)

    r = t.test("POST", "/profile/save", json_data={"name": "api-test-profile"},
               description="Save profile")
    t.print_result(r)

    r = t.test("POST", "/profile/load", json_data={"name": "api-test-profile"},
               description="Load profile")
    t.print_result(r)

    # ── n8n Workflow Simulation ──
    print("\n  N8N WORKFLOW SIMULATION")
    print("  " + "-" * 60)

    # Step 1: Connect
    r = t.test("POST", "/connect", description="n8n: Connect")
    t.print_result(r)

    # Step 2: Apply preset
    r = t.test("POST", "/presets/mythicalarchives.com/apply",
               description="n8n: Apply preset")
    t.print_result(r)

    # Step 3: Set titles
    r = t.test("POST", "/titles", json_data={
        "titles": ["The Legend of the Minotaur: A Complete Guide"],
    }, description="n8n: Set titles")
    t.print_result(r)

    # Step 4: Check status
    r = t.test("GET", "/status", description="n8n: Check status")
    t.print_result(r)

    # Don't actually start - just verify the pipeline works
    r = t.test("POST", "/clear", description="n8n: Cleanup")
    t.print_result(r)

    # ── Summary ──
    print("\n" + "=" * 70)
    summary = t.summary()
    print("=" * 70)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "api_test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "summary": summary,
            "results": t.results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    return t


def main():
    parser = argparse.ArgumentParser(description="Test ZimmWriter API server")
    parser.add_argument("--base-url", default="http://localhost:8765",
                        help="API server base URL")
    args = parser.parse_args()

    # Quick connectivity check
    try:
        resp = requests.get(f"{args.base_url}/health", timeout=5)
        if resp.status_code != 200:
            print(f"Server returned {resp.status_code}. Is it running?")
    except requests.ConnectionError:
        print(f"Cannot connect to {args.base_url}")
        print("Start the server first:")
        print("  python -m uvicorn src.api:app --host 0.0.0.0 --port 8765")
        sys.exit(1)

    tester = run_all_tests(args.base_url)

    # Exit code based on results
    summary = tester.summary()
    sys.exit(0 if summary["failed"] == 0 and summary["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
