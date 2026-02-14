"""
Controller + Intelligence Integration Test
Tests the full stack: Controller -> Intelligence Hub -> FORGE -> AMPLIFY -> Vision -> Screenpipe
"""
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 65)
    print("  CONTROLLER + INTELLIGENCE INTEGRATION TEST")
    print("=" * 65)
    print()

    results = []

    # 1. Connect to ZimmWriter
    print("[1/7] Connecting to ZimmWriter...")
    try:
        from src.controller import ZimmWriterController
        ctrl = ZimmWriterController()
        ctrl.connect()
        connected = ctrl._connected
        print(f"  Connected: {connected}")
        if connected:
            title = ctrl.get_window_title()
            print(f"  Window: {title}")
            results.append(("Controller Connect", "PASS"))
        else:
            results.append(("Controller Connect", "FAIL - not connected"))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("Controller Connect", f"FAIL - {e}"))
        print("\n  Cannot proceed without controller connection.")
        print_results(results)
        return

    # 2. Initialize Intelligence Hub
    print("\n[2/7] Initializing Intelligence Hub...")
    try:
        from src.intelligence import IntelligenceHub
        hub = IntelligenceHub()
        vision_ok = hub.vision.is_available()
        sp_ok = hub.screenpipe.is_available()
        print(f"  Vision Service: {'available' if vision_ok else 'offline'}")
        print(f"  Screenpipe: {'available' if sp_ok else 'offline'}")
        results.append(("Intelligence Hub Init", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("Intelligence Hub Init", f"FAIL - {e}"))
        print_results(results)
        return

    # 3. FORGE Pre-job Analysis
    print("\n[3/7] Running FORGE pre-job analysis...")
    try:
        test_config = {
            "ai_model": "GPT 4o Mini",
            "section_length": "Standard",
            "number_of_sections": "5",
        }
        test_titles = ["Integration Test Article - Smart Home Automation Guide"]

        report = hub.pre_job(test_config, test_titles, "start_bulk_writer")
        job_id = report.get("job_id", "unknown")
        ready = report.get("ready", False)
        warnings = report.get("warnings", [])
        issues = report.get("forge_report", {}).get("config_issues", [])
        fixes = report.get("forge_report", {}).get("auto_fixes", [])

        print(f"  Job ID: {job_id}")
        print(f"  Ready: {ready}")
        print(f"  Config issues: {len(issues)}")
        print(f"  Auto-fixes applied: {len(fixes)}")
        print(f"  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"    - {w}")

        results.append(("FORGE Pre-job Analysis", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("FORGE Pre-job Analysis", f"FAIL - {e}"))

    # 4. Screenpipe State Check
    print("\n[4/7] Checking Screenpipe for ZimmWriter state...")
    try:
        state = hub.screenpipe.read_current_state()
        screen = state.get("current_screen", "unknown")
        has_errors = state.get("has_errors", False)
        ocr_snippets = state.get("recent_text", [])

        print(f"  Current screen: {screen}")
        print(f"  Has errors: {has_errors}")
        print(f"  OCR snippets: {len(ocr_snippets)}")
        if ocr_snippets:
            for s in ocr_snippets[:3]:
                text = s if isinstance(s, str) else str(s)
                print(f"    - {text[:80]}...")

        results.append(("Screenpipe State", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("Screenpipe State", f"FAIL - {e}"))

    # 5. Vision Verification
    print("\n[5/7] Vision verification of current screen...")
    try:
        if hub.vision.is_available():
            verify = hub.verify_screen("Bulk Writer main screen", controller=ctrl)
            matches = verify.get("matches", False)
            confidence = verify.get("confidence", 0)
            description = verify.get("description", "")

            print(f"  Matches expected: {matches}")
            print(f"  Confidence: {confidence}")
            print(f"  Description: {description[:100]}")
            results.append(("Vision Verification", "PASS"))
        else:
            print("  Vision Service offline - skipped")
            results.append(("Vision Verification", "SKIP - offline"))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("Vision Verification", f"FAIL - {e}"))

    # 6. Error Detection (combined Vision + Screenpipe)
    print("\n[6/7] Combined error detection...")
    try:
        errors = hub.detect_errors(controller=ctrl)
        has_errors = errors.get("has_errors", False)
        vision_errs = errors.get("vision_errors", {})
        sp_errs = errors.get("screenpipe_errors", [])

        print(f"  Has errors: {has_errors}")
        print(f"  Vision errors: {vision_errs.get('errors_found', 'N/A')}")
        print(f"  Screenpipe errors: {len(sp_errs)}")

        results.append(("Error Detection", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("Error Detection", f"FAIL - {e}"))

    # 7. Controller Status
    print("\n[7/7] Full controller status check...")
    try:
        status = ctrl.get_status()
        buttons = len(status.get("buttons", []))
        checkboxes = len(status.get("checkboxes", []))
        dropdowns = len(status.get("dropdowns", []))
        current_model = status.get("current_settings", {}).get("ai_model", "unknown")

        print(f"  Buttons: {buttons}")
        print(f"  Checkboxes: {checkboxes}")
        print(f"  Dropdowns: {dropdowns}")
        print(f"  Current AI model: {current_model}")

        results.append(("Controller Status", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("Controller Status", f"FAIL - {e}"))

    # Summary
    print_results(results)


def print_results(results):
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    passed = sum(1 for _, r in results if r == "PASS")
    failed = sum(1 for _, r in results if r.startswith("FAIL"))
    skipped = sum(1 for _, r in results if r.startswith("SKIP"))

    for name, result in results:
        status = "[OK]" if result == "PASS" else "[SKIP]" if result.startswith("SKIP") else "[FAIL]"
        print(f"  {status} {name}: {result}")

    print()
    print(f"  Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    if failed == 0:
        print("\n  ALL TESTS PASSED")
    else:
        print(f"\n  {failed} TEST(S) FAILED")


if __name__ == "__main__":
    main()
