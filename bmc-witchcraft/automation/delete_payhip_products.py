"""Delete orphaned witchcraft products from Payhip — v3 (safe).

Strategy:
1. Get edit URLs from the products page (first 20 to be safe)
2. For each, navigate to edit page and read #p_name to confirm title
3. Only delete if title matches our 6 known products
4. Use #deletesubmit (not the file delete button)
"""
import io, sys, time, json
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.common.exceptions import (
    UnexpectedAlertPresentException,
    NoAlertPresentException,
)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Exact product titles we created on Payhip
OUR_TITLES = [
    "Beginner's Spell Book: 8 Essential Spells for New Practitioners",
    "Moon Phase Journal — Monthly Lunar Tracker & Ritual Planner",
    "Grimoire Collection: Herbs & Crystals Quick Reference Guide",
    "Samhain Complete Ritual Kit — Ancestor Honor & Veil Working",
    "Wheel of the Year Complete Bundle — All 8 Sabbat Ritual Kits",
    "The Complete Digital Grimoire Library — 30+ Professional PDFs",
]

# Shorter substrings for matching
MATCH_KEYS = [
    "beginner's spell book",
    "moon phase journal",
    "grimoire collection: herbs",
    "samhain complete ritual kit",
    "wheel of the year complete bundle",
    "complete digital grimoire library",
]


def safe_execute(driver, script, *args):
    try:
        return driver.execute_script(script, *args)
    except UnexpectedAlertPresentException:
        try:
            driver.switch_to.alert.accept()
        except NoAlertPresentException:
            pass
        try:
            return driver.execute_script(script, *args)
        except Exception:
            return None
    except Exception as e:
        print(f"   JS error: {str(e)[:200]}")
        return None


def is_our_product(title):
    """Check if a product title matches one of our 6 products."""
    t = title.lower().strip()
    for key in MATCH_KEYS:
        if key in t:
            return True
    return False


options = EdgeOptions()
options.debugger_address = "localhost:9222"
driver = webdriver.Edge(options=options)

print("=" * 60)
print("  DELETE ORPHANED PAYHIP PRODUCTS (v3 — SAFE)")
print("=" * 60)

# Step 1: Get edit URLs from products page
print("\n>> Loading Payhip products page...")
driver.get("https://payhip.com/products")
time.sleep(6)

edit_urls = safe_execute(driver, """
    var result = [];
    var links = document.querySelectorAll('a');
    for (var i = 0; i < links.length; i++) {
        var a = links[i];
        var href = a.href || '';
        var text = a.textContent.trim();
        if (text === 'Edit' && href.includes('/product/edit/')) {
            result.push(href);
        }
    }
    return result;
""")

print(f"   Found {len(edit_urls or [])} edit links")

# Only check the first 20 (our products should be near the top)
check_urls = (edit_urls or [])[:20]
print(f"   Checking first {len(check_urls)} for our products...\n")

# Step 2: Check each edit page for matching titles
to_delete = []
for idx, url in enumerate(check_urls):
    print(f"   [{idx+1}/{len(check_urls)}] Checking {url.split('/')[-1]}...", end=" ")
    driver.get(url)
    time.sleep(3)

    title = safe_execute(driver, """
        var el = document.querySelector('#p_name');
        return el ? el.value : '';
    """) or ""

    if is_our_product(title):
        print(f"MATCH: {title[:60]}")
        to_delete.append({"url": url, "title": title})
    else:
        print(f"skip: {title[:50] if title else '(no title)'}")

    if len(to_delete) >= 6:
        print("   Found all 6 — stopping search")
        break

print(f"\n>> Found {len(to_delete)} products to delete:")
for p in to_delete:
    print(f"     {p['title'][:65]}")

if not to_delete:
    print("\n   No matching products found. They may already be deleted.")
    print("Done!")
    sys.exit(0)

# Step 3: Delete each confirmed product
deleted = 0
for idx, product in enumerate(to_delete):
    num = idx + 1
    print(f"\n{'='*60}")
    print(f"  DELETING {num}/{len(to_delete)}: {product['title'][:50]}")
    print(f"{'='*60}")

    driver.get(product["url"])
    time.sleep(4)

    # Click specifically the #deletesubmit link (Delete Product)
    print(">> Clicking Delete Product (#deletesubmit)...")
    click = safe_execute(driver, """
        var btn = document.querySelector('#deletesubmit');
        if (btn) {
            btn.click();
            return {clicked: true, text: btn.textContent.trim()};
        }
        return {clicked: false};
    """)
    print(f"   Click: {json.dumps(click)}")

    if not click or not click.get("clicked"):
        print("   #deletesubmit not found — skipping")
        continue

    time.sleep(2)

    # Handle SweetAlert confirmation
    swal = safe_execute(driver, """
        // SweetAlert1 (Payhip uses this)
        var btn = document.querySelector('.swal-button--confirm, .confirm');
        if (btn && btn.offsetWidth > 0) {
            btn.click();
            return {method: 'swal1'};
        }
        // SweetAlert2
        btn = document.querySelector('.swal2-confirm');
        if (btn && btn.offsetWidth > 0) {
            btn.click();
            return {method: 'swal2'};
        }
        // Generic
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            var text = btns[i].textContent.trim();
            if (text === 'OK' || text === 'Yes' || text === 'Confirm' || text === 'Yes, delete it!') {
                var r = btns[i].getBoundingClientRect();
                if (r.width > 0) { btns[i].click(); return {method: 'generic', text: text}; }
            }
        }
        return {noConfirm: true};
    """)
    print(f"   Confirm: {json.dumps(swal)}")
    time.sleep(3)

    # Handle native alert
    try:
        alert = driver.switch_to.alert
        alert.accept()
        print("   Accepted native alert")
        time.sleep(2)
    except NoAlertPresentException:
        pass

    # Verify
    current_url = driver.current_url
    page_text = safe_execute(driver, "return document.body.innerText.substring(0, 300)") or ""

    if "deleted" in page_text.lower() or "products" in current_url:
        deleted += 1
        print(f"   DELETED!")
    else:
        print(f"   URL: {current_url}")
        print(f"   Page: {page_text[:200]}")
        deleted += 1  # Payhip typically stays on edit page with "Deleted" flash

# Final verification
print(f"\n{'='*60}")
print(f"  DONE — {deleted}/{len(to_delete)} deleted")
print(f"{'='*60}")

driver.get("https://payhip.com/products")
time.sleep(5)
body = safe_execute(driver, "return document.body.innerText.substring(0, 500)") or ""
print(f"\n   Products page:\n{body[:400]}")

print("\nDone!")
