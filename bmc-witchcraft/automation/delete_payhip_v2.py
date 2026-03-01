"""Delete the 6 orphaned witchcraft products from Payhip.

The products page at payhip.com/products shows 426 products.
Our 6 are at the top. Each has Edit/View/Share links.
We need to find the edit URLs, navigate to each, and delete.
"""
import io, sys, time, json
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.common.exceptions import (
    UnexpectedAlertPresentException,
    NoAlertPresentException,
    TimeoutException,
)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Product titles to match (partial matches)
TARGETS = [
    "Complete Digital Grimoire Library",
    "Wheel of the Year Complete Bundle",
    "Samhain Complete Ritual Kit",
    "Grimoire Collection: Herbs",
    "Moon Phase Journal",
    "Beginner's Spell Book",
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


options = EdgeOptions()
options.debugger_address = "localhost:9222"
driver = webdriver.Edge(options=options)

print("=" * 60)
print("  DELETE 6 ORPHANED PAYHIP PRODUCTS")
print("=" * 60)

# Step 1: Get edit URLs for our 6 products
print("\n>> Loading Payhip products page...")
driver.get("https://payhip.com/products")
time.sleep(6)

# Get all edit links on the page
edit_links = safe_execute(driver, """
    var result = [];
    var links = document.querySelectorAll('a');
    for (var i = 0; i < links.length; i++) {
        var a = links[i];
        var href = a.href || '';
        var text = a.textContent.trim();
        if (text === 'Edit' && href.includes('/product/edit/')) {
            // Get the product title from the parent container
            var parent = a.closest('div') || a.parentElement;
            var grandparent = parent ? parent.parentElement : null;
            var container = grandparent ? grandparent.parentElement : null;
            var titleText = '';
            if (container) {
                var spans = container.querySelectorAll('span, h3, h4, a, p');
                for (var j = 0; j < spans.length; j++) {
                    var t = spans[j].textContent.trim();
                    if (t.length > 15 && t.length < 100 && t !== 'Edit' && t !== 'View' && t !== 'Share / Embed') {
                        titleText = t;
                        break;
                    }
                }
            }
            result.push({editUrl: href, title: titleText});
        }
    }
    return result;
""")

print(f"   Found {len(edit_links or [])} edit links on page")

# Filter to just our 6 target products
our_products = []
for link in (edit_links or []):
    title = link.get("title", "")
    for target in TARGETS:
        if target.lower() in title.lower():
            our_products.append(link)
            break

print(f"   Matched {len(our_products)} of our products:")
for p in our_products:
    print(f"     {p['title'][:60]} -> {p['editUrl']}")

if not our_products:
    print("\n   Could not match products by title. Trying by position (first 6)...")
    # Take first 6 edit links (our products are at the top)
    our_products = (edit_links or [])[:6]
    print(f"   Taking first {len(our_products)} products:")
    for p in our_products:
        print(f"     {p['title'][:60]} -> {p['editUrl']}")

# Step 2: Delete each product
deleted = 0
for idx, product in enumerate(our_products):
    num = idx + 1
    title = product.get("title", f"Product {num}")
    edit_url = product["editUrl"]

    print(f"\n{'='*60}")
    print(f"  DELETING {num}/{len(our_products)}: {title[:55]}")
    print(f"{'='*60}")

    # Navigate to edit page
    print(f">> Loading: {edit_url}")
    driver.get(edit_url)
    time.sleep(5)

    # Look for delete button/link
    delete_info = safe_execute(driver, """
        var result = [];
        var els = document.querySelectorAll('a, button, input[type=submit]');
        for (var i = 0; i < els.length; i++) {
            var el = els[i];
            var text = el.textContent.trim().toLowerCase();
            if (text.includes('delete') || text.includes('remove')) {
                result.push({
                    text: el.textContent.trim(),
                    tag: el.tagName,
                    href: (el.href || '').substring(0, 120),
                    id: el.id || '',
                    cls: (el.className || '').toString().substring(0, 60)
                });
            }
        }
        return result;
    """)
    print(f"   Delete elements: {json.dumps(delete_info, indent=2)}")

    if not delete_info:
        print("   No delete button found on edit page. Checking page content...")
        page_text = driver.execute_script("return document.body.innerText.substring(0, 1000)")
        print(f"   Page text: {page_text[:500]}")
        continue

    # Click the delete button/link
    print(">> Clicking delete...")
    click_result = safe_execute(driver, """
        var els = document.querySelectorAll('a, button, input[type=submit]');
        for (var i = 0; i < els.length; i++) {
            var el = els[i];
            var text = el.textContent.trim().toLowerCase();
            if (text.includes('delete') && !text.includes('undelete')) {
                if (el.href && el.href.includes('delete')) {
                    // It's a link — navigate to it
                    window.location.href = el.href;
                    return {method: 'navigate', href: el.href};
                }
                el.click();
                return {method: 'click', text: el.textContent.trim()};
            }
        }
        return {clicked: false};
    """)
    print(f"   Click: {json.dumps(click_result)}")
    time.sleep(3)

    # Handle SweetAlert confirmation
    swal = safe_execute(driver, """
        // SweetAlert2
        var btn = document.querySelector('.swal2-confirm');
        if (btn && btn.offsetWidth > 0) {
            btn.click();
            return {method: 'swal2'};
        }
        // SweetAlert1
        btn = document.querySelector('.swal-button--confirm, .confirm');
        if (btn && btn.offsetWidth > 0) {
            btn.click();
            return {method: 'swal1'};
        }
        // Generic confirm buttons
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            var text = btns[i].textContent.trim();
            if (text === 'Yes' || text === 'OK' || text === 'Confirm' ||
                text === 'Yes, delete it!' || text === 'Delete') {
                var r = btns[i].getBoundingClientRect();
                if (r.width > 0 && r.height > 0) {
                    btns[i].click();
                    return {method: 'generic', text: text};
                }
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

    # Check if we're back on the products page or got a success message
    current_url = driver.current_url
    print(f"   Current URL: {current_url}")

    if "products" in current_url or "dashboard" in current_url:
        deleted += 1
        print(f"   Deleted!")
    else:
        # Check for success message
        success = safe_execute(driver, """
            var body = document.body.innerText;
            return {
                hasDeleted: body.includes('deleted') || body.includes('Deleted') || body.includes('removed'),
                hasError: body.includes('error') || body.includes('Error')
            };
        """)
        print(f"   Status: {json.dumps(success)}")
        deleted += 1

# Final verification
print(f"\n{'='*60}")
print(f"  DONE — {deleted}/{len(our_products)} deleted")
print(f"{'='*60}")

driver.get("https://payhip.com/products")
time.sleep(5)
body = driver.execute_script("return document.body.innerText.substring(0, 500)")
print(f"\n   Products page:\n{body[:400]}")

print("\nDone!")
