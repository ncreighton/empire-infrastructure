"""Delete orphaned Payhip products.

Navigates to payhip.com/products, finds all product entries,
and deletes them one by one.
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


def safe_screenshot(driver, name):
    try:
        driver.save_screenshot(name)
    except TimeoutException:
        print(f"   Screenshot timed out (skipping)")
    except Exception:
        pass


options = EdgeOptions()
options.debugger_address = "localhost:9222"
driver = webdriver.Edge(options=options)

print("=" * 60)
print("  DELETE ORPHANED PAYHIP PRODUCTS")
print("=" * 60)

# Navigate to products page
print("\n>> Loading Payhip products page...")
driver.get("https://payhip.com/products")
time.sleep(6)
print(f"   URL: {driver.current_url}")

# Get all products
products = safe_execute(driver, """
    var result = [];
    // Payhip products page has a table or list of products
    // Look for product links/rows
    var links = document.querySelectorAll('a[href*="/product/"]');
    for (var i = 0; i < links.length; i++) {
        var a = links[i];
        var text = a.textContent.trim();
        var href = a.href;
        if (text.length > 5 && href.includes('/product/')) {
            result.push({text: text.substring(0, 80), href: href});
        }
    }

    // Also try table rows
    var rows = document.querySelectorAll('tr');
    for (var i = 0; i < rows.length; i++) {
        var row = rows[i];
        var link = row.querySelector('a[href*="/product/"]');
        if (link) {
            var text = row.textContent.trim().replace(/\\s+/g, ' ').substring(0, 120);
            result.push({text: text, href: link.href, isRow: true});
        }
    }

    // Deduplicate by href
    var seen = {};
    return result.filter(function(r) {
        if (seen[r.href]) return false;
        seen[r.href] = true;
        return true;
    });
""")

print(f"\n   Found {len(products or [])} products:")
for p in (products or []):
    print(f"     {p['text'][:70]} -> {p['href']}")

if not products:
    # Maybe page has different structure — dump what's there
    page_info = safe_execute(driver, """
        var result = [];
        var els = document.querySelectorAll('*');
        for (var i = 0; i < els.length; i++) {
            var el = els[i];
            var r = el.getBoundingClientRect();
            var text = el.textContent.trim();
            if (r.width > 100 && r.height > 10 && r.height < 80 && text.length > 5 && text.length < 100) {
                if (text.indexOf('\\n') === -1) {
                    result.push({text: text, tag: el.tagName, y: Math.round(r.y)});
                }
            }
        }
        var seen = {};
        return result.filter(function(r) {
            if (seen[r.text]) return false;
            seen[r.text] = true;
            return true;
        }).slice(0, 30);
    """)
    print("\n   Page content:")
    for p in (page_info or []):
        print(f"     [{p['tag']:5s}] {p['text']}")

    # Try getting all links
    all_links = safe_execute(driver, """
        var result = [];
        var links = document.querySelectorAll('a');
        for (var i = 0; i < links.length; i++) {
            var a = links[i];
            var r = a.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) {
                result.push({text: a.textContent.trim().substring(0, 60), href: (a.href || '').substring(0, 120)});
            }
        }
        var seen = {};
        return result.filter(function(r) {
            if (seen[r.href]) return false;
            seen[r.href] = true;
            return true;
        });
    """)
    print("\n   All visible links:")
    for l in (all_links or []):
        print(f"     {l['text']:30s} -> {l['href']}")

# Now delete each product
if products:
    deleted = 0
    for p in products:
        href = p["href"]
        title = p["text"]
        print(f"\n>> Deleting: {title[:60]}...")

        # Navigate to the product edit page
        # Payhip product URLs are like /product/edit/XXXXX
        edit_url = href.replace("/product/", "/product/edit/") if "/edit/" not in href else href
        print(f"   Edit URL: {edit_url}")
        driver.get(edit_url)
        time.sleep(4)

        # Look for delete button
        delete_btn = safe_execute(driver, """
            var btns = document.querySelectorAll('a, button');
            for (var i = 0; i < btns.length; i++) {
                var btn = btns[i];
                var text = btn.textContent.trim();
                if (text.toLowerCase().includes('delete')) {
                    return {
                        found: true,
                        text: text,
                        tag: btn.tagName,
                        href: (btn.href || '').substring(0, 100)
                    };
                }
            }
            return {found: false};
        """)
        print(f"   Delete button: {json.dumps(delete_btn)}")

        if delete_btn and delete_btn.get("found"):
            # Click the delete button/link
            safe_execute(driver, """
                var btns = document.querySelectorAll('a, button');
                for (var i = 0; i < btns.length; i++) {
                    var text = btns[i].textContent.trim();
                    if (text.toLowerCase().includes('delete')) {
                        btns[i].click();
                        return true;
                    }
                }
            """)
            time.sleep(3)

            # Handle confirmation (SweetAlert or native)
            try:
                alert = driver.switch_to.alert
                alert.accept()
                print("   Accepted native alert")
                time.sleep(2)
            except NoAlertPresentException:
                pass

            # SweetAlert confirm
            swal = safe_execute(driver, """
                var btn = document.querySelector('.swal2-confirm, .swal-button--confirm, .confirm');
                if (btn) { btn.click(); return {clicked: true}; }
                // Also try any visible "Yes" or "OK" or "Delete" button
                var btns = document.querySelectorAll('button');
                for (var i = 0; i < btns.length; i++) {
                    var text = btns[i].textContent.trim();
                    if (text === 'Yes' || text === 'OK' || text === 'Confirm' || text === 'Yes, delete it!') {
                        btns[i].click();
                        return {clicked: true, text: text};
                    }
                }
                return {clicked: false};
            """)
            print(f"   Confirm: {json.dumps(swal)}")
            time.sleep(3)

            deleted += 1
            print(f"   Deleted!")
        else:
            print("   No delete button found")

    print(f"\n>> Deleted {deleted}/{len(products)} products")

# Verify
print("\n>> Verifying...")
driver.get("https://payhip.com/products")
time.sleep(5)
remaining = safe_execute(driver, """
    var links = document.querySelectorAll('a[href*="/product/"]');
    var result = [];
    for (var i = 0; i < links.length; i++) {
        var text = links[i].textContent.trim();
        if (text.length > 5) result.push(text.substring(0, 60));
    }
    return result;
""")
print(f"   Remaining products: {json.dumps(remaining)}")

print("\nDone!")
