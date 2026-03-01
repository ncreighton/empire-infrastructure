"""Explore BMC's native Shop and connect/create products."""
import io, sys, time, json
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.common.exceptions import UnexpectedAlertPresentException, NoAlertPresentException

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SS_DIR = ROOT / "assets" / "screenshots"


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


def screenshot(driver, name):
    try:
        driver.save_screenshot(str(SS_DIR / name))
        print(f"   Screenshot: {name}")
    except Exception:
        pass


def get_clickables(driver):
    return safe_execute(driver, """
        var result = [];
        var els = document.querySelectorAll('button, a, [role="button"], [role="tab"]');
        for (var i = 0; i < els.length; i++) {
            var el = els[i];
            var r = el.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) {
                var text = el.textContent.trim();
                if (text && text.length > 1 && text.length < 80) {
                    result.push({
                        text: text,
                        tag: el.tagName,
                        href: (el.href || '').substring(0, 120),
                        x: Math.round(r.x),
                        y: Math.round(r.y)
                    });
                }
            }
        }
        return result;
    """) or []


options = EdgeOptions()
options.debugger_address = "localhost:9222"
driver = webdriver.Edge(options=options)

# Step 1: Navigate to BMC extras (the Shop page)
print(">> Navigating to BMC Shop page...")
driver.get("https://studio.buymeacoffee.com/extras")
time.sleep(6)
screenshot(driver, "connect2_01_shop.png")
print(f"   URL: {driver.current_url}")

# Step 2: Scroll down to see if there are existing products below the "Add" section
print("\n>> Scrolling to see existing products...")
safe_execute(driver, "window.scrollTo(0, 600)")
time.sleep(2)
screenshot(driver, "connect2_02_below_add.png")

safe_execute(driver, "window.scrollTo(0, 1200)")
time.sleep(1)
screenshot(driver, "connect2_03_further.png")

# Step 3: Get all page content below the fold
print("\n>> Getting full page content...")
page_content = safe_execute(driver, """
    var result = [];
    var all = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, span, div, a, button, label');
    for (var i = 0; i < all.length; i++) {
        var el = all[i];
        var r = el.getBoundingClientRect();
        var text = el.textContent.trim();
        if (r.width > 30 && r.height > 5 && r.height < 200 && text && text.length > 2 && text.length < 120) {
            // Skip if text contains newlines (composite elements)
            if (text.indexOf('\\n') === -1) {
                result.push({
                    text: text,
                    tag: el.tagName,
                    y: Math.round(r.y + window.scrollY),
                    visible: r.y >= 0 && r.y < window.innerHeight
                });
            }
        }
    }
    // Deduplicate
    var seen = {};
    return result.filter(function(r) {
        if (seen[r.text]) return false;
        seen[r.text] = true;
        return true;
    });
""") or []

print(f"   Found {len(page_content)} text elements:")
for p in page_content[:30]:
    vis = "VIS" if p.get("visible") else "   ";
    print(f"   [{vis}] y={p['y']:4d} [{p['tag']:6s}] {p['text']}")

# Step 4: Check if Payhip products are already connected
# Look for the verification/completed-verification prompt
print("\n>> Checking for 'Complete verification' button...")
clickables = get_clickables(driver)
for c in clickables:
    print(f"   [{c['tag']}] {c['text']} -> {c.get('href', '')[:80]}")

# Step 5: Check the Integrations page for Payhip
print("\n>> Checking Integrations page...")
driver.get("https://studio.buymeacoffee.com/integrations")
time.sleep(5)
screenshot(driver, "connect2_04_integrations.png")
print(f"   URL: {driver.current_url}")

integrations = safe_execute(driver, """
    var result = [];
    var all = document.querySelectorAll('h1, h2, h3, h4, h5, p, span, div, a, button');
    for (var i = 0; i < all.length; i++) {
        var el = all[i];
        var r = el.getBoundingClientRect();
        var text = el.textContent.trim();
        if (r.width > 30 && r.height > 5 && r.height < 100 && text && text.length > 2 && text.length < 100) {
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
    });
""") or []

print(f"   Integrations page content:")
for item in integrations[:25]:
    print(f"     [{item['tag']:6s}] {item['text']}")

# Step 6: Go back to extras and try clicking "Digital product"
print("\n>> Going back to Shop and clicking 'Digital product'...")
driver.get("https://studio.buymeacoffee.com/extras")
time.sleep(5)

click_result = safe_execute(driver, """
    var els = document.querySelectorAll('div, button, a, span');
    for (var i = 0; i < els.length; i++) {
        var el = els[i];
        var text = el.textContent.trim();
        var r = el.getBoundingClientRect();
        if (r.width > 50 && r.height > 50 && text === 'Digital product') {
            el.click();
            return {clicked: true, text: text};
        }
    }
    return {clicked: false};
""")
print(f"   Click result: {json.dumps(click_result)}")
time.sleep(5)
screenshot(driver, "connect2_05_digital_product.png")
print(f"   URL after click: {driver.current_url}")

# Catalog the new page
print("\n>> Cataloging digital product form...")
form_elements = safe_execute(driver, """
    var result = {inputs: [], buttons: [], texts: []};

    var inputs = document.querySelectorAll('input, textarea, select');
    for (var i = 0; i < inputs.length; i++) {
        var inp = inputs[i];
        var r = inp.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
            result.inputs.push({
                type: inp.type,
                name: inp.name || '',
                id: inp.id || '',
                placeholder: (inp.placeholder || '').substring(0, 60),
                value: inp.value.substring(0, 30)
            });
        }
    }

    var btns = document.querySelectorAll('button, [role="button"], input[type="submit"]');
    for (var i = 0; i < btns.length; i++) {
        var b = btns[i];
        var r = b.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
            result.buttons.push({text: b.textContent.trim().substring(0, 60), tag: b.tagName});
        }
    }

    var headings = document.querySelectorAll('h1, h2, h3, h4, label, p');
    for (var i = 0; i < headings.length; i++) {
        var h = headings[i];
        var r = h.getBoundingClientRect();
        if (r.width > 30 && r.height > 0) {
            var text = h.textContent.trim();
            if (text && text.length > 2 && text.length < 100 && text.indexOf('\\n') === -1) {
                result.texts.push({text: text, tag: h.tagName, y: Math.round(r.y)});
            }
        }
    }
    return result;
""")

if form_elements:
    print("\n   Visible inputs:")
    for inp in form_elements.get("inputs", []):
        print(f"     <{inp['type']}> name={inp['name']} id={inp['id']} placeholder={inp['placeholder']}")

    print("\n   Buttons:")
    for b in form_elements.get("buttons", []):
        print(f"     [{b['tag']}] {b['text']}")

    print("\n   Labels/Text:")
    for t in form_elements.get("texts", [])[:15]:
        print(f"     [{t['tag']}@y={t['y']}] {t['text']}")

# Scroll to see full form
safe_execute(driver, "window.scrollTo(0, 400)")
time.sleep(1)
screenshot(driver, "connect2_06_form_scrolled.png")

safe_execute(driver, "window.scrollTo(0, 800)")
time.sleep(1)
screenshot(driver, "connect2_07_form_more.png")

print("\n>> Done exploring BMC shop.")
