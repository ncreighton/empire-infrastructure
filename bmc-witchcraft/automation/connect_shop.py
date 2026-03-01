"""Explore and connect the Payhip shop to BMC."""
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


options = EdgeOptions()
options.debugger_address = "localhost:9222"
driver = webdriver.Edge(options=options)

# Step 1: Navigate to BMC shop extras
print(">> Navigating to BMC extras/shop...")
driver.get("https://studio.buymeacoffee.com/extras/shop")
time.sleep(6)
print(f"   URL: {driver.current_url}")
print(f"   Title: {driver.title}")
screenshot(driver, "shop_connect_01_page.png")

# Step 2: Catalog all visible elements
print("\n>> Cataloging page elements...")
elements = safe_execute(driver, """
    var result = {buttons: [], links: [], texts: [], inputs: [], iframes: []};

    // Buttons
    var btns = document.querySelectorAll('button, [role="button"], input[type="submit"]');
    for (var i = 0; i < btns.length; i++) {
        var b = btns[i];
        var r = b.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
            result.buttons.push({
                text: b.textContent.trim().substring(0, 80),
                tag: b.tagName,
                classes: b.className.substring(0, 60)
            });
        }
    }

    // Links
    var links = document.querySelectorAll('a');
    for (var i = 0; i < links.length; i++) {
        var a = links[i];
        var r = a.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
            var text = a.textContent.trim();
            if (text && text.length < 80) {
                result.links.push({text: text, href: (a.href || '').substring(0, 100)});
            }
        }
    }

    // Text elements
    var blocks = document.querySelectorAll('h1, h2, h3, h4, h5, p, label');
    for (var i = 0; i < blocks.length; i++) {
        var b = blocks[i];
        var r = b.getBoundingClientRect();
        if (r.width > 50 && r.height > 0) {
            var text = b.textContent.trim();
            if (text && text.length > 2 && text.length < 200) {
                result.texts.push({text: text, tag: b.tagName, y: Math.round(r.y)});
            }
        }
    }

    // Inputs
    var inputs = document.querySelectorAll('input, select, textarea');
    for (var i = 0; i < inputs.length; i++) {
        var inp = inputs[i];
        var r = inp.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
            result.inputs.push({
                type: inp.type,
                name: inp.name,
                id: inp.id,
                value: inp.value.substring(0, 50),
                placeholder: (inp.placeholder || '').substring(0, 50)
            });
        }
    }

    // Iframes
    var iframes = document.querySelectorAll('iframe');
    for (var i = 0; i < iframes.length; i++) {
        result.iframes.push({src: (iframes[i].src || '').substring(0, 100)});
    }

    return result;
""")

if elements:
    print("\n   Buttons:")
    for b in elements.get("buttons", []):
        print(f"     [{b['tag']}] {b['text']}")

    print("\n   Links:")
    for l in elements.get("links", []):
        print(f"     {l['text']} -> {l['href']}")

    print("\n   Texts:")
    for t in elements.get("texts", [])[:20]:
        print(f"     [{t['tag']}@y={t['y']}] {t['text']}")

    print("\n   Inputs:")
    for inp in elements.get("inputs", []):
        print(f"     <{inp['type']}> name={inp['name']} id={inp['id']} val={inp['value']}")

    print("\n   Iframes:")
    for f in elements.get("iframes", []):
        print(f"     {f['src']}")
else:
    print("   No elements found")

# Step 3: Get the full page HTML structure
print("\n>> Getting page structure...")
structure = safe_execute(driver, """
    var main = document.querySelector('main, #app, #__next, .content, .main');
    if (!main) main = document.body;
    return main.innerHTML.substring(0, 5000);
""")
if structure:
    print(f"   HTML (first 3000 chars):\n{structure[:3000]}")

# Scroll and screenshot
print("\n>> Scrolling page...")
safe_execute(driver, "window.scrollTo(0, 300)")
time.sleep(1)
screenshot(driver, "shop_connect_02_scrolled.png")

safe_execute(driver, "window.scrollTo(0, 600)")
time.sleep(1)
screenshot(driver, "shop_connect_03_more.png")

safe_execute(driver, "window.scrollTo(0, 1200)")
time.sleep(1)
screenshot(driver, "shop_connect_04_bottom.png")

# Step 4: Check other potential pages
print("\n>> Checking extras page...")
driver.get("https://studio.buymeacoffee.com/extras")
time.sleep(5)
screenshot(driver, "shop_connect_05_extras_main.png")
print(f"   URL: {driver.current_url}")

extras_info = safe_execute(driver, """
    var result = [];
    var els = document.querySelectorAll('button, a, [role="button"]');
    for (var i = 0; i < els.length; i++) {
        var el = els[i];
        var r = el.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
            var text = el.textContent.trim();
            if (text && text.length > 2 && text.length < 80) {
                result.push({text: text, tag: el.tagName, href: (el.href || '').substring(0, 100)});
            }
        }
    }
    return result;
""")
print("\n   Extras page clickables:")
for e in (extras_info or []):
    print(f"     [{e['tag']}] {e['text']} -> {e.get('href', '')}")

print("\n>> Done exploring.")
