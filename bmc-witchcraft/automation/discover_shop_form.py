"""Discovery script: catalog the Payhip 'Add Digital Product' form fields.

Connects to Edge CDP (port 9222), navigates to the BMC shop/add-product page,
catalogs all form fields, buttons, and tabs, and takes a screenshot.
"""
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
        print(f"   JS error: {str(e)[:150]}")
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

# Step 1: Navigate to BMC shop extras page
print(">> Navigating to BMC shop page...")
driver.get("https://studio.buymeacoffee.com/extras/shop")
time.sleep(6)
screenshot(driver, "shop_discover_01_extras.png")

print(f"   Current URL: {driver.current_url}")
print(f"   Title: {driver.title}")

# Step 2: Look for "Add product" or similar button
print("\n>> Looking for add product button/link...")
links = safe_execute(driver, """
    const result = [];
    const els = document.querySelectorAll('a, button, [role="button"]');
    for (const el of els) {
        const r = el.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
            const text = el.textContent.trim();
            if (text && text.length < 80) {
                result.push({
                    text: text,
                    tag: el.tagName,
                    href: el.href || '',
                    x: Math.round(r.x),
                    y: Math.round(r.y)
                });
            }
        }
    }
    return result;
""") or []

for link in links:
    print(f"   [{link['tag']}] {link['text'][:60]} -> {link.get('href', '')[:80]}")

# Step 3: Try to get to the add product page
print("\n>> Navigating to Payhip add digital product page...")
driver.get("https://payhip.com/product/add/digital")
time.sleep(6)
screenshot(driver, "shop_discover_02_payhip_form.png")
print(f"   URL: {driver.current_url}")
print(f"   Title: {driver.title}")

# Step 4: Catalog all form fields
print("\n>> Cataloging form fields...")
fields = safe_execute(driver, """
    const result = [];
    // Inputs
    const inputs = document.querySelectorAll('input, textarea, select');
    for (const inp of inputs) {
        const r = inp.getBoundingClientRect();
        result.push({
            type: 'input',
            tag: inp.tagName,
            inputType: inp.type,
            name: inp.name,
            id: inp.id,
            placeholder: inp.placeholder,
            value: inp.value,
            visible: r.width > 0 && r.height > 0,
            classes: inp.className.substring(0, 80)
        });
    }
    // Contenteditable (Quill editor)
    const editables = document.querySelectorAll('[contenteditable="true"]');
    for (const ed of editables) {
        const r = ed.getBoundingClientRect();
        result.push({
            type: 'contenteditable',
            tag: ed.tagName,
            classes: ed.className.substring(0, 80),
            placeholder: ed.getAttribute('data-placeholder'),
            visible: r.width > 0 && r.height > 0,
            x: Math.round(r.x),
            y: Math.round(r.y),
            w: Math.round(r.width),
            h: Math.round(r.height)
        });
    }
    return result;
""") or []

print(f"\n   Found {len(fields)} form elements:")
for f in fields:
    vis = "VISIBLE" if f.get('visible') else "hidden"
    if f['type'] == 'input':
        print(f"   [{vis}] <{f['tag']} type={f['inputType']}> name={f['name']} id={f['id']} placeholder={f.get('placeholder','')}")
    else:
        print(f"   [{vis}] <{f['tag']} contenteditable> class={f['classes']} placeholder={f.get('placeholder','')}")

# Step 5: Catalog tabs
print("\n>> Cataloging tabs...")
tabs = safe_execute(driver, """
    const result = [];
    const tabLinks = document.querySelectorAll('[role="tab"], .category-and-tags-tab-link, [data-toggle="tab"]');
    for (const tab of tabLinks) {
        const r = tab.getBoundingClientRect();
        result.push({
            text: tab.textContent.trim(),
            href: tab.getAttribute('href'),
            visible: r.width > 0 && r.height > 0,
            classes: tab.className.substring(0, 60)
        });
    }
    return result;
""") or []

print(f"   Found {len(tabs)} tabs:")
for t in tabs:
    vis = "VISIBLE" if t.get('visible') else "hidden"
    print(f"   [{vis}] {t['text']} -> {t['href']}")

# Step 6: Check submit button
print("\n>> Checking submit button...")
submit = safe_execute(driver, """
    const btn = document.querySelector('#addsubmit');
    if (!btn) return {found: false};
    const r = btn.getBoundingClientRect();
    return {
        found: true,
        value: btn.value,
        visible: r.width > 0 && r.height > 0,
        x: Math.round(r.x),
        y: Math.round(r.y)
    };
""")
print(f"   Submit button: {json.dumps(submit, indent=2)}")

# Step 7: Check file upload mechanism
print("\n>> Checking file upload...")
upload = safe_execute(driver, """
    const fileInputs = document.querySelectorAll('input[type="file"]');
    const result = [];
    for (const fi of fileInputs) {
        const parent = fi.closest('.moxie-shim') || fi.parentElement;
        result.push({
            id: fi.id,
            multiple: fi.multiple,
            accept: fi.accept,
            parentId: parent ? parent.id : '',
            parentClass: parent ? parent.className.substring(0, 60) : ''
        });
    }
    // Also check for plupload
    const pluploadBtn = document.querySelector('#files');
    return {
        fileInputs: result,
        pluploadButton: pluploadBtn ? {
            id: pluploadBtn.id,
            text: pluploadBtn.textContent.trim(),
            visible: pluploadBtn.getBoundingClientRect().width > 0
        } : null
    };
""")
print(f"   File upload: {json.dumps(upload, indent=2)}")

# Scroll and take final screenshot
safe_execute(driver, "window.scrollTo(0, 0)")
time.sleep(1)
screenshot(driver, "shop_discover_03_form_top.png")

safe_execute(driver, "window.scrollTo(0, document.body.scrollHeight)")
time.sleep(1)
screenshot(driver, "shop_discover_04_form_bottom.png")

print("\n>> Discovery complete!")
print("   Check screenshots in:", SS_DIR)
