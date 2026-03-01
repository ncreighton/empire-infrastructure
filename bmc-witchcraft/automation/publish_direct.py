"""Publish drafted products by navigating directly to their edit URLs.

Edit URLs discovered from the kebab dropdown menus:
  /extras/edit/{id}

For each: navigate to edit URL, fill confirmation message, click Publish.
"""
import io, sys, time, json
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.common.exceptions import (
    UnexpectedAlertPresentException,
    NoAlertPresentException,
)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SS_DIR = ROOT / "assets" / "screenshots"

PRODUCTS = [
    {
        "url": "https://studio.buymeacoffee.com/extras/edit/515493",
        "title": "Beginner's Spell Book",
        "confirmation": "Thank you for your purchase! Your Beginner's Spell Book is ready to download. Blessed be!",
    },
    {
        "url": "https://studio.buymeacoffee.com/extras/edit/515494",
        "title": "Moon Phase Journal",
        "confirmation": "Thank you! Your Moon Phase Journal is ready to download. May the moon guide your practice!",
    },
    {
        "url": "https://studio.buymeacoffee.com/extras/edit/515495",
        "title": "Grimoire Collection",
        "confirmation": "Thank you! Your Grimoire Collection is ready to download. May it serve your practice well!",
    },
    {
        "url": "https://studio.buymeacoffee.com/extras/edit/515496",
        "title": "Samhain Ritual Kit",
        "confirmation": "Thank you! Your Samhain Ritual Kit is ready. May your workings be powerful and protected!",
    },
    {
        "url": "https://studio.buymeacoffee.com/extras/edit/515497",
        "title": "Wheel of the Year Bundle",
        "confirmation": "Thank you! Your Wheel of the Year Bundle is ready. Honor the turning of the Wheel!",
    },
    {
        "url": "https://studio.buymeacoffee.com/extras/edit/515498",
        "title": "Complete Digital Grimoire Library",
        "confirmation": "Thank you! Your Complete Digital Grimoire Library is ready. The magick is within you!",
    },
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


def screenshot(driver, name):
    try:
        driver.save_screenshot(str(SS_DIR / name))
        print(f"   Screenshot: {name}")
    except Exception:
        pass


options = EdgeOptions()
options.debugger_address = "localhost:9222"
driver = webdriver.Edge(options=options)

print("=" * 60)
print("  PUBLISH DRAFTED PRODUCTS (DIRECT URL)")
print("=" * 60)

published = 0

for idx, product in enumerate(PRODUCTS):
    num = idx + 1
    print(f"\n{'='*60}")
    print(f"  PRODUCT {num}/6: {product['title']}")
    print(f"  URL: {product['url']}")
    print(f"{'='*60}")

    # Navigate to edit URL
    print("\n>> Loading edit page...")
    driver.get(product["url"])
    time.sleep(5)

    # Check if form loaded with data
    form_data = safe_execute(driver, """
        var titleEl = document.querySelector('#reward_title');
        var priceEl = document.querySelector('#reward_coffee_price');
        return {
            hasForm: !!(titleEl),
            title: titleEl ? titleEl.value : '',
            price: priceEl ? priceEl.value : ''
        };
    """)
    print(f"   Form data: title='{(form_data or {}).get('title', '')[:50]}' price=${(form_data or {}).get('price', '')}")
    screenshot(driver, f"direct{num}_01_edit.png")

    if not form_data or not form_data.get("title"):
        print("   ERROR: Form didn't load with draft data")
        screenshot(driver, f"direct{num}_error.png")
        continue

    # Scroll down to Success page section
    print(">> Scrolling to confirmation message...")
    safe_execute(driver, "window.scrollTo(0, 500)")
    time.sleep(1)

    # Find and fill the confirmation message field
    print(f">> Filling confirmation: {product['confirmation'][:50]}...")

    # First check the "Confirmation message" radio is selected
    safe_execute(driver, """
        // Ensure "Confirmation message" radio is selected (not "Redirect to URL")
        var radios = document.querySelectorAll('input[type="radio"]');
        for (var i = 0; i < radios.length; i++) {
            var r = radios[i];
            var label = r.closest('label') || r.parentElement;
            if (label && label.textContent.includes('Confirmation message') && !r.checked) {
                r.click();
            }
        }
    """)
    time.sleep(0.5)

    conf_result = safe_execute(driver, """
        var msg = arguments[0];

        // Look for ALL textareas and contenteditables
        var textareas = document.querySelectorAll('textarea');
        var editables = document.querySelectorAll('[contenteditable="true"]');

        // Try textarea with confirmation-related placeholder
        for (var i = 0; i < textareas.length; i++) {
            var ta = textareas[i];
            var ph = (ta.getAttribute('placeholder') || '').toLowerCase();
            var r = ta.getBoundingClientRect();
            if (r.width > 50 && (ph.includes('confirmation') || ph.includes('enter confirmation'))) {
                ta.focus();
                var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
                if (setter && setter.set) setter.set.call(ta, msg);
                else ta.value = msg;
                ta.dispatchEvent(new Event('input', {bubbles: true}));
                ta.dispatchEvent(new Event('change', {bubbles: true}));
                ta.dispatchEvent(new Event('blur', {bubbles: true}));
                return {filled: true, method: 'textarea-ph', value: ta.value.substring(0, 50)};
            }
        }

        // Try the second contenteditable (first is description, second should be confirmation)
        var editableIdx = 0;
        for (var i = 0; i < editables.length; i++) {
            var ed = editables[i];
            var r = ed.getBoundingClientRect();
            if (r.width > 200 && r.height > 20) {
                editableIdx++;
                if (editableIdx >= 2) {
                    // This should be the confirmation editor
                    ed.focus();
                    ed.textContent = msg;
                    ed.dispatchEvent(new Event('input', {bubbles: true}));
                    return {filled: true, method: 'contenteditable-2nd', idx: i};
                }
            }
        }

        // Fallback: any empty textarea
        for (var i = 0; i < textareas.length; i++) {
            var ta = textareas[i];
            var r = ta.getBoundingClientRect();
            if (r.width > 50 && r.height > 20 && (!ta.value || ta.value.trim() === '')) {
                ta.focus();
                var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
                if (setter && setter.set) setter.set.call(ta, msg);
                else ta.value = msg;
                ta.dispatchEvent(new Event('input', {bubbles: true}));
                ta.dispatchEvent(new Event('change', {bubbles: true}));
                return {filled: true, method: 'empty-textarea', idx: i, value: ta.value.substring(0, 50)};
            }
        }

        // Debug info
        var info = {filled: false, textareas: [], editables: []};
        for (var i = 0; i < textareas.length; i++) {
            var ta = textareas[i];
            var r = ta.getBoundingClientRect();
            info.textareas.push({
                ph: (ta.getAttribute('placeholder') || '').substring(0, 40),
                val: ta.value.substring(0, 30),
                visible: r.width > 0,
                y: Math.round(r.y)
            });
        }
        for (var i = 0; i < editables.length; i++) {
            var ed = editables[i];
            var r = ed.getBoundingClientRect();
            info.editables.push({
                text: ed.textContent.trim().substring(0, 30),
                visible: r.width > 0,
                y: Math.round(r.y),
                w: Math.round(r.width),
                h: Math.round(r.height)
            });
        }
        return info;
    """, product["confirmation"])
    print(f"   Confirmation: {json.dumps(conf_result, indent=2)}")

    # Ensure checkbox is checked
    safe_execute(driver, """
        var cb = document.querySelector('#extra-agree');
        if (cb && !cb.checked) cb.click();
    """)
    time.sleep(0.5)

    screenshot(driver, f"direct{num}_02_filled.png")

    # Click Publish
    print(">> Publishing...")
    safe_execute(driver, """
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            if (btns[i].textContent.trim() === 'Publish') {
                btns[i].scrollIntoView({behavior: 'instant', block: 'center'});
            }
        }
    """)
    time.sleep(0.5)

    pub = safe_execute(driver, """
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            if (btns[i].textContent.trim() === 'Publish') {
                btns[i].click();
                return {clicked: true};
            }
        }
        // Also try "Update" for already-saved drafts
        for (var i = 0; i < btns.length; i++) {
            var text = btns[i].textContent.trim();
            if (text === 'Update' || text === 'Save & Publish') {
                btns[i].click();
                return {clicked: true, text: text};
            }
        }
        return {clicked: false};
    """)
    print(f"   Publish click: {json.dumps(pub)}")
    time.sleep(6)

    # Check result
    screenshot(driver, f"direct{num}_03_result.png")
    post = safe_execute(driver, """
        return {
            url: window.location.href,
            hasForm: !!(document.querySelector('#reward_title')),
            errors: (function() {
                var errs = [];
                var els = document.querySelectorAll('p, span, div');
                for (var i = 0; i < els.length; i++) {
                    var t = els[i].textContent.trim();
                    if (t.startsWith('Please enter') && t.length < 100) errs.push(t);
                }
                return errs;
            })()
        };
    """)
    print(f"   Post-publish: {json.dumps(post, indent=2)}")

    if post and len(post.get("errors", [])) == 0 and not post.get("hasForm"):
        published += 1
        print(f"   PUBLISHED!")
    elif post and post.get("errors"):
        print(f"   Errors remaining: {post['errors']}")
    else:
        published += 1
        print(f"   Likely published")

    time.sleep(2)


# Final verification
print(f"\n{'='*60}")
print(f"  FINAL STATUS — {published}/6 published")
print(f"{'='*60}")

driver.get("https://studio.buymeacoffee.com/extras")
time.sleep(5)
safe_execute(driver, "window.scrollTo(0, 500)")
time.sleep(1)
screenshot(driver, "direct_final_dashboard.png")

status = safe_execute(driver, """
    var body = document.body.textContent;
    return {
        drafts: (body.match(/Drafted/g) || []).length,
        published: (body.match(/Published/g) || []).length
    };
""")
print(f"   Dashboard: {json.dumps(status)}")

driver.get("https://buymeacoffee.com/witchcraft/extras")
time.sleep(6)
screenshot(driver, "direct_final_public.png")

print("\nDone! Browser left open.")
