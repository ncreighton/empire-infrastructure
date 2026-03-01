"""Create 6 shop products via BMC's native Shop interface.

BMC has its own shop system (not Payhip). The form is at:
  studio.buymeacoffee.com/extras -> click "Digital product"

Form fields:
  - Name: #reward_title (text input)
  - Description: contenteditable rich text editor
  - Featured image: optional image upload
  - Price: #reward_coffee_price (number input, USD)
  - Success page: confirmation message
  - Upload file: file upload button
  - Categories: "+ Add new categories"
  - Checkbox: #extra-agree ("I created this...")
  - Publish button / Save as draft

Connects to Edge CDP on port 9222. Does NOT call driver.quit().
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
PLACEHOLDERS_DIR = ROOT / "assets" / "placeholders"

# ---------------------------------------------------------------------------
# Product data (from playbook/03-shop-products.md)
# Descriptions are shorter here — BMC's editor is simpler than Payhip's.
# ---------------------------------------------------------------------------
PRODUCTS = [
    {
        "title": "Beginner's Spell Book: 8 Essential Spells for New Practitioners",
        "price": "5.99",
        "filename": "beginners_spell_book.pdf",
        "description": (
            "Your first grimoire starts here.\n\n"
            "This beautifully designed spell book contains 8 foundational workings "
            "that every new practitioner should know — from a simple protection ward to "
            "a full moon manifestation spell. Each spell includes:\n\n"
            "- Complete ingredient list with substitutions\n"
            "- Step-by-step casting instructions\n"
            "- Optimal timing (moon phase, day, hour)\n"
            "- Safety notes and ethical considerations\n"
            "- Space to journal your results\n\n"
            "Format: Fillable PDF (works on any device)\n"
            "Pages: 24 pages, fully illustrated\n\n"
            "Perfect for beginners. No experience required — your intention is "
            "the strongest ingredient."
        ),
        "confirmation": "Thank you for your purchase! Your Beginner's Spell Book is ready to download. Blessed be!",
    },
    {
        "title": "Moon Phase Journal — Monthly Lunar Tracker & Ritual Planner",
        "price": "7.99",
        "filename": "moon_phase_journal.pdf",
        "description": (
            "Track your practice by the light of the moon.\n\n"
            "This interactive Moon Phase Journal gives you a full year of lunar "
            "tracking — with dedicated pages for each moon phase, space for ritual "
            "notes, intention setting, and reflection.\n\n"
            "What's inside:\n"
            "- New Moon: Intention-setting worksheets\n"
            "- Waxing Moon: Growth & action planning pages\n"
            "- Full Moon: Manifestation ritual templates\n"
            "- Waning Moon: Release & reflection prompts\n"
            "- Dark Moon: Rest & shadow work journaling\n\n"
            "Features: 13 lunar cycles, fillable PDF, moon sign reference guide, "
            "monthly herb & crystal correspondences.\n\n"
            "Format: Interactive fillable PDF — 56 pages"
        ),
        "confirmation": "Thank you! Your Moon Phase Journal is ready to download. May the moon guide your practice!",
    },
    {
        "title": "Grimoire Collection: Herbs & Crystals Quick Reference Guide",
        "price": "7.99",
        "filename": "grimoire_herbs_crystals.pdf",
        "description": (
            "Your essential correspondence reference — always within reach.\n\n"
            "This grimoire brings together two of the most-requested reference guides:\n\n"
            "HERB CORRESPONDENCES (49 herbs) — magickal properties, planetary ruler, "
            "element, best uses in spellwork, safety notes, and substitutions.\n\n"
            "CRYSTAL CORRESPONDENCES (40 crystals) — magickal properties, chakra alignment, "
            "cleansing methods, spell applications, and pairing suggestions.\n\n"
            "Bonus: Herb-crystal pairing chart, seasonal harvesting guide, "
            "crystal grid templates, quick-reference by intention.\n\n"
            "Format: PDF (printable + digital) — 38 pages, illustrated"
        ),
        "confirmation": "Thank you! Your Grimoire Collection is ready to download. May it serve your practice well!",
    },
    {
        "title": "Samhain Complete Ritual Kit — Ancestor Honor & Veil Working",
        "price": "9.99",
        "filename": "samhain_ritual_kit.pdf",
        "description": (
            "The most sacred night of the Witch's Year — fully prepared for you.\n\n"
            "This comprehensive Samhain ritual kit includes:\n\n"
            "- Complete Samhain Ritual — full script with opening, invocation, working, and closing\n"
            "- Ancestor Honor Ceremony — step-by-step with altar setup guide\n"
            "- Samhain Divination Spread — custom 7-card tarot spread for veil-thin energy\n"
            "- Reflection Journal Pages — guided prompts for processing and integrating\n\n"
            "Also includes: ingredient shopping list, timing guide, Samhain correspondences, "
            "and protective ward instructions.\n\n"
            "Format: PDF bundle — 32 pages across 4 documents"
        ),
        "confirmation": "Thank you! Your Samhain Ritual Kit is ready. May your workings be powerful and protected!",
    },
    {
        "title": "Wheel of the Year Complete Bundle — All 8 Sabbat Ritual Kits",
        "price": "39.99",
        "filename": "wheel_of_the_year_bundle.pdf",
        "description": (
            "A full year of sabbat celebrations — every ritual, every correspondence, "
            "every working.\n\n"
            "Complete ritual kits for all 8 sabbats: Imbolc, Ostara, Beltane, Litha, "
            "Lughnasadh, Mabon, Samhain, and Yule.\n\n"
            "Each kit includes: complete ritual script, altar setup guide, "
            "sabbat-specific divination spread, ingredient list with substitutions, "
            "timing guide and correspondences, guided reflection journal pages.\n\n"
            "8 complete kits — over 240 pages of professional-grade ritual content.\n\n"
            "Buying individually: $79.92 — Bundle price: $39.99 (50% savings)\n\n"
            "Format: PDF bundle (8 separate documents + master index)"
        ),
        "confirmation": "Thank you! Your Wheel of the Year Bundle is ready. Honor the turning of the Wheel!",
    },
    {
        "title": "The Complete Digital Grimoire Library — 30+ Professional PDFs",
        "price": "59.99",
        "filename": "complete_digital_grimoire_library.pdf",
        "description": (
            "Every grimoire. Every guide. Every workbook. One sacred collection.\n\n"
            "The complete VelvetVeil digital library — over 30 professional-grade "
            "witchcraft PDFs. Normally valued at $390+, this BMC-exclusive bundle "
            "gives you everything for $59.99.\n\n"
            "Includes: 5 Spell Books, 8 Sabbat Ritual Kits, 13 Moon Phase Journals, "
            "Grimoire Reference Pages (herbs, crystals, tarot, spell templates, deities), "
            "Planners & Trackers.\n\n"
            "This bundle is only available here on Buy Me a Coffee — exclusive to "
            "supporters of this community.\n\n"
            "Format: PDF bundle (30+ files) — 800+ total pages"
        ),
        "confirmation": "Thank you! Your Complete Digital Grimoire Library is ready. The magick is within you!",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def wait_for_load(driver, timeout=10):
    for _ in range(timeout * 2):
        state = safe_execute(driver, "return document.readyState")
        if state == "complete":
            return True
        time.sleep(0.5)
    return False


def fill_react_input(driver, selector, value):
    """Fill an input using React-compatible value setter + native events."""
    result = safe_execute(driver, """
        var el = document.querySelector(arguments[0]);
        if (!el) return {found: false, selector: arguments[0]};

        // Try React setter first
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value');
        if (nativeInputValueSetter && nativeInputValueSetter.set) {
            nativeInputValueSetter.set.call(el, arguments[1]);
        } else {
            el.value = arguments[1];
        }

        el.dispatchEvent(new Event('input', {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));
        el.dispatchEvent(new Event('blur', {bubbles: true}));
        return {found: true, value: el.value};
    """, selector, value)
    return result


def fill_description(driver, text):
    """Fill the BMC rich text editor (contenteditable div in description area)."""
    # The description editor is a contenteditable div
    # Convert newlines to <br> for HTML, preserve paragraph structure
    html_text = text.replace("\n\n", "</p><p>").replace("\n", "<br>")
    html_text = "<p>" + html_text + "</p>"

    result = safe_execute(driver, """
        // Find the description editor — it's a contenteditable div
        // Look for one that has a toolbar nearby (B, I, U buttons)
        var editables = document.querySelectorAll('[contenteditable="true"]');
        for (var i = 0; i < editables.length; i++) {
            var ed = editables[i];
            var r = ed.getBoundingClientRect();
            // The description editor should be visible and reasonably sized
            if (r.width > 200 && r.height > 50 && r.y > 100) {
                ed.focus();
                ed.innerHTML = arguments[0];
                ed.dispatchEvent(new Event('input', {bubbles: true}));
                ed.dispatchEvent(new Event('change', {bubbles: true}));
                return {found: true, index: i, width: Math.round(r.width), height: Math.round(r.height)};
            }
        }
        return {found: false, count: editables.length};
    """, html_text)
    return result


def upload_file_bmc(driver, filepath):
    """Upload a file via BMC's file upload button.

    BMC uses a hidden <input type="file"> triggered by the "Upload file" button.
    """
    abs_path = str(Path(filepath).resolve())

    # Find and make the file input visible
    result = safe_execute(driver, """
        // BMC may have a hidden file input near the "Upload file" button
        var fileInputs = document.querySelectorAll('input[type="file"]');
        for (var i = 0; i < fileInputs.length; i++) {
            var fi = fileInputs[i];
            // Make it visible for send_keys
            fi.style.opacity = '1';
            fi.style.position = 'relative';
            fi.style.width = '200px';
            fi.style.height = '50px';
            fi.style.zIndex = '9999';
            fi.style.display = 'block';
        }
        return {count: fileInputs.length};
    """)
    print(f"   File inputs found: {json.dumps(result)}")

    if not result or result.get("count", 0) == 0:
        print("   WARNING: No file inputs found")
        return False

    try:
        # Try to find the file input and send keys
        file_inputs = driver.find_elements("css selector", "input[type='file']")
        for fi in file_inputs:
            try:
                fi.send_keys(abs_path)
                print(f"   Sent file: {abs_path}")
                time.sleep(3)
                return True
            except Exception:
                continue
        print("   WARNING: Could not send file to any input")
        return False
    except Exception as e:
        print(f"   File upload error: {str(e)[:150]}")
        return False


def check_agree_checkbox(driver):
    """Check the 'I created this' agreement checkbox."""
    result = safe_execute(driver, """
        var cb = document.querySelector('#extra-agree');
        if (!cb) return {found: false};
        if (!cb.checked) {
            cb.click();
            // Also try the React way
            var setter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'checked');
            if (setter && setter.set) {
                setter.set.call(cb, true);
            }
            cb.dispatchEvent(new Event('change', {bubbles: true}));
            cb.dispatchEvent(new Event('input', {bubbles: true}));
            cb.dispatchEvent(new Event('click', {bubbles: true}));
        }
        return {found: true, checked: cb.checked};
    """)
    return result


def click_publish(driver):
    """Click the Publish button."""
    result = safe_execute(driver, """
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            var btn = btns[i];
            var text = btn.textContent.trim();
            if (text === 'Publish') {
                btn.scrollIntoView({behavior: 'instant', block: 'center'});
                return {found: true, text: text, disabled: btn.disabled};
            }
        }
        return {found: false};
    """)

    if result and result.get("found") and not result.get("disabled"):
        time.sleep(0.5)
        safe_execute(driver, """
            var btns = document.querySelectorAll('button');
            for (var i = 0; i < btns.length; i++) {
                if (btns[i].textContent.trim() === 'Publish') {
                    btns[i].click();
                    return true;
                }
            }
        """)
    return result


def click_save_draft(driver):
    """Click Save as draft as fallback."""
    result = safe_execute(driver, """
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            var text = btns[i].textContent.trim();
            if (text.includes('Save as draft') || text.includes('Draft')) {
                btns[i].click();
                return {clicked: true, text: text};
            }
        }
        return {clicked: false};
    """)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 60)
print("  BMC NATIVE SHOP — CREATE 6 PRODUCTS")
print("=" * 60)

options = EdgeOptions()
options.debugger_address = "localhost:9222"
driver = webdriver.Edge(options=options)

print(f"\nConnected. Current URL: {driver.current_url}")

# Verify placeholders exist
for product in PRODUCTS:
    pdf_path = PLACEHOLDERS_DIR / product["filename"]
    if not pdf_path.exists():
        print(f"ERROR: Missing PDF: {pdf_path}")
        sys.exit(1)
print("All placeholder PDFs verified.\n")

created_count = 0

for idx, product in enumerate(PRODUCTS):
    num = idx + 1
    print(f"\n{'='*60}")
    print(f"  PRODUCT {num}/6: {product['title'][:55]}...")
    print(f"  Price: ${product['price']}")
    print(f"{'='*60}")

    # Step 1: Navigate to extras and click "Digital product"
    print("\n>> Navigating to Shop page...")
    driver.get("https://studio.buymeacoffee.com/extras")
    time.sleep(5)
    wait_for_load(driver)

    print(">> Clicking 'Digital product'...")
    click_result = safe_execute(driver, """
        var els = document.querySelectorAll('div, button, a, span');
        for (var i = 0; i < els.length; i++) {
            var el = els[i];
            var text = el.textContent.trim();
            var r = el.getBoundingClientRect();
            if (r.width > 50 && r.height > 50 && text === 'Digital product') {
                el.click();
                return {clicked: true};
            }
        }
        return {clicked: false};
    """)
    print(f"   Click result: {json.dumps(click_result)}")
    time.sleep(4)
    screenshot(driver, f"bmc_shop{num}_01_form.png")

    # Verify form loaded
    has_form = safe_execute(driver, """
        return !!(document.querySelector('#reward_title') || document.querySelector('[placeholder*="offering"]'));
    """)
    if not has_form:
        print("   ERROR: Form not found. Retrying...")
        time.sleep(3)
        has_form = safe_execute(driver, """
            return !!(document.querySelector('#reward_title') || document.querySelector('[placeholder*="offering"]'));
        """)
        if not has_form:
            print("   ERROR: Still no form. Skipping product.")
            screenshot(driver, f"bmc_shop{num}_error.png")
            continue

    # Step 2: Fill name
    print(f"\n>> Filling name: {product['title'][:50]}...")
    name_result = fill_react_input(driver, "#reward_title", product["title"])
    print(f"   Name: {json.dumps(name_result)}")
    time.sleep(0.5)

    # Step 3: Fill description
    print("\n>> Filling description...")
    desc_result = fill_description(driver, product["description"])
    print(f"   Description: {json.dumps(desc_result)}")
    time.sleep(0.5)

    # Step 4: Fill price — scroll to it first
    print(f"\n>> Setting price: ${product['price']}...")
    safe_execute(driver, """
        var el = document.querySelector('#reward_coffee_price');
        if (el) el.scrollIntoView({behavior: 'instant', block: 'center'});
    """)
    time.sleep(0.5)

    # Clear and set price
    safe_execute(driver, """
        var el = document.querySelector('#reward_coffee_price');
        if (el) { el.focus(); el.select(); }
    """)
    time.sleep(0.3)
    price_result = fill_react_input(driver, "#reward_coffee_price", product["price"])
    print(f"   Price: {json.dumps(price_result)}")
    time.sleep(0.5)

    screenshot(driver, f"bmc_shop{num}_02_filled.png")

    # Step 5: Upload file
    print("\n>> Uploading file...")
    # First click the "Upload file" button to trigger the file input
    safe_execute(driver, """
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            var text = btns[i].textContent.trim();
            if (text.includes('Upload file')) {
                // Don't click — just scroll to it so we know where we are
                btns[i].scrollIntoView({behavior: 'instant', block: 'center'});
                return {found: true};
            }
        }
        return {found: false};
    """)
    time.sleep(1)

    pdf_path = PLACEHOLDERS_DIR / product["filename"]
    upload_ok = upload_file_bmc(driver, pdf_path)
    time.sleep(2)
    screenshot(driver, f"bmc_shop{num}_03_uploaded.png")

    # Step 6: Check the agreement checkbox
    print("\n>> Checking agreement checkbox...")
    safe_execute(driver, """
        var cb = document.querySelector('#extra-agree');
        if (cb) cb.scrollIntoView({behavior: 'instant', block: 'center'});
    """)
    time.sleep(0.5)
    agree_result = check_agree_checkbox(driver)
    print(f"   Agree: {json.dumps(agree_result)}")
    time.sleep(0.5)

    # Also try clicking the label if the checkbox didn't check
    if agree_result and not agree_result.get("checked"):
        safe_execute(driver, """
            var cb = document.querySelector('#extra-agree');
            if (cb) {
                var label = cb.closest('label') || cb.parentElement;
                if (label) label.click();
            }
        """)
        time.sleep(0.5)
        # Verify
        agree_result = safe_execute(driver, """
            var cb = document.querySelector('#extra-agree');
            return cb ? {checked: cb.checked} : {found: false};
        """)
        print(f"   Agree (retry): {json.dumps(agree_result)}")

    screenshot(driver, f"bmc_shop{num}_04_ready.png")

    # Step 7: Click Publish
    print("\n>> Publishing...")
    publish_result = click_publish(driver)
    print(f"   Publish button: {json.dumps(publish_result)}")
    time.sleep(6)

    # Check result
    screenshot(driver, f"bmc_shop{num}_05_after_publish.png")
    current_url = driver.current_url
    print(f"   URL after publish: {current_url}")

    # Check for success or errors
    post_state = safe_execute(driver, """
        var result = {url: window.location.href, errors: []};

        // Check for error messages
        var errors = document.querySelectorAll('.error, .alert-error, [class*="error"]');
        for (var i = 0; i < errors.length; i++) {
            var text = errors[i].textContent.trim();
            if (text && text.length < 200) result.errors.push(text);
        }

        // Check for success indicators (redirect, toast, etc.)
        result.hasForm = !!(document.querySelector('#reward_title'));
        result.title = document.title;

        return result;
    """)
    print(f"   Post-publish state: {json.dumps(post_state, indent=2)}")

    # If publish failed (maybe verification needed), try Save as draft
    if post_state and post_state.get("hasForm"):
        print("   Form still visible — trying Save as draft...")
        draft_result = click_save_draft(driver)
        print(f"   Draft: {json.dumps(draft_result)}")
        time.sleep(4)
        screenshot(driver, f"bmc_shop{num}_06_draft.png")
        created_count += 1
    else:
        created_count += 1
        print(f"   Product {num} published!")

    time.sleep(2)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  VERIFICATION — {created_count}/{len(PRODUCTS)} products attempted")
print(f"{'='*60}")

# Check BMC public shop
print("\n>> Checking public BMC shop...")
driver.get("https://buymeacoffee.com/witchcraft/extras")
time.sleep(6)
screenshot(driver, "bmc_shop_verify_01_public.png")

# Also check the extras dashboard for listed products
print(">> Checking shop dashboard...")
driver.get("https://studio.buymeacoffee.com/extras")
time.sleep(5)
screenshot(driver, "bmc_shop_verify_02_dashboard.png")

# Scroll down to see any products
safe_execute(driver, "window.scrollTo(0, 800)")
time.sleep(1)
screenshot(driver, "bmc_shop_verify_03_scrolled.png")

print(f"\n{'='*60}")
print(f"  DONE — {created_count}/{len(PRODUCTS)} products processed")
print(f"{'='*60}")
print("\nBrowser left open. Check screenshots in:", SS_DIR)
