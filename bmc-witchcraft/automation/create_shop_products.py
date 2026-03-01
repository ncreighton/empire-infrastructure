"""Create 6 shop products on Payhip (BMC's shop provider) via Selenium.

Connects to Edge CDP on port 9222 (browser must already be running with
--remote-debugging-port=9222). For each product:
  1. Navigate to https://payhip.com/product/add/digital
  2. Upload the placeholder PDF via the hidden file input
  3. Fill title (#p_name), price (#p_price)
  4. Fill description via Quill editor (.ql-editor)
  5. Click the Category & Tags tab and attempt to add tags
  6. Click #addsubmit to submit
  7. Wait for confirmation, take screenshot, then proceed to next product

Does NOT call driver.quit() — keeps the browser session alive.
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
# ---------------------------------------------------------------------------
PRODUCTS = [
    {
        "title": "Beginner's Spell Book: 8 Essential Spells for New Practitioners",
        "price": "5.99",
        "filename": "beginners_spell_book.pdf",
        "tags": ["spells", "beginner", "grimoire", "witchcraft", "digital download"],
        "description": (
            "<p>Your first grimoire starts here.</p>"
            "<p>This beautifully designed spell book contains 8 foundational workings "
            "that every new practitioner should know — from a simple protection ward to "
            "a full moon manifestation spell. Each spell includes:</p>"
            "<p>✦ Complete ingredient list with substitutions<br>"
            "✦ Step-by-step casting instructions<br>"
            "✦ Optimal timing (moon phase, day, hour)<br>"
            "✦ Safety notes and ethical considerations<br>"
            "✦ Space to journal your results</p>"
            "<p>Spells included:<br>"
            "1. Circle of Light — Personal Protection<br>"
            "2. New Moon Intention Setting<br>"
            "3. Full Moon Manifestation<br>"
            "4. Herbal Clarity Tea Ritual<br>"
            "5. Candle Spell for Peace<br>"
            "6. Crystal Grid for Grounding<br>"
            "7. Banishing Negativity Bath<br>"
            "8. Gratitude &amp; Abundance Working</p>"
            "<p>Format: Fillable PDF (works on any device)<br>"
            "Pages: 24 pages, fully illustrated</p>"
            "<p>Perfect for beginners. No experience required — your intention is "
            "the strongest ingredient.</p>"
        ),
    },
    {
        "title": "Moon Phase Journal — Monthly Lunar Tracker & Ritual Planner",
        "price": "7.99",
        "filename": "moon_phase_journal.pdf",
        "tags": ["moon journal", "lunar tracker", "planner", "witchcraft", "moon phases"],
        "description": (
            "<p>Track your practice by the light of the moon.</p>"
            "<p>This interactive Moon Phase Journal gives you a full year of lunar "
            "tracking — with dedicated pages for each moon phase, space for ritual "
            "notes, intention setting, and reflection. Designed for practitioners who "
            "want to align their magick with the natural rhythms of the moon.</p>"
            "<p>What's inside:</p>"
            "<p>🌑 New Moon: Intention-setting worksheets<br>"
            "🌒 Waxing Moon: Growth &amp; action planning pages<br>"
            "🌕 Full Moon: Manifestation ritual templates<br>"
            "🌘 Waning Moon: Release &amp; reflection prompts<br>"
            "🌑 Dark Moon: Rest &amp; shadow work journaling</p>"
            "<p>Features:<br>"
            "✦ 13 lunar cycles (covers a full year)<br>"
            "✦ Fillable PDF — type or handwrite<br>"
            "✦ Moon sign reference guide included<br>"
            "✦ Monthly herb &amp; crystal correspondences<br>"
            "✦ Print-friendly and tablet-friendly layouts</p>"
            "<p>Your practice deepens when you listen to the moon. "
            "This journal helps you hear her.</p>"
            "<p>Format: Interactive fillable PDF<br>"
            "Pages: 56 pages</p>"
        ),
    },
    {
        "title": "Grimoire Collection: Herbs & Crystals Quick Reference Guide",
        "price": "7.99",
        "filename": "grimoire_herbs_crystals.pdf",
        "tags": ["grimoire", "herbs", "crystals", "correspondences", "reference guide"],
        "description": (
            "<p>Your essential correspondence reference — always within reach.</p>"
            "<p>This grimoire collection brings together two of the most-requested "
            "reference guides into one beautifully designed PDF:</p>"
            "<p>🌿 HERB CORRESPONDENCES (49 herbs)<br>"
            "Every entry includes: magickal properties, planetary ruler, element, "
            "best uses in spellwork, safety notes, and substitution suggestions. "
            "From lavender to mugwort, basil to yarrow — the herbs your practice "
            "needs most.</p>"
            "<p>💎 CRYSTAL CORRESPONDENCES (40 crystals)<br>"
            "Every entry includes: magickal properties, chakra alignment, cleansing "
            "methods, spell applications, and pairing suggestions. From amethyst to "
            "black tourmaline, rose quartz to selenite.</p>"
            "<p>Bonus sections:<br>"
            "✦ Herb-crystal pairing chart for 20 common intentions<br>"
            "✦ Seasonal herb harvesting guide<br>"
            "✦ Crystal grid templates (3 layouts)<br>"
            "✦ Quick-reference table sorted by intention</p>"
            "<p>Whether you're mid-ritual and need a quick substitution or planning "
            "a complex working, this guide has you covered.</p>"
            "<p>Format: PDF (printable + digital)<br>"
            "Pages: 38 pages, illustrated</p>"
        ),
    },
    {
        "title": "Samhain Complete Ritual Kit — Ancestor Honor & Veil Working",
        "price": "9.99",
        "filename": "samhain_ritual_kit.pdf",
        "tags": ["samhain", "ritual kit", "sabbat", "ancestor", "halloween", "witchcraft"],
        "description": (
            "<p>The most sacred night of the Witch's Year — fully prepared for you.</p>"
            "<p>This comprehensive Samhain ritual kit gives you everything you need to "
            "honor the thinning veil, connect with ancestors, and perform powerful magick "
            "on the most potent night of the Wheel of the Year.</p>"
            "<p>Kit includes:</p>"
            "<p>🕯️ COMPLETE SAMHAIN RITUAL<br>"
            "Full script with opening, invocation, working, and closing. Designed for "
            "solitary practitioners with coven adaptation notes.</p>"
            "<p>👻 ANCESTOR HONOR CEREMONY<br>"
            "A reverent, step-by-step ceremony for connecting with those who have "
            "crossed the veil. Includes altar setup guide and offering suggestions.</p>"
            "<p>🔮 SAMHAIN DIVINATION SPREAD<br>"
            "A custom 7-card tarot spread designed specifically for veil-thin energy, "
            "with detailed position meanings and interpretation guide.</p>"
            "<p>📝 REFLECTION JOURNAL PAGES<br>"
            "Guided prompts for processing messages, recording experiences, and "
            "integrating the night's magick.</p>"
            "<p>Also includes:<br>"
            "✦ Ingredient shopping list with substitutions<br>"
            "✦ Timing guide (optimal hours for each working)<br>"
            "✦ Samhain correspondences reference (herbs, crystals, colors, deities)<br>"
            "✦ Protective ward instructions (always practice safely)</p>"
            "<p>Format: PDF bundle<br>"
            "Pages: 32 pages across 4 documents</p>"
        ),
    },
    {
        "title": "Wheel of the Year Complete Bundle — All 8 Sabbat Ritual Kits",
        "price": "39.99",
        "filename": "wheel_of_the_year_bundle.pdf",
        "tags": ["wheel of the year", "sabbats", "ritual kit", "bundle", "pagan", "witchcraft"],
        "description": (
            "<p>A full year of sabbat celebrations — every ritual, every correspondence, "
            "every working.</p>"
            "<p>This premium bundle contains complete ritual kits for all 8 sabbats on "
            "the Wheel of the Year. Each kit follows the same comprehensive format: "
            "full ritual script, ceremony guide, divination spread, journal pages, and "
            "correspondence references.</p>"
            "<p>🌱 IMBOLC (Feb 1-2) — Brigid's flame, purification, new beginnings<br>"
            "🌸 OSTARA (Mar 19-22) — Spring equinox, balance, fertility, growth<br>"
            "🔥 BELTANE (Apr 30-May 1) — Sacred fire, passion, abundance<br>"
            "☀️ LITHA (Jun 20-22) — Summer solstice, peak power, solar magick<br>"
            "🌾 LUGHNASADH (Aug 1-2) — First harvest, gratitude, skill craft<br>"
            "🍂 MABON (Sep 21-24) — Autumn equinox, balance, harvest home<br>"
            "🎃 SAMHAIN (Oct 31-Nov 1) — Ancestor honor, veil magick, divination<br>"
            "❄️ YULE (Dec 20-23) — Winter solstice, rebirth, return of light</p>"
            "<p>Each kit includes:<br>"
            "✦ Complete ritual script (solitary + coven adaptable)<br>"
            "✦ Altar setup and decoration guide<br>"
            "✦ Sabbat-specific divination spread<br>"
            "✦ Ingredient list with substitutions<br>"
            "✦ Timing guide and correspondences<br>"
            "✦ Guided reflection journal pages</p>"
            "<p>That's 8 complete kits — over 240 pages of professional-grade "
            "ritual content.</p>"
            "<p>Buying individually: $79.92 — Bundle price: $39.99 (50% savings)</p>"
            "<p>Honor the turning of the Wheel. Your practice deserves this depth.</p>"
            "<p>Format: PDF bundle (8 separate documents + master index)<br>"
            "Pages: 240+ pages total</p>"
        ),
    },
    {
        "title": "The Complete Digital Grimoire Library — 30+ Professional PDFs",
        "price": "59.99",
        "filename": "complete_digital_grimoire_library.pdf",
        "tags": ["grimoire library", "complete collection", "digital download", "witchcraft", "exclusive"],
        "description": (
            "<p>Every grimoire. Every guide. Every workbook. One sacred collection.</p>"
            "<p>This is the complete VelvetVeil digital library — over 30 "
            "professional-grade witchcraft PDFs spanning every aspect of practice. "
            "Normally valued at $390+, this BMC-exclusive bundle gives you instant "
            "access to the entire collection at a fraction of the cost.</p>"
            "<p>📚 WHAT'S INCLUDED:</p>"
            "<p>Spell Books (5 volumes)<br>"
            "✦ Love &amp; Relationships<br>"
            "✦ Protection Magic<br>"
            "✦ Abundance &amp; Money<br>"
            "✦ Healing Magic<br>"
            "✦ Banishing &amp; Cleansing</p>"
            "<p>Sabbat Ritual Kits (8 kits)<br>"
            "✦ Complete Wheel of the Year — Imbolc through Yule</p>"
            "<p>Moon Phase Journals (13 volumes)<br>"
            "✦ One for each lunar cycle + master annual journal</p>"
            "<p>Grimoire Reference Pages<br>"
            "✦ 49 Herb Correspondences<br>"
            "✦ 40 Crystal Correspondences<br>"
            "✦ 78 Tarot Card Meanings (full RWS deck)<br>"
            "✦ Spell Templates Collection<br>"
            "✦ Deity Profiles Reference</p>"
            "<p>Planners &amp; Trackers<br>"
            "✦ Yearly Practice Planner<br>"
            "✦ Spell Journal<br>"
            "✦ Moon Garden Planner<br>"
            "✦ Tarot Journal</p>"
            "<p>Why this deal exists:<br>"
            "This bundle is only available here on Buy Me a Coffee. It's my way of "
            "saying thank you to the supporters who make this community possible. "
            "The same content sells for $390+ across individual listings — you get "
            "everything for $59.99.</p>"
            "<p>Format: PDF bundle (30+ files, organized in folders)<br>"
            "Total pages: 800+</p>"
            "<p>The magick is already within you. These are just the tools to help "
            "you find it.</p>"
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers (same pattern as create_memberships.py / create_tiers_2_3.py)
# ---------------------------------------------------------------------------

def safe_execute(driver, script, *args):
    """Execute JS with alert handling."""
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
    """Save screenshot to the screenshots dir."""
    try:
        driver.save_screenshot(str(SS_DIR / name))
        print(f"   Screenshot: {name}")
    except Exception:
        pass


def wait_for_page(driver, timeout=10):
    """Wait until document.readyState is complete."""
    for _ in range(timeout * 2):
        state = safe_execute(driver, "return document.readyState")
        if state == "complete":
            return True
        time.sleep(0.5)
    return False


def dismiss_alerts(driver):
    """Dismiss any JS alerts or SweetAlert modals."""
    # Native alert
    try:
        alert = driver.switch_to.alert
        alert.accept()
        time.sleep(0.5)
    except NoAlertPresentException:
        pass
    # SweetAlert (Payhip uses it)
    safe_execute(driver, """
        const btn = document.querySelector('.swal2-confirm, .swal-button--confirm');
        if (btn) btn.click();
    """)


def fill_input(driver, selector, value):
    """Fill an input field using the React-compatible value setter pattern."""
    result = safe_execute(driver, """
        const el = document.querySelector(arguments[0]);
        if (!el) return {found: false, selector: arguments[0]};
        const setter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value').set;
        setter.call(el, arguments[1]);
        el.dispatchEvent(new Event('input', {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));
        el.dispatchEvent(new Event('blur', {bubbles: true}));
        return {found: true, value: el.value};
    """, selector, value)
    return result


def fill_quill_editor(driver, html_content):
    """Fill the Quill editor (.ql-editor) with HTML content."""
    result = safe_execute(driver, """
        const editor = document.querySelector('.ql-editor');
        if (!editor) return {found: false};
        editor.innerHTML = arguments[0];
        editor.dispatchEvent(new Event('input', {bubbles: true}));
        // Also update the hidden textarea that Quill syncs to
        const hidden = document.querySelector('input[name="p_desc"], textarea[name="p_desc"]');
        if (hidden) {
            hidden.value = arguments[0];
        }
        return {found: true, length: editor.innerHTML.length};
    """, html_content)
    return result


def upload_file(driver, filepath):
    """Upload a file via the plupload hidden file input.

    Payhip uses plupload which creates a moxie-shim wrapper around a hidden
    <input type="file">. We find the file input inside #ebook_files and
    send_keys the absolute path to it.
    """
    abs_path = str(Path(filepath).resolve())
    # Make the hidden file input visible so Selenium can interact
    result = safe_execute(driver, """
        // Find file inputs inside the ebook upload area
        const container = document.querySelector('#ebook_files');
        if (!container) return {found: false, reason: 'no #ebook_files'};

        // Find the moxie-shim file input
        const shim = container.querySelector('.moxie-shim input[type="file"]');
        if (shim) {
            // Make it interactable
            shim.style.opacity = '1';
            shim.style.position = 'relative';
            shim.style.width = '200px';
            shim.style.height = '50px';
            shim.style.zIndex = '9999';
            shim.style.display = 'block';
            return {found: true, id: shim.id, method: 'moxie-shim'};
        }

        // Fallback: any file input in the area
        const anyFile = container.querySelector('input[type="file"]');
        if (anyFile) {
            anyFile.style.opacity = '1';
            anyFile.style.position = 'relative';
            anyFile.style.width = '200px';
            anyFile.style.height = '50px';
            anyFile.style.zIndex = '9999';
            anyFile.style.display = 'block';
            return {found: true, id: anyFile.id, method: 'fallback'};
        }

        // Last resort: first file input on page
        const first = document.querySelector('input[type="file"]');
        if (first) {
            first.style.opacity = '1';
            first.style.position = 'relative';
            first.style.width = '200px';
            first.style.height = '50px';
            first.style.zIndex = '9999';
            first.style.display = 'block';
            return {found: true, id: first.id, method: 'first-on-page'};
        }

        return {found: false, reason: 'no file input found'};
    """)

    if not result or not result.get("found"):
        print(f"   WARNING: Could not find file input. {result}")
        return False

    file_input_id = result.get("id")
    print(f"   File input found: #{file_input_id} (method: {result.get('method')})")

    try:
        if file_input_id:
            element = driver.find_element("id", file_input_id)
        else:
            element = driver.find_element("css selector", "#ebook_files input[type='file']")

        element.send_keys(abs_path)
        print(f"   Sent file path: {abs_path}")
        time.sleep(3)  # Wait for plupload to process

        # Check if upload completed
        upload_check = safe_execute(driver, """
            const fileList = document.querySelector('#filelist');
            if (!fileList) return {status: 'no filelist'};
            const noFile = fileList.querySelector('.no_ebook_file_added_yet');
            if (noFile && noFile.offsetWidth > 0) return {status: 'no files yet'};
            const rows = fileList.querySelectorAll('.upload-digital-file-row');
            return {status: 'ok', fileCount: rows.length, html: fileList.textContent.trim().substring(0, 200)};
        """)
        print(f"   Upload status: {json.dumps(upload_check)}")
        return True

    except Exception as e:
        print(f"   File upload error: {str(e)[:200]}")
        return False


def add_tags(driver, tags):
    """Click Category & Tags tab and try to add tags.

    The tag system varies — could be bootstrap-tagsinput or just text inputs.
    We try multiple approaches.
    """
    # Click the Category & Tags tab
    tab_result = safe_execute(driver, """
        const tab = document.querySelector('.js-category-and-tags-tab-link, .category-and-tags-tab-link');
        if (!tab) return {found: false};
        tab.click();
        return {found: true, text: tab.textContent.trim()};
    """)
    print(f"   Tags tab: {json.dumps(tab_result)}")

    if not tab_result or not tab_result.get("found"):
        print("   WARNING: Tags tab not found, skipping tags")
        return False

    time.sleep(2)

    # Look for tag input field in the now-visible panel
    tag_result = safe_execute(driver, """
        const panel = document.querySelector('#category-and-tags-tab-panel');
        if (!panel) return {found: false, reason: 'no panel'};

        // Look for various tag input patterns
        const tagInput = panel.querySelector(
            'input[type="text"], .bootstrap-tagsinput input, .tt-input, input.tag-input'
        );
        if (!tagInput) {
            // Return what's actually in the panel
            return {
                found: false,
                reason: 'no tag input',
                panelHTML: panel.innerHTML.substring(0, 500),
                inputs: Array.from(panel.querySelectorAll('input, select, textarea'))
                    .map(i => ({tag: i.tagName, type: i.type, name: i.name, id: i.id, class: i.className.substring(0, 60)}))
            };
        }
        return {found: true, id: tagInput.id, class: tagInput.className.substring(0, 60)};
    """)
    print(f"   Tag input: {json.dumps(tag_result, indent=2)}")

    if tag_result and tag_result.get("found"):
        # Type each tag and press Enter/comma
        for tag in tags:
            safe_execute(driver, """
                const panel = document.querySelector('#category-and-tags-tab-panel');
                const tagInput = panel.querySelector(
                    'input[type="text"], .bootstrap-tagsinput input, .tt-input, input.tag-input'
                );
                if (tagInput) {
                    tagInput.focus();
                    tagInput.value = arguments[0];
                    tagInput.dispatchEvent(new Event('input', {bubbles: true}));
                    // Press Enter to add the tag
                    tagInput.dispatchEvent(new KeyboardEvent('keypress', {
                        key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true
                    }));
                    tagInput.dispatchEvent(new KeyboardEvent('keydown', {
                        key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true
                    }));
                }
            """, tag)
            time.sleep(0.5)
        return True

    # If no tag input found, try collections checkboxes
    safe_execute(driver, """
        const panel = document.querySelector('#category-and-tags-tab-panel');
        if (!panel) return;
        const checkboxes = panel.querySelectorAll('input[type="checkbox"]');
        // Just log what's available
        return Array.from(checkboxes).map(c => ({
            name: c.name, value: c.value, checked: c.checked
        }));
    """)

    return False


def click_submit(driver):
    """Click the Add Product submit button and wait for response."""
    # Scroll to submit
    safe_execute(driver, """
        const btn = document.querySelector('#addsubmit');
        if (btn) btn.scrollIntoView({behavior: 'instant', block: 'center'});
    """)
    time.sleep(1)

    # Click submit
    result = safe_execute(driver, """
        const btn = document.querySelector('#addsubmit');
        if (!btn) return {found: false};
        btn.click();
        return {found: true, value: btn.value};
    """)
    print(f"   Submit click: {json.dumps(result)}")
    return result


# ---------------------------------------------------------------------------
# Main: connect and create products
# ---------------------------------------------------------------------------

print("=" * 60)
print("  BMC SHOP PRODUCTS — PAYHIP AUTOMATION")
print("=" * 60)

options = EdgeOptions()
options.debugger_address = "localhost:9222"
driver = webdriver.Edge(options=options)

print(f"\nConnected to browser. Current URL: {driver.current_url}")
print(f"Placeholder PDFs dir: {PLACEHOLDERS_DIR}")
print(f"Screenshots dir: {SS_DIR}\n")

# Verify placeholder files exist
for product in PRODUCTS:
    pdf_path = PLACEHOLDERS_DIR / product["filename"]
    if not pdf_path.exists():
        print(f"ERROR: Missing placeholder PDF: {pdf_path}")
        sys.exit(1)
print("All placeholder PDFs verified.\n")

created_count = 0

for idx, product in enumerate(PRODUCTS):
    product_num = idx + 1
    print(f"\n{'='*60}")
    print(f"  PRODUCT {product_num}/6: {product['title'][:50]}...")
    print(f"  Price: ${product['price']}")
    print(f"{'='*60}")

    # Step 1: Navigate to the add product page
    print("\n>> Navigating to Payhip add product page...")
    driver.get("https://payhip.com/product/add/digital")
    time.sleep(5)
    wait_for_page(driver)
    dismiss_alerts(driver)

    current_url = driver.current_url
    print(f"   URL: {current_url}")

    # Verify we're on the right page
    has_form = safe_execute(driver, """
        return !!(document.querySelector('#p_name') && document.querySelector('#p_price'));
    """)
    if not has_form:
        print("   WARNING: Form not found. Taking screenshot and trying again...")
        screenshot(driver, f"shop_product{product_num}_00_no_form.png")
        time.sleep(5)
        has_form = safe_execute(driver, """
            return !!(document.querySelector('#p_name') && document.querySelector('#p_price'));
        """)
        if not has_form:
            print("   ERROR: Cannot find product form. Skipping this product.")
            continue

    screenshot(driver, f"shop_product{product_num}_01_blank_form.png")

    # Step 2: Upload placeholder PDF
    print("\n>> Uploading placeholder PDF...")
    pdf_path = PLACEHOLDERS_DIR / product["filename"]
    upload_success = upload_file(driver, pdf_path)
    time.sleep(2)
    screenshot(driver, f"shop_product{product_num}_02_after_upload.png")

    # Step 3: Fill title
    print(f"\n>> Filling title: {product['title'][:60]}...")
    title_result = fill_input(driver, "#p_name", product["title"])
    print(f"   Title result: {json.dumps(title_result)}")
    time.sleep(0.5)

    # Step 4: Fill price
    print(f"\n>> Setting price: ${product['price']}...")
    # Clear existing value first
    safe_execute(driver, """
        const el = document.querySelector('#p_price');
        if (el) {
            el.focus();
            el.select();
        }
    """)
    time.sleep(0.3)
    price_result = fill_input(driver, "#p_price", product["price"])
    print(f"   Price result: {json.dumps(price_result)}")
    time.sleep(0.5)

    # Step 5: Fill description via Quill editor
    print("\n>> Filling description...")
    desc_result = fill_quill_editor(driver, product["description"])
    print(f"   Description result: {json.dumps(desc_result)}")
    time.sleep(0.5)

    screenshot(driver, f"shop_product{product_num}_03_filled.png")

    # Step 6: Try adding tags
    print("\n>> Adding tags...")
    tags_success = add_tags(driver, product["tags"])
    time.sleep(1)
    screenshot(driver, f"shop_product{product_num}_04_tags.png")

    # Switch back to Product tab before submitting
    safe_execute(driver, """
        const tab = document.querySelector('.js-product-tab-link');
        if (tab) tab.click();
    """)
    time.sleep(1)

    # Step 7: Submit
    print("\n>> Submitting product...")
    screenshot(driver, f"shop_product{product_num}_05_pre_submit.png")
    submit_result = click_submit(driver)
    time.sleep(8)  # Wait for server response

    # Check for success or error
    dismiss_alerts(driver)
    new_url = driver.current_url
    print(f"   Post-submit URL: {new_url}")

    # Check for success indicators
    success_check = safe_execute(driver, """
        // Toastify success message
        const toast = document.querySelector('.toastify, .Toastify__toast--success');
        // SweetAlert success
        const swal = document.querySelector('.swal2-success, .swal-icon--success');
        // URL changed (redirected to product list or edit page)
        const urlChanged = !window.location.href.includes('/add/');
        // Error messages
        const errors = document.querySelectorAll('.alert-danger, .error-message, .text-danger');
        const errorTexts = Array.from(errors).map(e => e.textContent.trim()).filter(t => t);

        return {
            toast: toast ? toast.textContent.trim() : null,
            swal: !!swal,
            urlChanged: urlChanged,
            currentUrl: window.location.href,
            errors: errorTexts
        };
    """)
    print(f"   Success check: {json.dumps(success_check, indent=2)}")

    screenshot(driver, f"shop_product{product_num}_06_result.png")

    if success_check and (success_check.get("urlChanged") or success_check.get("swal")):
        created_count += 1
        print(f"\n   Product {product_num} created successfully!")
    elif success_check and success_check.get("errors"):
        print(f"\n   WARNING: Errors detected: {success_check['errors']}")
    else:
        print(f"\n   Product {product_num} submission completed (check screenshot)")
        created_count += 1  # Optimistic count

    time.sleep(2)


# ---------------------------------------------------------------------------
# Verification: check the public shop page
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  VERIFICATION — {created_count}/{len(PRODUCTS)} products submitted")
print(f"{'='*60}")

print("\n>> Checking Payhip shop page...")
# Navigate to the seller's product list
driver.get("https://payhip.com/seller/products")
time.sleep(5)
screenshot(driver, "shop_verify_01_product_list.png")

products_list = safe_execute(driver, """
    const rows = document.querySelectorAll('.product-row, .js-product-row, tr[data-id], .card, .product-card');
    if (rows.length > 0) {
        return Array.from(rows).map(r => ({
            text: r.textContent.trim().substring(0, 100),
            tag: r.tagName
        }));
    }
    // Fallback: just grab page content
    return {
        pageText: document.body.textContent.trim().substring(0, 500),
        title: document.title
    };
""")
print(f"   Product list: {json.dumps(products_list, indent=2)}")

print("\n>> Checking public BMC shop page...")
driver.get("https://buymeacoffee.com/witchcraft/extras")
time.sleep(6)
screenshot(driver, "shop_verify_02_public_page.png")

public_check = safe_execute(driver, """
    const items = document.querySelectorAll('[class*="product"], [class*="extra"], [class*="shop"]');
    const texts = Array.from(items)
        .map(i => i.textContent.trim().substring(0, 100))
        .filter(t => t.length > 5);
    return {
        itemCount: items.length,
        texts: texts.slice(0, 20),
        pageTitle: document.title,
        url: window.location.href
    };
""")
print(f"   Public page: {json.dumps(public_check, indent=2)}")

screenshot(driver, "shop_verify_03_final.png")

print(f"\n{'='*60}")
print(f"  DONE — {created_count}/{len(PRODUCTS)} products submitted")
print(f"{'='*60}")
print("\nBrowser left open. Check screenshots in:", SS_DIR)
print("NOTE: No driver.quit() — browser session preserved.")
