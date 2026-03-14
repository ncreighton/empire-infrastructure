"""
Shared Systeme.io Automation using SeleniumBase UC Mode.

Replaces 16 identical per-site systeme_uc_automation.py files.
Uses undetected chromedriver to bypass bot detection.

Credentials loaded from environment variables:
    SYSTEME_EMAIL — Systeme.io login email
    SYSTEME_PASSWORD — Systeme.io login password

Usage:
    from EMPIRE_BRAIN.shared.systeme_base import setup_systeme_automation
    setup_systeme_automation()
"""

import os
import time

from seleniumbase import SB

SYSTEME_EMAIL = os.getenv("SYSTEME_EMAIL", "aiautomationblueprint@gmail.com")
SYSTEME_PASSWORD = os.getenv("SYSTEME_PASSWORD", "Ashlynn.09")


def setup_systeme_automation():
    """Set up automation rule in Systeme.io using UC Mode"""

    # UC Mode with incognito for maximum stealth
    with SB(uc=True, incognito=True, headed=True) as sb:
        try:
            # Step 1: Navigate to login page with reconnect to avoid detection
            print("Navigating to Systeme.io login...")
            sb.uc_open_with_reconnect("https://systeme.io/en/login", reconnect_time=4)
            time.sleep(2)

            # Take screenshot
            sb.save_screenshot("systeme_uc_login.png")

            # Step 2: Fill in login credentials
            print("Filling login credentials...")

            # Wait for email field and fill
            sb.wait_for_element('input[type="email"]', timeout=15)
            sb.uc_click('input[type="email"]')
            time.sleep(0.5)
            sb.type('input[type="email"]', SYSTEME_EMAIL)
            time.sleep(0.5)

            # Fill password
            sb.uc_click('input[type="password"]')
            time.sleep(0.5)
            sb.type('input[type="password"]', SYSTEME_PASSWORD)
            time.sleep(0.5)

            sb.save_screenshot("systeme_uc_filled.png")

            # Step 3: Click login button using UC click
            print("Clicking login button...")
            sb.uc_click('button[type="submit"]')
            time.sleep(5)

            sb.save_screenshot("systeme_uc_after_login.png")

            # Check current URL
            current_url = sb.get_current_url()
            print(f"Current URL after login: {current_url}")

            # Step 4: Navigate to automation rules
            if "dashboard" in current_url or "systeme.io" in current_url:
                print("Login successful! Navigating to automation rules...")

                # Use UC open to navigate
                sb.uc_open_with_reconnect(
                    "https://systeme.io/dashboard/automation-rules/create",
                    reconnect_time=4
                )
                time.sleep(3)

                sb.save_screenshot("systeme_uc_create_rule.png")
                print(f"Current URL: {sb.get_current_url()}")

                # Step 5: Click the Trigger + button
                print("Setting up trigger...")

                # Wait for the page to load - use XPath for text matching
                sb.wait_for_element('//span[contains(text(), "Trigger")]', timeout=15, by="xpath")
                time.sleep(2)

                # Find and click the + button for trigger
                # The + buttons are SVG elements in circular containers
                # Try clicking with JavaScript to find the right element

                # Get all clickable elements info
                elements_info = sb.execute_script("""
                    const results = [];
                    const svgs = document.querySelectorAll('svg');
                    svgs.forEach((svg, i) => {
                        const rect = svg.getBoundingClientRect();
                        if (rect.width > 20 && rect.width < 50 && rect.y > 150 && rect.y < 250) {
                            results.push({
                                index: i,
                                x: rect.x,
                                y: rect.y,
                                w: rect.width,
                                h: rect.height,
                                parent: svg.parentElement?.className || 'no-class'
                            });
                        }
                    });
                    return results;
                """)
                print(f"Found SVG elements: {elements_info}")

                # Click the first one that's likely the trigger + button
                # It should be on the left side of the page
                for el in elements_info:
                    if el['x'] < 900:  # Left side
                        # Calculate center of SVG
                        click_x = int(el['x'] + el['w'] / 2)
                        click_y = int(el['y'] + el['h'] / 2)
                        print(f"Clicking trigger + at ({click_x}, {click_y})")

                        # Use JavaScript to click the parent element or dispatch click event
                        sb.execute_script(f"""
                            const svgs = document.querySelectorAll('svg');
                            const svg = svgs[{el['index']}];
                            // Try clicking the parent button/container
                            const parent = svg.closest('button') || svg.closest('[role="button"]') || svg.parentElement;
                            if (parent) {{
                                parent.click();
                            }} else {{
                                // Dispatch click event on SVG
                                const event = new MouseEvent('click', {{
                                    bubbles: true,
                                    cancelable: true,
                                    view: window,
                                    clientX: {click_x},
                                    clientY: {click_y}
                                }});
                                svg.dispatchEvent(event);
                            }}
                        """)
                        time.sleep(3)
                        break

                sb.save_screenshot("systeme_uc_trigger_clicked.png")

                # Check if Tag added option appeared in the modal
                page_text = sb.get_page_source()
                if "Tag added" in page_text:
                    print("Modal opened! Looking for 'Tag added' option...")

                    # Use JavaScript to find and click "Tag added"
                    clicked = sb.execute_script("""
                        // Find all elements containing "Tag added"
                        const elements = document.querySelectorAll('*');
                        for (let el of elements) {
                            if (el.textContent === 'Tag added' ||
                                (el.textContent && el.textContent.trim().startsWith('Tag added') &&
                                 el.textContent.length < 50)) {
                                // Check if it's visible
                                const rect = el.getBoundingClientRect();
                                if (rect.width > 0 && rect.height > 0 && rect.y > 100) {
                                    el.click();
                                    return 'clicked: ' + el.tagName + ' at ' + rect.x + ',' + rect.y;
                                }
                            }
                        }
                        return 'not found';
                    """)
                    print(f"Tag added click result: {clicked}")
                    time.sleep(3)
                    sb.save_screenshot("systeme_uc_tag_added.png")

                    # Now we should see a tag selection dropdown with "Choose tag" text
                    print("Looking for 'Choose tag' dropdown...")
                    time.sleep(2)

                    # Debug: Get all text content on page to understand structure
                    page_text_debug = sb.execute_script("""
                        // Get unique text snippets on page - wider range
                        const texts = new Set();
                        document.querySelectorAll('*').forEach(el => {
                            const text = el.textContent?.trim();
                            if (text && text.length > 3 && text.length < 50) {
                                const rect = el.getBoundingClientRect();
                                if (rect.y > 250 && rect.y < 600 && rect.width > 50) {
                                    texts.add(text.substring(0, 30) + ' @y=' + Math.round(rect.y));
                                }
                            }
                        });
                        return Array.from(texts).slice(0, 50);
                    """)
                    print(f"Page text elements (y=250-600): {page_text_debug}")

                    # Also look for input elements, select elements, or divs that look like dropdowns
                    inputs_debug = sb.execute_script("""
                        const results = [];
                        // Look for inputs, selects, and clickable divs
                        document.querySelectorAll('input, select, [role="listbox"], [role="combobox"], [role="button"], button').forEach(el => {
                            const rect = el.getBoundingClientRect();
                            if (rect.y > 300 && rect.y < 500 && rect.width > 100) {
                                results.push({
                                    tag: el.tagName,
                                    type: el.type || el.getAttribute('role') || '',
                                    classes: (el.className?.toString?.() || '').substring(0, 50),
                                    x: Math.round(rect.x),
                                    y: Math.round(rect.y),
                                    w: Math.round(rect.width),
                                    h: Math.round(rect.height),
                                    placeholder: el.placeholder || '',
                                    value: el.value?.substring(0, 20) || ''
                                });
                            }
                        });
                        return results;
                    """)
                    print(f"Input/select elements: {inputs_debug}")

                    # First, let's debug and see the dropdown structure - search by "Choose" or "tag"
                    dropdown_debug = sb.execute_script("""
                        const results = [];
                        // Find elements containing "Choose" anywhere
                        document.querySelectorAll('*').forEach(el => {
                            const text = el.textContent?.toLowerCase() || '';
                            if ((text.includes('choose') || text.includes('tag')) && el.textContent.length < 100) {
                                const rect = el.getBoundingClientRect();
                                if (rect.width > 30 && rect.height > 5 && rect.y > 300 && rect.y < 450) {
                                    results.push({
                                        tag: el.tagName,
                                        classes: (el.className?.toString?.() || '').substring(0, 40),
                                        x: Math.round(rect.x),
                                        y: Math.round(rect.y),
                                        w: Math.round(rect.width),
                                        h: Math.round(rect.height),
                                        text: el.textContent?.trim().substring(0, 30)
                                    });
                                }
                            }
                        });
                        return results.slice(0, 20);
                    """)
                    print(f"Dropdown elements found: {dropdown_debug}")

                    # Click the "Choose tag" dropdown - it's an INPUT with placeholder="Choose tag"
                    dropdown_clicked = sb.execute_script("""
                        // The dropdown is an INPUT element with placeholder "Choose tag"
                        // or a BUTTON that triggers the dropdown

                        // Method 1: Find INPUT with placeholder "Choose tag"
                        const inputs = document.querySelectorAll('input');
                        for (let input of inputs) {
                            if (input.placeholder === 'Choose tag' || input.placeholder?.includes('Choose')) {
                                const rect = input.getBoundingClientRect();
                                if (rect.width > 100 && rect.y > 300) {
                                    input.click();
                                    input.focus();
                                    return 'clicked INPUT with placeholder at y=' + Math.round(rect.y);
                                }
                            }
                        }

                        // Method 2: Find the BUTTON that wraps or is near the input
                        const buttons = document.querySelectorAll('button');
                        for (let btn of buttons) {
                            const rect = btn.getBoundingClientRect();
                            // The button is at y=352 with width 408
                            if (rect.y > 340 && rect.y < 380 && rect.width > 300) {
                                btn.click();
                                return 'clicked BUTTON at y=' + Math.round(rect.y);
                            }
                        }

                        // Method 3: Click the container div that has the dropdown
                        const divs = document.querySelectorAll('div');
                        for (let div of divs) {
                            const rect = div.getBoundingClientRect();
                            // Look for the dropdown container (around y=352, width ~408)
                            if (rect.y > 340 && rect.y < 380 && rect.width > 300 && rect.height > 35 && rect.height < 60) {
                                const hasInput = div.querySelector('input[placeholder="Choose tag"]');
                                if (hasInput) {
                                    div.click();
                                    return 'clicked DIV containing input at y=' + Math.round(rect.y);
                                }
                            }
                        }

                        return 'Choose tag dropdown not found';
                    """)
                    print(f"Dropdown click result: {dropdown_clicked}")
                    time.sleep(3)
                    sb.save_screenshot("systeme_uc_dropdown.png")

                    # Now look for our tag in the dropdown list
                    print("Looking for aidiscoverydigest-toolkit in dropdown...")

                    # First, let's see what options are available (individual items only)
                    options_info = sb.execute_script("""
                        // Get individual dropdown options (not containers)
                        const options = [];
                        const elements = document.querySelectorAll('div, li');
                        for (let el of elements) {
                            // Only get leaf elements (no nested divs with content)
                            const text = el.textContent?.trim();
                            const rect = el.getBoundingClientRect();
                            // Check that this is a single option, not a container
                            const childDivs = el.querySelectorAll('div');
                            if (text && text.length > 5 && text.length < 60 &&
                                rect.width > 200 && rect.height > 25 && rect.height < 50 &&
                                rect.y > 390 && rect.y < 700 &&
                                childDivs.length === 0) {  // No nested divs
                                options.push({
                                    text: text,
                                    y: Math.round(rect.y),
                                    tag: el.tagName
                                });
                            }
                        }
                        return options;
                    """)
                    print(f"Available options: {options_info}")

                    # Type in the input to filter for our tag using SeleniumBase
                    print("Typing to filter dropdown...")
                    sb.type('input[placeholder="Choose tag"]', 'aidiscoverydigest-toolkit')
                    time.sleep(2)  # Wait for filtering
                    sb.save_screenshot("systeme_uc_filtered.png")

                    # Now look for the filtered options
                    filtered_options = sb.execute_script("""
                        const options = [];
                        const elements = document.querySelectorAll('div, li, span');
                        for (let el of elements) {
                            const text = el.textContent?.trim();
                            const rect = el.getBoundingClientRect();
                            if (text && text.includes('aidiscoverydigest') && text.length < 60 &&
                                rect.width > 100 && rect.height > 15 && rect.height < 60 &&
                                rect.y > 390 && rect.y < 700) {
                                options.push({
                                    text: text,
                                    y: Math.round(rect.y),
                                    w: Math.round(rect.width),
                                    h: Math.round(rect.height),
                                    tag: el.tagName
                                });
                            }
                        }
                        return options;
                    """)
                    print(f"Filtered options: {filtered_options}")

                    # Now try to find and click the exact tag option in the dropdown
                    tag_clicked = sb.execute_script("""
                        // Look for the tag option "aidiscoverydigest-toolkit" in dropdown list
                        const elements = document.querySelectorAll('div, li, span');

                        // First: look for exact match in dropdown area
                        for (let el of elements) {
                            const text = el.textContent?.trim();
                            if (text === 'aidiscoverydigest-toolkit') {
                                const rect = el.getBoundingClientRect();
                                // Must be in dropdown area (below input at y~352)
                                if (rect.width > 100 && rect.height > 20 && rect.height < 50 &&
                                    rect.y > 390 && rect.y < 700) {
                                    el.click();
                                    return 'clicked EXACT option at y=' + Math.round(rect.y);
                                }
                            }
                        }

                        // Second: look for any clickable element with our tag
                        for (let el of elements) {
                            const text = el.textContent?.trim();
                            if (text && text.includes('aidiscoverydigest-toolkit') && text.length < 50) {
                                const rect = el.getBoundingClientRect();
                                if (rect.width > 100 && rect.height > 20 && rect.y > 390) {
                                    el.click();
                                    return 'clicked option containing tag at y=' + Math.round(rect.y);
                                }
                            }
                        }

                        // Third: Try pressing down arrow and Enter to select first option
                        const input = document.querySelector('input[placeholder="Choose tag"]');
                        if (input) {
                            input.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
                            return 'dispatched ArrowDown - need to click option';
                        }

                        return 'aidiscoverydigest-toolkit option not found';
                    """)
                    print(f"Tag click result: {tag_clicked}")

                    # If tag not clicked, try using keyboard
                    if 'not found' in tag_clicked or 'ArrowDown' in tag_clicked:
                        print("Trying keyboard navigation...")
                        time.sleep(0.5)
                        # Press down arrow and Enter
                        sb.execute_script("""
                            const input = document.querySelector('input[placeholder="Choose tag"]');
                            if (input) {
                                input.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
                            }
                        """)
                        time.sleep(0.3)
                        sb.execute_script("""
                            const input = document.querySelector('input[placeholder="Choose tag"]');
                            if (input) {
                                input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
                            }
                        """)
                        time.sleep(1)
                        tag_clicked = "used keyboard Enter"
                    print(f"Tag selection result: {tag_clicked}")
                    time.sleep(3)

                    # Verify tag was selected by checking if dropdown closed and tag shows
                    tag_verify = sb.execute_script("""
                        // Check if the dropdown input now shows the selected tag
                        const input = document.querySelector('input[placeholder="Choose tag"]');
                        if (input && input.value && input.value.includes('aidiscoverydigest')) {
                            return 'verified: input shows ' + input.value;
                        }
                        // Also check if there's a selected tag displayed
                        const selected = document.querySelector('[class*="selected"]');
                        if (selected && selected.textContent?.includes('aidiscoverydigest')) {
                            return 'verified: selected shows ' + selected.textContent;
                        }
                        // Check if dropdown closed (no visible options)
                        const options = document.querySelectorAll('div');
                        for (let opt of options) {
                            const text = opt.textContent?.trim();
                            const rect = opt.getBoundingClientRect();
                            if (text === 'aidiscoverydigest-toolkit' && rect.y > 300 && rect.y < 450 &&
                                rect.width > 200) {
                                return 'tag visible in trigger section: ' + Math.round(rect.y);
                            }
                        }
                        return 'tag selection not verified';
                    """)
                    print(f"Tag verification: {tag_verify}")
                    sb.save_screenshot("systeme_uc_tag_selected.png")

                # === STEP 6: SET UP ACTION ===
                print("\n=== Setting up Action (Send email) ===")

                # First scroll page to top to ensure we see the full rule builder
                sb.execute_script("window.scrollTo(0, 0);")
                time.sleep(1)
                sb.save_screenshot("systeme_uc_before_action.png")

                # Re-detect SVG elements for the + buttons (wider search range)
                elements_info = sb.execute_script("""
                    const results = [];
                    const svgs = document.querySelectorAll('svg');
                    svgs.forEach((svg, i) => {
                        const rect = svg.getBoundingClientRect();
                        // Look for + button SVGs (small, circular, in the rule builder area)
                        if (rect.width > 20 && rect.width < 50 && rect.y > 100 && rect.y < 350) {
                            const parent = svg.parentElement;
                            const parentClasses = parent?.className || '';
                            // Only include if parent looks like a button (has rounded-full class)
                            if (parentClasses.includes('rounded-full') || parentClasses.includes('group')) {
                                results.push({
                                    index: i,
                                    x: Math.round(rect.x),
                                    y: Math.round(rect.y),
                                    w: Math.round(rect.width),
                                    h: Math.round(rect.height),
                                    parent: parentClasses.substring(0, 50)
                                });
                            }
                        }
                    });
                    return results;
                """)
                print(f"SVG elements after tag selection: {elements_info}")

                # Click the Action + button (right side of page, x > 900)
                action_clicked = False
                for el in elements_info:
                    if el['x'] > 900:  # Right side - Action button
                        click_x = int(el['x'] + el['w'] / 2)
                        click_y = int(el['y'] + el['h'] / 2)
                        print(f"Clicking Action + at ({click_x}, {click_y})")
                        sb.execute_script(f"""
                            const svgs = document.querySelectorAll('svg');
                            const svg = svgs[{el['index']}];
                            const parent = svg.closest('button') || svg.closest('[role="button"]') || svg.parentElement;
                            if (parent) {{
                                parent.click();
                            }}
                        """)
                        action_clicked = True
                        time.sleep(3)
                        break

                # If no Action + found via SVG, try clicking directly by position
                if not action_clicked:
                    print("SVG Action + not found, trying direct click on Action area...")
                    action_result = sb.execute_script("""
                        // Look for the Action section + button
                        const elements = document.querySelectorAll('*');
                        for (let el of elements) {
                            const rect = el.getBoundingClientRect();
                            const classes = el.className?.toString?.() || '';
                            // Look for rounded-full elements on the right side
                            if (classes.includes('rounded-full') && classes.includes('group') &&
                                rect.x > 1000 && rect.y > 100 && rect.y < 300 &&
                                rect.width > 25 && rect.width < 50) {
                                el.click();
                                return 'clicked Action + at x=' + Math.round(rect.x) + ' y=' + Math.round(rect.y);
                            }
                        }
                        // Try finding by the Action text and its nearby + button
                        const allEls = document.querySelectorAll('*');
                        for (let h of allEls) {
                            if (h.textContent?.trim() === 'Action') {
                                const rect = h.getBoundingClientRect();
                                if (rect.x > 600 && rect.y > 100 && rect.y < 250) {
                                    // The + button should be to the right of Action text
                                    const parent = h.closest('div');
                                    if (parent) {
                                        const siblings = parent.querySelectorAll('svg');
                                        for (let sib of siblings) {
                                            const sibRect = sib.getBoundingClientRect();
                                            if (sibRect.width > 25 && sibRect.width < 50) {
                                                const btn = sib.closest('button') || sib.parentElement;
                                                if (btn) {
                                                    btn.click();
                                                    return 'clicked + near Action text at y=' + Math.round(sibRect.y);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        return 'Action + button not found';
                    """)
                    print(f"Direct Action click: {action_result}")
                    time.sleep(3)

                sb.save_screenshot("systeme_uc_action_clicked.png")

                # Look for "Send email" option and click it
                if "Send email" in sb.get_page_source():
                    print("Found 'Send email' option, clicking...")
                    send_clicked = sb.execute_script("""
                        const elements = document.querySelectorAll('*');
                        for (let el of elements) {
                            if (el.textContent && el.textContent.trim() === 'Send email') {
                                const rect = el.getBoundingClientRect();
                                if (rect.width > 0 && rect.height > 0 && rect.y > 100) {
                                    el.click();
                                    return 'clicked Send email at ' + rect.y;
                                }
                            }
                        }
                        return 'Send email not found';
                    """)
                    print(f"Send email click result: {send_clicked}")
                    time.sleep(2)

                sb.save_screenshot("systeme_uc_action_set.png")

                # Check if we need to select an email
                page_check = sb.get_page_source()
                if "Choose email" in page_check:
                    print("Need to select an email for Send email action...")

                    # Click the Choose email dropdown
                    email_dropdown = sb.execute_script("""
                        const inputs = document.querySelectorAll('input');
                        for (let input of inputs) {
                            if (input.placeholder === 'Choose email' || input.placeholder?.includes('email')) {
                                const rect = input.getBoundingClientRect();
                                if (rect.x > 600 && rect.y > 300) {  // Right side of page
                                    input.click();
                                    input.focus();
                                    return 'clicked email input at y=' + Math.round(rect.y);
                                }
                            }
                        }
                        // Try button
                        const buttons = document.querySelectorAll('button');
                        for (let btn of buttons) {
                            const rect = btn.getBoundingClientRect();
                            if (rect.x > 600 && rect.y > 340 && rect.y < 400 && rect.width > 200) {
                                btn.click();
                                return 'clicked email button at y=' + Math.round(rect.y);
                            }
                        }
                        return 'email dropdown not found';
                    """)
                    print(f"Email dropdown: {email_dropdown}")
                    time.sleep(2)
                    sb.save_screenshot("systeme_uc_email_dropdown.png")

                    # Look for available emails and select first one (or create new)
                    email_options = sb.execute_script("""
                        const options = [];
                        const elements = document.querySelectorAll('div, li, span');
                        for (let el of elements) {
                            const text = el.textContent?.trim();
                            const rect = el.getBoundingClientRect();
                            if (text && text.length > 5 && text.length < 100 &&
                                rect.x > 600 && rect.y > 400 && rect.y < 700 &&
                                rect.width > 100 && rect.height > 20 && rect.height < 50) {
                                options.push({
                                    text: text.substring(0, 50),
                                    y: Math.round(rect.y)
                                });
                            }
                        }
                        return options.slice(0, 10);
                    """)
                    print(f"Email options: {email_options}")

                    # Check if there's a "Create email" or "+" button
                    create_email = sb.execute_script("""
                        // Look for a + button or "Create" option near the email dropdown
                        const elements = document.querySelectorAll('*');
                        for (let el of elements) {
                            const text = el.textContent?.trim();
                            const rect = el.getBoundingClientRect();
                            if ((text === '+' || text === 'Create' || text === 'Create email' ||
                                 text?.includes('New email')) &&
                                rect.x > 600 && rect.y > 300 && rect.y < 450) {
                                el.click();
                                return 'clicked create at ' + text;
                            }
                        }
                        // Look for the + icon near Email label
                        const svgs = document.querySelectorAll('svg');
                        for (let svg of svgs) {
                            const rect = svg.getBoundingClientRect();
                            if (rect.x > 700 && rect.x < 800 && rect.y > 320 && rect.y < 360 &&
                                rect.width < 30) {
                                const parent = svg.parentElement;
                                if (parent) {
                                    parent.click();
                                    return 'clicked + icon for email';
                                }
                            }
                        }
                        return 'create email not found';
                    """)
                    print(f"Create email result: {create_email}")
                    time.sleep(3)
                    sb.save_screenshot("systeme_uc_email_create.png")

                    # Check if email creation modal opened
                    if "Create email message" in sb.get_page_source():
                        print("Email creation modal opened! Filling in email details...")

                        # Fill in Subject
                        subject_filled = sb.execute_script("""
                            const inputs = document.querySelectorAll('input');
                            for (let input of inputs) {
                                const label = input.closest('div')?.querySelector('label');
                                if (label?.textContent?.includes('Subject') ||
                                    input.previousElementSibling?.textContent?.includes('Subject')) {
                                    input.value = 'Your AI Discovery Digest Toolkit is Here!';
                                    input.dispatchEvent(new Event('input', { bubbles: true }));
                                    return 'filled subject';
                                }
                            }
                            // Try finding by position (first visible input in modal)
                            for (let input of inputs) {
                                const rect = input.getBoundingClientRect();
                                if (rect.y > 150 && rect.y < 250 && rect.width > 300) {
                                    input.value = 'Your AI Discovery Digest Toolkit is Here!';
                                    input.dispatchEvent(new Event('input', { bubbles: true }));
                                    return 'filled subject by position at y=' + Math.round(rect.y);
                                }
                            }
                            return 'subject input not found';
                        """)
                        print(f"Subject: {subject_filled}")
                        time.sleep(1)

                        # Fill in Preview text
                        preview_filled = sb.execute_script("""
                            const inputs = document.querySelectorAll('input');
                            for (let input of inputs) {
                                const rect = input.getBoundingClientRect();
                                // Preview is second input, below subject
                                if (rect.y > 250 && rect.y < 350 && rect.width > 300) {
                                    input.value = 'Download your free Top 100 AI Tools Database';
                                    input.dispatchEvent(new Event('input', { bubbles: true }));
                                    return 'filled preview';
                                }
                            }
                            return 'preview input not found';
                        """)
                        print(f"Preview: {preview_filled}")
                        time.sleep(1)

                        # Fill in Body - use the rich text editor
                        body_filled = sb.execute_script("""
                            // Find the body editor (contenteditable div or iframe)
                            const editors = document.querySelectorAll('[contenteditable="true"], .ProseMirror, [class*="editor"]');
                            for (let editor of editors) {
                                const rect = editor.getBoundingClientRect();
                                if (rect.y > 350 && rect.width > 300) {
                                    editor.innerHTML = `
                                        <p>Hi there!</p>
                                        <p>Welcome to AI Discovery Digest! Thank you for joining our community of AI explorers.</p>
                                        <p>As promised, here's your <strong>Top 100 AI Tools Database</strong> - a curated collection of the most useful AI tools across different categories.</p>
                                        <p>Inside you'll find:</p>
                                        <ul>
                                            <li>100 hand-picked AI tools organized by category</li>
                                            <li>Ratings and quick descriptions for each tool</li>
                                            <li>Direct links to try them out</li>
                                        </ul>
                                        <p>Download your toolkit using the attachment below!</p>
                                        <p>Stay curious,<br>The AI Discovery Digest Team</p>
                                    `;
                                    editor.dispatchEvent(new Event('input', { bubbles: true }));
                                    return 'filled body';
                                }
                            }
                            return 'body editor not found';
                        """)
                        print(f"Body: {body_filled}")
                        time.sleep(1)

                        # Select sender email address
                        sender_selected = sb.execute_script("""
                            // Click the sender email dropdown
                            const selects = document.querySelectorAll('select, [role="listbox"], button');
                            for (let sel of selects) {
                                const rect = sel.getBoundingClientRect();
                                if (rect.x > 850 && rect.y > 350 && rect.y < 420 && rect.width > 100) {
                                    sel.click();
                                    return 'clicked sender dropdown';
                                }
                            }
                            return 'sender dropdown not found';
                        """)
                        print(f"Sender dropdown: {sender_selected}")
                        time.sleep(2)

                        # Select first available sender email
                        sb.execute_script("""
                            const options = document.querySelectorAll('div, li, option');
                            for (let opt of options) {
                                const text = opt.textContent?.trim();
                                const rect = opt.getBoundingClientRect();
                                if (text && text.includes('@') && rect.y > 400 && rect.width > 100) {
                                    opt.click();
                                    return 'selected ' + text;
                                }
                            }
                        """)
                        time.sleep(1)

                        sb.save_screenshot("systeme_uc_email_filled.png")

                        # Scroll the modal to find the Save button
                        scroll_modal = sb.execute_script("""
                            // Find the modal container and scroll it
                            const modals = document.querySelectorAll('[class*="modal"], [class*="dialog"], [role="dialog"]');
                            for (let modal of modals) {
                                const rect = modal.getBoundingClientRect();
                                if (rect.height > 300) {
                                    modal.scrollTop = modal.scrollHeight;
                                    return 'scrolled modal';
                                }
                            }
                            // Try scrolling the main content area
                            const scrollables = document.querySelectorAll('[class*="overflow"]');
                            for (let el of scrollables) {
                                const rect = el.getBoundingClientRect();
                                if (rect.height > 400 && rect.width > 500) {
                                    el.scrollTop = el.scrollHeight;
                                    return 'scrolled content area';
                                }
                            }
                            // Scroll window
                            window.scrollTo(0, document.body.scrollHeight);
                            return 'scrolled window';
                        """)
                        print(f"Scroll modal: {scroll_modal}")
                        time.sleep(1)
                        sb.save_screenshot("systeme_uc_modal_scrolled.png")

                        # Look for Save button in the modal (search more broadly)
                        save_email = sb.execute_script("""
                            const buttons = document.querySelectorAll('button');
                            for (let btn of buttons) {
                                const text = btn.textContent?.trim().toLowerCase();
                                const rect = btn.getBoundingClientRect();
                                // Look for save/create button that's visible
                                if ((text === 'save' || text === 'create' || text.includes('save') ||
                                     text === 'ok' || text === 'confirm') &&
                                    rect.width > 50 && rect.height > 25) {
                                    btn.click();
                                    return 'clicked button: ' + text + ' at y=' + Math.round(rect.y);
                                }
                            }
                            // Look for primary/submit button style
                            for (let btn of buttons) {
                                const classes = btn.className?.toLowerCase() || '';
                                const rect = btn.getBoundingClientRect();
                                if ((classes.includes('primary') || classes.includes('submit') ||
                                     classes.includes('btn-') || btn.type === 'submit') &&
                                    rect.width > 50 && rect.y > 500) {
                                    btn.click();
                                    return 'clicked primary button at y=' + Math.round(rect.y);
                                }
                            }
                            return 'save button not found';
                        """)
                        print(f"Save email: {save_email}")

                        # If still not found, try using keyboard shortcut or pressing Enter
                        if 'not found' in save_email:
                            print("Trying Enter key to save...")
                            sb.execute_script("""
                                document.dispatchEvent(new KeyboardEvent('keydown', {
                                    key: 'Enter',
                                    ctrlKey: true,
                                    bubbles: true
                                }));
                            """)
                            time.sleep(1)

                        time.sleep(3)
                        sb.save_screenshot("systeme_uc_email_saved.png")

                # === STEP 7: SAVE THE RULE ===
                print("\n=== Saving the rule ===")

                # Click the Save rule button
                save_clicked = sb.execute_script("""
                    const buttons = document.querySelectorAll('button');
                    for (let btn of buttons) {
                        if (btn.textContent && btn.textContent.includes('Save rule')) {
                            btn.click();
                            return 'clicked Save rule';
                        }
                    }
                    return 'Save button not found';
                """)
                print(f"Save result: {save_clicked}")
                time.sleep(3)
                sb.save_screenshot("systeme_uc_saved.png")

                sb.save_screenshot("systeme_uc_final.png")

                # Keep browser open to see results
                print("\nBrowser will stay open for 60 seconds for inspection...")
                print("Check the screenshots in the project folder.")
                time.sleep(60)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sb.save_screenshot("systeme_uc_error.png")
            time.sleep(30)


if __name__ == '__main__':
    setup_systeme_automation()
