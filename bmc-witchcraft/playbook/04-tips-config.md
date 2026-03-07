# Buy Me a Coffee — Tips Configuration

> Configure in **BMC Dashboard → Settings → Page**.

---

## Custom Coffee Name

| Setting | Value |
|---------|-------|
| Item name | `potion` |
| Singular | "Buy me a potion" |
| Plural | "Buy me 3 potions" |

---

## Suggested Amounts

| Potions | Price | Suggested Label |
|---------|-------|-----------------|
| 1 | $3 | "A quick blessing" |
| 3 | $9 | "A moon offering" |
| 5 | $15 | "A sacred tribute" |

Configure these three as the default suggestion buttons on the tip page.

---

## Thank-You Message (after tip received)

```
Your generosity fuels the magick! ✨

Thank you for supporting Witchcraft For Beginners. Every potion helps me create more free guides, grimoire references, and ritual resources for our growing community of practitioners.

May your kindness return to you threefold.

Blessed be,
— Witchcraft For Beginners 🌙
```

---

## Tip Page Prompt Text

BMC shows a prompt above the tip buttons. Customize it:

```
Every potion you buy fuels another spell guide, another ritual, another resource for our community of practitioners. Your support keeps the magick flowing. 🌙
```

---

## Optional: Supporter Wall Message

If BMC displays a wall of recent supporters, set the default message prompt to:

```
Leave a message with your potion (optional — share your favorite spell, a moon wish, or just say hello!)
```

---

## Revenue Expectations

| Scenario | Monthly Tips | Annual |
|----------|-------------|--------|
| Conservative (10 tips/mo avg 1.5 potions) | $45 | $540 |
| Moderate (25 tips/mo avg 2 potions) | $150 | $1,800 |
| Active (50 tips/mo avg 2 potions) | $300 | $3,600 |

After BMC's 5% fee. Tips have the highest margin of any BMC revenue stream.

---

## Setup Checklist

- [x] Change coffee item name to "potion" — **DONE** via `automation/configure_tips.py` (verified: `tag_alternative: potion`)
- [ ] Set price per potion to $3 — **MANUAL** (see note below)
- [ ] Set suggested amounts to 1, 3, 5 — **NOT CONFIGURABLE** (BMC uses fixed +10/+25/+50 multiplier buttons)
- [ ] Paste thank-you message — **MANUAL** (see note below)
- [ ] Customize tip page prompt text — **NOT AVAILABLE** in current BMC UI
- [x] Preview the tip page to verify "Buy me a potion" displays correctly — **DONE**

---

## Automation Notes (2026-03-02)

### What was automated
- Item name "potion" set via Edit Page modal → verified via `[data-page]` JSON

### What requires manual action
**Coffee price ($5 → $3):** The `coffee_price` field is stored server-side but NOT
exposed in any BMC web UI (Edit Page modal, Settings, Extras, or any studio page).
The `update_page` API accepts the field in the payload but silently ignores changes.
Exhaustive search of all 32 studio URLs confirmed no price input exists.
Options: contact BMC support, check BMC mobile app, or re-do onboarding.

**Thank-you message:** The `thank_you_message` field is `null` in the database and
not editable through any discovered UI or API endpoint. The default message
("Thank you for being part of this. Truly.") appears to be platform-provided.
Contact BMC support to set a custom thank-you message.

### Scripts created
- `automation/configure_tips.py` — Sets item name via Edit Page modal
- `automation/discover_tips_settings.py` — Discovery/catalog of all BMC settings pages
- `automation/find_price_setting.py` — Exhaustive search of all studio URLs for price fields
- `automation/check_settings_pricing.py` — Deep scan of Settings + Extras pages
