"""Generate placeholder PDFs for the 6 BMC shop products.

Each PDF is a single page with the product title and a placeholder message.
The user will replace these with real product files later.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "placeholders"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = [
    {
        "filename": "beginners_spell_book.pdf",
        "title": "Beginner's Spell Book",
        "subtitle": "8 Essential Spells for New Practitioners",
        "pages": "24 pages, fillable PDF",
    },
    {
        "filename": "moon_phase_journal.pdf",
        "title": "Moon Phase Journal",
        "subtitle": "Monthly Lunar Tracker & Ritual Planner",
        "pages": "56 pages, interactive fillable PDF",
    },
    {
        "filename": "grimoire_herbs_crystals.pdf",
        "title": "Grimoire Collection",
        "subtitle": "Herbs & Crystals Quick Reference Guide",
        "pages": "38 pages, illustrated PDF",
    },
    {
        "filename": "samhain_ritual_kit.pdf",
        "title": "Samhain Complete Ritual Kit",
        "subtitle": "Ancestor Honor & Veil Working",
        "pages": "32 pages across 4 documents",
    },
    {
        "filename": "wheel_of_the_year_bundle.pdf",
        "title": "Wheel of the Year Complete Bundle",
        "subtitle": "All 8 Sabbat Ritual Kits",
        "pages": "240+ pages, 8 separate documents + master index",
    },
    {
        "filename": "complete_digital_grimoire_library.pdf",
        "title": "The Complete Digital Grimoire Library",
        "subtitle": "30+ Professional PDFs",
        "pages": "800+ pages, 30+ files organized in folders",
    },
]


def create_placeholder(product):
    path = OUT_DIR / product["filename"]
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    # Purple gradient background (solid)
    c.setFillColorRGB(0.2, 0.05, 0.3)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    # Title
    c.setFillColorRGB(0.85, 0.75, 0.5)  # Gold
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(width / 2, height - 2.5 * inch, product["title"])

    # Subtitle
    c.setFont("Helvetica", 18)
    c.drawCentredString(width / 2, height - 3.2 * inch, product["subtitle"])

    # Divider line
    c.setStrokeColorRGB(0.85, 0.75, 0.5)
    c.setLineWidth(1)
    c.line(2 * inch, height - 4 * inch, width - 2 * inch, height - 4 * inch)

    # Placeholder message
    c.setFillColorRGB(0.8, 0.8, 0.8)
    c.setFont("Helvetica", 14)
    c.drawCentredString(width / 2, height - 5 * inch,
                        "This is a placeholder file.")
    c.drawCentredString(width / 2, height - 5.4 * inch,
                        "The full product will be uploaded shortly.")

    # Product info
    c.setFont("Helvetica", 11)
    c.setFillColorRGB(0.6, 0.6, 0.6)
    c.drawCentredString(width / 2, height - 6.2 * inch,
                        f"Final product: {product['pages']}")

    # Footer
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawCentredString(width / 2, 1 * inch,
                        "WitchcraftForBeginners.com  |  VelvetVeil Printables")

    c.save()
    return path


if __name__ == "__main__":
    print("Creating placeholder PDFs...")
    for product in PRODUCTS:
        path = create_placeholder(product)
        print(f"  Created: {path.name}")
    print(f"\nAll {len(PRODUCTS)} placeholders saved to: {OUT_DIR}")
