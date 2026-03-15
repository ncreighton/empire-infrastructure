"""Publishing playbooks — per-category content upload patterns.

Each playbook describes how to publish content on a category of platforms:
what fields to fill, what files to upload, whether pricing is required,
whether the platform queues submissions for review, etc.

Zero AI cost — purely algorithmic knowledge base.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─── Field definition ────────────────────────────────────────────────────────


@dataclass
class PublishingField:
    """A single field to fill when publishing content on a platform."""

    name: str
    """Logical name: title, description, price, category, tags, file, cover_image, etc."""

    field_type: str
    """Input type: text, textarea, number, select, file_upload, tag_input, checkbox."""

    required: bool = True
    max_length: int = 0
    """0 = no platform-imposed limit."""

    hint: str = ""
    """Human-readable guidance for the browser agent."""


# ─── Playbook definition ──────────────────────────────────────────────────────


@dataclass
class PublishingPlaybook:
    """Defines how to publish content on a platform category.

    The ContentPublisher uses this to know:
    - Which fields to fill
    - Whether to upload a product file and/or a cover image
    - Whether to set a price
    - Which file formats are accepted
    - Whether the platform queues the listing for review
    """

    category: str
    """Matches PlatformCategory.value strings (e.g. 'ai_marketplace')."""

    content_type: str
    """What gets published: product | workflow | model | prompt | course | post."""

    fields: list[PublishingField] = field(default_factory=list)

    has_preview: bool = False
    """Platform supports a short preview / excerpt."""

    has_pricing: bool = True
    """Platform allows sellers to set a price (0 = free listing)."""

    requires_review: bool = False
    """True when the platform manually reviews submissions before going live."""

    max_file_size_mb: int = 50
    accepted_formats: list[str] = field(default_factory=list)
    """File extensions accepted for the main product file, e.g. ['.zip', '.json']."""

    upload_page_hint: str = ""
    """Natural-language hint to help the browser agent find the upload/create page."""

    submit_button_hint: str = ""
    """Natural-language hint to identify the final publish/submit button."""


# ─── Playbook registry ────────────────────────────────────────────────────────


_PLAYBOOKS: dict[str, PublishingPlaybook] = {}


def _register(pb: PublishingPlaybook) -> PublishingPlaybook:
    _PLAYBOOKS[pb.category] = pb
    return pb


# ─── ai_marketplace ──────────────────────────────────────────────────────────

_register(PublishingPlaybook(
    category="ai_marketplace",
    content_type="product",
    fields=[
        PublishingField(
            name="title",
            field_type="text",
            required=True,
            max_length=100,
            hint="Short headline describing the AI tool or agent.",
        ),
        PublishingField(
            name="description",
            field_type="textarea",
            required=True,
            max_length=2000,
            hint="Detailed description: what it does, who it's for, use cases.",
        ),
        PublishingField(
            name="category",
            field_type="select",
            required=False,
            hint="Choose the closest AI tool category available (e.g. Productivity, Developer).",
        ),
        PublishingField(
            name="tags",
            field_type="tag_input",
            required=False,
            hint="Comma-separated keywords: ai, automation, productivity.",
        ),
        PublishingField(
            name="price",
            field_type="number",
            required=False,
            hint="Monthly or one-time price in USD. Leave 0 for free.",
        ),
        PublishingField(
            name="cover_image",
            field_type="file_upload",
            required=False,
            hint="Upload a 1200x630 PNG or JPG thumbnail/cover image.",
        ),
    ],
    has_preview=True,
    has_pricing=True,
    requires_review=True,
    max_file_size_mb=20,
    accepted_formats=[".zip", ".py", ".json", ".tar.gz"],
    upload_page_hint=(
        "Look for a 'Submit your GPT', 'Add tool', 'Create listing', "
        "'New product', or 'Submit' button in the navigation or dashboard."
    ),
    submit_button_hint=(
        "Click the 'Submit', 'Publish', 'Save', or 'Create' button to finalize."
    ),
))

# ─── digital_product ─────────────────────────────────────────────────────────

_register(PublishingPlaybook(
    category="digital_product",
    content_type="product",
    fields=[
        PublishingField(
            name="title",
            field_type="text",
            required=True,
            max_length=150,
            hint="Product name as it will appear in the store.",
        ),
        PublishingField(
            name="description",
            field_type="textarea",
            required=True,
            max_length=5000,
            hint=(
                "Full product description. Include what's inside the download, "
                "who it's for, and the key benefits."
            ),
        ),
        PublishingField(
            name="price",
            field_type="number",
            required=True,
            hint="Price in USD. Use 0 for a free / pay-what-you-want product.",
        ),
        PublishingField(
            name="category",
            field_type="select",
            required=False,
            hint="Select the product category (e.g. Templates, eBooks, Software).",
        ),
        PublishingField(
            name="tags",
            field_type="tag_input",
            required=False,
            max_length=200,
            hint="Comma-separated tags to improve discoverability.",
        ),
        PublishingField(
            name="file",
            field_type="file_upload",
            required=True,
            hint="Upload the main product file (ZIP, PDF, etc.).",
        ),
        PublishingField(
            name="cover_image",
            field_type="file_upload",
            required=False,
            hint="Upload a cover/thumbnail image (recommended 1200x630 or square).",
        ),
        PublishingField(
            name="preview_text",
            field_type="textarea",
            required=False,
            max_length=500,
            hint="Short excerpt shown on the product card before purchase.",
        ),
    ],
    has_preview=True,
    has_pricing=True,
    requires_review=False,
    max_file_size_mb=200,
    accepted_formats=[".zip", ".pdf", ".docx", ".epub", ".png", ".jpg"],
    upload_page_hint=(
        "Look for a 'New product', 'Add product', 'Upload', 'Sell a product', "
        "or '+' button in the dashboard."
    ),
    submit_button_hint=(
        "Click 'Publish', 'Create product', 'Save and continue', or 'Go live'."
    ),
))

# ─── workflow_marketplace ────────────────────────────────────────────────────

_register(PublishingPlaybook(
    category="workflow_marketplace",
    content_type="workflow",
    fields=[
        PublishingField(
            name="title",
            field_type="text",
            required=True,
            max_length=100,
            hint="Workflow name (e.g. 'AI Blog Post Generator').",
        ),
        PublishingField(
            name="description",
            field_type="textarea",
            required=True,
            max_length=3000,
            hint=(
                "Describe what the workflow does, what triggers it, "
                "what apps it connects, and what the output is."
            ),
        ),
        PublishingField(
            name="tags",
            field_type="tag_input",
            required=False,
            hint="Tags: automation, n8n, ai, content, etc.",
        ),
        PublishingField(
            name="category",
            field_type="select",
            required=False,
            hint="Choose the closest workflow category (Marketing, DevOps, AI, etc.).",
        ),
        PublishingField(
            name="price",
            field_type="number",
            required=False,
            hint="Price in USD. Use 0 for a free workflow.",
        ),
        PublishingField(
            name="file",
            field_type="file_upload",
            required=True,
            hint="Upload the exported workflow JSON file.",
        ),
        PublishingField(
            name="cover_image",
            field_type="file_upload",
            required=False,
            hint="Upload a screenshot or diagram of the workflow (PNG/JPG).",
        ),
    ],
    has_preview=False,
    has_pricing=True,
    requires_review=False,
    max_file_size_mb=10,
    accepted_formats=[".json"],
    upload_page_hint=(
        "Look for a 'Share workflow', 'Contribute', 'Upload', or 'Add workflow' "
        "link in the navigation or profile menu."
    ),
    submit_button_hint=(
        "Click 'Publish', 'Submit', 'Share', or 'Save'."
    ),
))

# ─── code_repository ─────────────────────────────────────────────────────────

_register(PublishingPlaybook(
    category="code_repository",
    content_type="product",
    fields=[
        PublishingField(
            name="title",
            field_type="text",
            required=True,
            max_length=100,
            hint="Repository or release name.",
        ),
        PublishingField(
            name="description",
            field_type="textarea",
            required=True,
            max_length=2000,
            hint=(
                "README-style description: what it is, how to install, "
                "quick start example."
            ),
        ),
        PublishingField(
            name="tags",
            field_type="tag_input",
            required=False,
            hint="Topics/tags for the repository.",
        ),
        PublishingField(
            name="file",
            field_type="file_upload",
            required=False,
            hint="Upload a release asset ZIP if this is a release-based deploy.",
        ),
        PublishingField(
            name="cover_image",
            field_type="file_upload",
            required=False,
            hint="Social preview image (1280x640 PNG recommended).",
        ),
    ],
    has_preview=False,
    has_pricing=False,
    requires_review=False,
    max_file_size_mb=100,
    accepted_formats=[".zip", ".tar.gz", ".whl", ".tgz"],
    upload_page_hint=(
        "Navigate to the repository settings or 'Releases' page to create a new release "
        "or deploy the project template."
    ),
    submit_button_hint=(
        "Click 'Publish release', 'Deploy', 'Create', or 'Save changes'."
    ),
))

# ─── prompt_marketplace ──────────────────────────────────────────────────────

_register(PublishingPlaybook(
    category="prompt_marketplace",
    content_type="prompt",
    fields=[
        PublishingField(
            name="title",
            field_type="text",
            required=True,
            max_length=80,
            hint="Short, benefit-driven title for the prompt (e.g. 'SEO Blog Post Writer').",
        ),
        PublishingField(
            name="description",
            field_type="textarea",
            required=True,
            max_length=1000,
            hint=(
                "Describe what the prompt does, what AI model it works best with, "
                "and what kind of output it produces."
            ),
        ),
        PublishingField(
            name="prompt_text",
            field_type="textarea",
            required=True,
            max_length=10000,
            hint="The full prompt text. Include placeholders like [TOPIC] if applicable.",
        ),
        PublishingField(
            name="category",
            field_type="select",
            required=False,
            hint="Select a category (Writing, Marketing, Coding, etc.).",
        ),
        PublishingField(
            name="price",
            field_type="number",
            required=False,
            hint="Price in USD. Many prompt marketplaces use $1–$9 range.",
        ),
        PublishingField(
            name="preview_text",
            field_type="textarea",
            required=False,
            max_length=300,
            hint="Example output snippet shown before purchase.",
        ),
        PublishingField(
            name="tags",
            field_type="tag_input",
            required=False,
            hint="Keywords: writing, seo, gpt-4, content-creation, etc.",
        ),
        PublishingField(
            name="cover_image",
            field_type="file_upload",
            required=False,
            hint="Optional thumbnail (square PNG recommended).",
        ),
    ],
    has_preview=True,
    has_pricing=True,
    requires_review=False,
    max_file_size_mb=5,
    accepted_formats=[".txt", ".md"],
    upload_page_hint=(
        "Look for a 'Sell prompt', 'Submit', 'New prompt', 'Add listing', "
        "or '+' button in the dashboard."
    ),
    submit_button_hint=(
        "Click 'Publish', 'List prompt', 'Submit', or 'Save'."
    ),
))

# ─── 3d_models ───────────────────────────────────────────────────────────────

_register(PublishingPlaybook(
    category="3d_models",
    content_type="product",
    fields=[
        PublishingField(
            name="title",
            field_type="text",
            required=True,
            max_length=100,
            hint="Model name (descriptive: material, style, intended use).",
        ),
        PublishingField(
            name="description",
            field_type="textarea",
            required=True,
            max_length=3000,
            hint=(
                "Describe the model: dimensions, polygon count, "
                "recommended print settings, intended use case."
            ),
        ),
        PublishingField(
            name="category",
            field_type="select",
            required=True,
            hint="Choose the closest model category (e.g. Figurines, Tools, Jewelry).",
        ),
        PublishingField(
            name="price",
            field_type="number",
            required=False,
            hint="Price in USD. Use 0 for a free model.",
        ),
        PublishingField(
            name="tags",
            field_type="tag_input",
            required=False,
            hint="e.g. 3d-printing, stl, figurine, articulated.",
        ),
        PublishingField(
            name="file",
            field_type="file_upload",
            required=True,
            hint="Upload the STL, OBJ, or 3MF model file.",
        ),
        PublishingField(
            name="cover_image",
            field_type="file_upload",
            required=True,
            hint="Upload a render or photo of the printed model (PNG/JPG).",
        ),
        PublishingField(
            name="license",
            field_type="select",
            required=False,
            hint="Choose a license (Creative Commons, Commercial, etc.).",
        ),
    ],
    has_preview=False,
    has_pricing=True,
    requires_review=False,
    max_file_size_mb=250,
    accepted_formats=[".stl", ".obj", ".3mf", ".step", ".zip"],
    upload_page_hint=(
        "Look for an 'Upload model', 'Add model', 'New design', or 'Contribute' "
        "button on the homepage or user dashboard."
    ),
    submit_button_hint=(
        "Click 'Publish', 'Upload', 'Submit', or 'Save model'."
    ),
))

# ─── education ───────────────────────────────────────────────────────────────

_register(PublishingPlaybook(
    category="education",
    content_type="course",
    fields=[
        PublishingField(
            name="title",
            field_type="text",
            required=True,
            max_length=100,
            hint="Course title — clear and benefit-driven.",
        ),
        PublishingField(
            name="description",
            field_type="textarea",
            required=True,
            max_length=5000,
            hint="What students will learn, course curriculum overview, prerequisites.",
        ),
        PublishingField(
            name="category",
            field_type="select",
            required=False,
            hint="Select the closest course topic (Development, Business, Design, etc.).",
        ),
        PublishingField(
            name="price",
            field_type="number",
            required=False,
            hint="Course price in USD.",
        ),
        PublishingField(
            name="tags",
            field_type="tag_input",
            required=False,
            hint="Course topics and keywords.",
        ),
        PublishingField(
            name="cover_image",
            field_type="file_upload",
            required=False,
            hint="Upload a course thumbnail (480x270 or 16:9 ratio recommended).",
        ),
    ],
    has_preview=True,
    has_pricing=True,
    requires_review=True,
    max_file_size_mb=500,
    accepted_formats=[".mp4", ".mov", ".zip", ".pdf"],
    upload_page_hint=(
        "Look for a 'Create course', 'New course', or 'Add course' button "
        "in the instructor dashboard."
    ),
    submit_button_hint=(
        "Click 'Submit for review', 'Publish', or 'Save draft'."
    ),
))


# ─── Lookup helpers ───────────────────────────────────────────────────────────


def get_publishing_playbook(category: str) -> PublishingPlaybook | None:
    """Get publishing playbook for a platform category string.

    Args:
        category: PlatformCategory.value string, e.g. 'ai_marketplace'.

    Returns:
        Matching PlayBook or None if the category has no defined playbook.
    """
    return _PLAYBOOKS.get(category)


def get_playbook_for_platform(platform_id: str) -> PublishingPlaybook | None:
    """Get publishing playbook using a platform's registered category.

    Looks up the PlatformConfig from the platform registry, then retrieves
    the playbook for its category.

    Args:
        platform_id: Platform ID as registered in platforms.py.

    Returns:
        Matching PlayBook or None.
    """
    from openclaw.knowledge.platforms import get_platform

    platform = get_platform(platform_id)
    if not platform:
        return None
    return get_publishing_playbook(platform.category.value)


def get_all_playbook_categories() -> list[str]:
    """Return all registered playbook category names."""
    return list(_PLAYBOOKS.keys())
