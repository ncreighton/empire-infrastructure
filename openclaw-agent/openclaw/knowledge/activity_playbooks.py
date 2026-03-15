"""Activity playbooks — defines organic human-like behavior patterns per platform category.

Each playbook contains a set of realistic Activity definitions with natural-language
descriptions that the browser-use Agent can execute directly. Activities are weighted
so browsing/viewing happens more often than liking/following, mirroring real user behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openclaw.models import PlatformCategory


@dataclass
class Activity:
    """A single organic activity to perform on a platform."""

    activity_type: str
    """browse_feed, view_profile, search, like, follow, comment, scroll, etc."""

    description: str
    """Natural language instruction for the browser agent to execute."""

    duration_seconds: tuple[int, int]
    """(min, max) seconds to spend on this activity."""

    requires_login: bool = True
    weight: float = 1.0
    """Relative probability weight for random selection. Higher = more frequent."""


@dataclass
class ActivityPlaybook:
    """Defines organic behavior patterns for a platform category."""

    category: str
    activities: list[Activity]
    session_duration: tuple[int, int]
    """(min, max) minutes for a full session."""

    activities_per_session: tuple[int, int]
    """(min, max) number of activities to run per session."""

    cooldown_hours: int = 24
    """Minimum hours between sessions on the same platform."""


# ─────────────────────────────────────────────────────────────────────────────
# Playbook definitions
# ─────────────────────────────────────────────────────────────────────────────

_AI_MARKETPLACE = ActivityPlaybook(
    category=PlatformCategory.AI_MARKETPLACE.value,
    activities=[
        Activity(
            activity_type="browse_feed",
            description=(
                "Navigate to the homepage or explore/discover section. "
                "Scroll slowly through the featured tools and trending AI products. "
                "Pause on 5-8 items to read their titles and short descriptions. "
                "Do not click anything — just browse like a curious visitor."
            ),
            duration_seconds=(45, 120),
            weight=3.0,
        ),
        Activity(
            activity_type="search",
            description=(
                "Find the search bar on the platform. "
                "Type one of these queries (pick one at random): "
                "'automation tools', 'writing assistant', 'image generator', 'productivity AI'. "
                "Browse the first page of search results, reading titles and descriptions."
            ),
            duration_seconds=(30, 90),
            weight=2.5,
        ),
        Activity(
            activity_type="view_profile",
            description=(
                "Click on a random creator or product listing that looks interesting. "
                "View their profile page or product detail page. "
                "Read their bio and look at the products/tools they offer. "
                "Spend at least 20 seconds on the page before returning."
            ),
            duration_seconds=(20, 60),
            weight=2.0,
        ),
        Activity(
            activity_type="like",
            description=(
                "Find an interesting AI tool or product on the current page or feed. "
                "Click the like, upvote, star, or bookmark button on one item. "
                "Only do this once — do not like everything on the page."
            ),
            duration_seconds=(15, 40),
            weight=1.0,
        ),
        Activity(
            activity_type="scroll",
            description=(
                "Scroll slowly down through the current page. "
                "Pause every few items as if reading. "
                "Scroll back up partway and then continue down. "
                "This simulates natural reading behavior."
            ),
            duration_seconds=(20, 50),
            weight=1.5,
        ),
    ],
    session_duration=(8, 20),
    activities_per_session=(3, 6),
    cooldown_hours=20,
)

_DIGITAL_PRODUCT = ActivityPlaybook(
    category=PlatformCategory.DIGITAL_PRODUCT.value,
    activities=[
        Activity(
            activity_type="browse_feed",
            description=(
                "Go to the main marketplace or discover page. "
                "Browse through new arrivals or trending digital products. "
                "Scroll through 8-12 products, pausing to read product names and prices. "
                "Behave like a shopper browsing casually."
            ),
            duration_seconds=(40, 100),
            weight=3.0,
        ),
        Activity(
            activity_type="view_profile",
            description=(
                "Click on a random seller's profile or storefront. "
                "Look at what products they sell, their sales count, and their bio. "
                "Spend 15-30 seconds reviewing their page before going back."
            ),
            duration_seconds=(15, 45),
            weight=2.0,
        ),
        Activity(
            activity_type="search",
            description=(
                "Use the search bar to look for products. "
                "Try one of these searches (choose one): "
                "'printable planner', 'digital template', 'ebook', 'notion template', 'bundle'. "
                "Look through the first page of results."
            ),
            duration_seconds=(30, 80),
            weight=2.5,
        ),
        Activity(
            activity_type="like",
            description=(
                "Find a product that looks appealing. "
                "Click the wishlist, save, favorite, or heart button on it. "
                "Only save one product per activity."
            ),
            duration_seconds=(15, 35),
            weight=1.0,
        ),
        Activity(
            activity_type="view_product",
            description=(
                "Click into a product listing to view its full detail page. "
                "Read the description, look at the preview images, and check the price. "
                "Spend 20-40 seconds on the product page."
            ),
            duration_seconds=(20, 50),
            weight=2.0,
        ),
    ],
    session_duration=(8, 22),
    activities_per_session=(3, 6),
    cooldown_hours=22,
)

_WORKFLOW_MARKETPLACE = ActivityPlaybook(
    category=PlatformCategory.WORKFLOW_MARKETPLACE.value,
    activities=[
        Activity(
            activity_type="browse_feed",
            description=(
                "Navigate to the templates or community section of the platform. "
                "Browse through workflow templates and automation examples. "
                "Scroll through 6-10 items, reading their names and descriptions."
            ),
            duration_seconds=(35, 90),
            weight=3.0,
        ),
        Activity(
            activity_type="search",
            description=(
                "Use the search or filter to find specific workflow types. "
                "Search for one of these (pick one): "
                "'email automation', 'social media', 'data sync', 'webhook', 'AI workflow'. "
                "Browse the results."
            ),
            duration_seconds=(25, 70),
            weight=2.5,
        ),
        Activity(
            activity_type="view_profile",
            description=(
                "Click on a community member's profile or a template creator's page. "
                "View their published workflows and any bio information. "
                "Spend 15-25 seconds reviewing their contributions."
            ),
            duration_seconds=(15, 35),
            weight=1.5,
        ),
        Activity(
            activity_type="view_template",
            description=(
                "Click on a workflow template that looks interesting. "
                "Read its description, check the integrations it uses, and view the node count. "
                "Spend 20-45 seconds on the template detail page."
            ),
            duration_seconds=(20, 50),
            weight=2.0,
        ),
        Activity(
            activity_type="like",
            description=(
                "Find a useful-looking workflow or template. "
                "Click the like, star, or upvote button on it. "
                "Only do this for one item."
            ),
            duration_seconds=(10, 30),
            weight=1.0,
        ),
    ],
    session_duration=(7, 18),
    activities_per_session=(2, 5),
    cooldown_hours=24,
)

_CODE_REPOSITORY = ActivityPlaybook(
    category=PlatformCategory.CODE_REPOSITORY.value,
    activities=[
        Activity(
            activity_type="browse_feed",
            description=(
                "Go to the explore or trending section of the platform. "
                "Browse through trending repositories or projects. "
                "Scroll through 6-10 repos, reading their names, descriptions, and star counts."
            ),
            duration_seconds=(40, 100),
            weight=3.0,
        ),
        Activity(
            activity_type="view_profile",
            description=(
                "Click on a developer's profile or username you see on the page. "
                "Look at their public repositories and any bio information. "
                "Spend 15-30 seconds on their profile."
            ),
            duration_seconds=(15, 40),
            weight=2.0,
        ),
        Activity(
            activity_type="search",
            description=(
                "Use the search bar to look for libraries or tools. "
                "Search for one of these (choose one): "
                "'python automation', 'react components', 'AI tools', 'CLI tools', 'open source'. "
                "Browse through the first page of results."
            ),
            duration_seconds=(30, 75),
            weight=2.5,
        ),
        Activity(
            activity_type="like",
            description=(
                "Find an interesting project on the current page. "
                "Click the star button to star that repository. "
                "Only star one repository per activity."
            ),
            duration_seconds=(15, 35),
            weight=1.0,
        ),
        Activity(
            activity_type="view_repo",
            description=(
                "Click into a repository to view its README and code. "
                "Read the README description and look at the file structure. "
                "Check the stars, forks, and recent activity. "
                "Spend 20-50 seconds on the repository page."
            ),
            duration_seconds=(20, 55),
            weight=2.0,
        ),
    ],
    session_duration=(8, 20),
    activities_per_session=(3, 6),
    cooldown_hours=20,
)

_SOCIAL_PLATFORM = ActivityPlaybook(
    category=PlatformCategory.SOCIAL_PLATFORM.value,
    activities=[
        Activity(
            activity_type="browse_feed",
            description=(
                "Go to the main feed, home page, or recent posts section. "
                "Scroll through 8-12 posts or discussions, pausing to read titles. "
                "Behave like someone catching up on their feed."
            ),
            duration_seconds=(45, 110),
            weight=3.5,
        ),
        Activity(
            activity_type="view_post",
            description=(
                "Click on an interesting-looking post or discussion thread. "
                "Read the first few comments or the full post. "
                "Spend 25-50 seconds reading before going back."
            ),
            duration_seconds=(25, 60),
            weight=2.5,
        ),
        Activity(
            activity_type="like",
            description=(
                "Find a post, comment, or discussion that you find interesting. "
                "Click the upvote, like, or heart button on it. "
                "Only like one item per activity."
            ),
            duration_seconds=(10, 30),
            weight=1.5,
        ),
        Activity(
            activity_type="search",
            description=(
                "Use the search feature to find discussions on a topic. "
                "Search for one of these (pick one): "
                "'AI tools', 'productivity', 'automation', 'indie maker', 'side project'. "
                "Browse through the results."
            ),
            duration_seconds=(25, 65),
            weight=2.0,
        ),
        Activity(
            activity_type="view_profile",
            description=(
                "Click on a user's name or avatar to view their profile. "
                "Look at their recent posts and any bio information. "
                "Spend 15-30 seconds on their profile."
            ),
            duration_seconds=(15, 35),
            weight=1.5,
        ),
        Activity(
            activity_type="follow",
            description=(
                "On a user's profile page or next to a post, click the follow button "
                "to follow that user. Only follow one person per activity."
            ),
            duration_seconds=(10, 25),
            weight=0.5,
        ),
    ],
    session_duration=(10, 25),
    activities_per_session=(3, 7),
    cooldown_hours=18,
)

_EDUCATION = ActivityPlaybook(
    category=PlatformCategory.EDUCATION.value,
    activities=[
        Activity(
            activity_type="browse_feed",
            description=(
                "Go to the course discovery or explore section. "
                "Browse through featured or popular courses. "
                "Scroll through 6-10 courses, reading titles, instructors, and ratings."
            ),
            duration_seconds=(35, 90),
            weight=3.0,
        ),
        Activity(
            activity_type="search",
            description=(
                "Use the search bar to look for courses on a topic. "
                "Search for one of these (choose one): "
                "'Python programming', 'AI course', 'marketing', 'freelancing', 'design'. "
                "Browse the first page of course results."
            ),
            duration_seconds=(25, 70),
            weight=2.5,
        ),
        Activity(
            activity_type="view_profile",
            description=(
                "Click on an instructor's name or profile link. "
                "View their profile, courses they teach, and student reviews. "
                "Spend 15-30 seconds reviewing their page."
            ),
            duration_seconds=(15, 40),
            weight=1.5,
        ),
        Activity(
            activity_type="view_course",
            description=(
                "Click on a course listing to view its full detail page. "
                "Read the course description, syllabus overview, and instructor bio. "
                "Check the rating and number of students. Spend 20-45 seconds."
            ),
            duration_seconds=(20, 50),
            weight=2.0,
        ),
        Activity(
            activity_type="like",
            description=(
                "Find a course that looks valuable. "
                "Click the wishlist, save, or bookmark button to save it for later. "
                "Only save one course per activity."
            ),
            duration_seconds=(10, 30),
            weight=1.0,
        ),
    ],
    session_duration=(8, 20),
    activities_per_session=(2, 5),
    cooldown_hours=24,
)

_PROMPT_MARKETPLACE = ActivityPlaybook(
    category=PlatformCategory.PROMPT_MARKETPLACE.value,
    activities=[
        Activity(
            activity_type="browse_feed",
            description=(
                "Navigate to the homepage or featured prompts section. "
                "Browse through popular or new prompt listings. "
                "Scroll through 8-12 prompts, reading their titles and descriptions. "
                "Look at the categories and tags."
            ),
            duration_seconds=(35, 90),
            weight=3.0,
        ),
        Activity(
            activity_type="search",
            description=(
                "Use the search or category filter. "
                "Look for prompts in one of these categories (choose one): "
                "'writing', 'coding', 'marketing', 'image generation', 'business'. "
                "Browse through the results."
            ),
            duration_seconds=(25, 65),
            weight=2.5,
        ),
        Activity(
            activity_type="view_profile",
            description=(
                "Click on a prompt creator's username or profile link. "
                "View their collection of prompts and their creator profile. "
                "Spend 15-25 seconds on their profile page."
            ),
            duration_seconds=(15, 35),
            weight=1.5,
        ),
        Activity(
            activity_type="view_prompt",
            description=(
                "Click on a prompt listing to view its full details. "
                "Read the description, example outputs if available, and reviews. "
                "Spend 20-40 seconds on the prompt detail page."
            ),
            duration_seconds=(20, 45),
            weight=2.0,
        ),
        Activity(
            activity_type="like",
            description=(
                "Find a high-quality or interesting prompt listing. "
                "Click the like, upvote, or favorite button on it. "
                "Only like one item per activity."
            ),
            duration_seconds=(10, 30),
            weight=1.0,
        ),
    ],
    session_duration=(7, 18),
    activities_per_session=(2, 5),
    cooldown_hours=24,
)

_THREE_D_MODELS = ActivityPlaybook(
    category=PlatformCategory.THREE_D_MODELS.value,
    activities=[
        Activity(
            activity_type="browse_feed",
            description=(
                "Go to the explore, trending, or popular models section. "
                "Browse through 3D model listings. "
                "Scroll through 8-12 models, pausing to look at previews and read titles. "
                "Notice the categories, download counts, and ratings."
            ),
            duration_seconds=(40, 100),
            weight=3.0,
        ),
        Activity(
            activity_type="search",
            description=(
                "Use the search bar to look for specific types of 3D models. "
                "Search for one of these (choose one): "
                "'dragon', 'architectural model', 'character', 'miniature', 'geometric'. "
                "Browse through the search results."
            ),
            duration_seconds=(30, 75),
            weight=2.5,
        ),
        Activity(
            activity_type="view_profile",
            description=(
                "Click on a 3D designer's username or profile link. "
                "Look at their portfolio of models and their designer bio. "
                "Spend 15-30 seconds viewing their profile."
            ),
            duration_seconds=(15, 40),
            weight=1.5,
        ),
        Activity(
            activity_type="view_model",
            description=(
                "Click on a 3D model listing to view its detail page. "
                "Look at the preview images, read the description, and check the file format. "
                "Note the download count and any user comments. Spend 20-50 seconds."
            ),
            duration_seconds=(20, 55),
            weight=2.0,
        ),
        Activity(
            activity_type="like",
            description=(
                "Find a 3D model that looks impressive or useful. "
                "Click the like, heart, or favorite button on it. "
                "Only like one model per activity."
            ),
            duration_seconds=(10, 30),
            weight=1.0,
        ),
    ],
    session_duration=(8, 22),
    activities_per_session=(3, 6),
    cooldown_hours=22,
)

# ─────────────────────────────────────────────────────────────────────────────
# Registry and lookup
# ─────────────────────────────────────────────────────────────────────────────

_PLAYBOOKS: dict[str, ActivityPlaybook] = {
    PlatformCategory.AI_MARKETPLACE.value: _AI_MARKETPLACE,
    PlatformCategory.DIGITAL_PRODUCT.value: _DIGITAL_PRODUCT,
    PlatformCategory.WORKFLOW_MARKETPLACE.value: _WORKFLOW_MARKETPLACE,
    PlatformCategory.CODE_REPOSITORY.value: _CODE_REPOSITORY,
    PlatformCategory.SOCIAL_PLATFORM.value: _SOCIAL_PLATFORM,
    PlatformCategory.EDUCATION.value: _EDUCATION,
    PlatformCategory.PROMPT_MARKETPLACE.value: _PROMPT_MARKETPLACE,
    PlatformCategory.THREE_D_MODELS.value: _THREE_D_MODELS,
}


def get_playbook(category: str) -> ActivityPlaybook | None:
    """Get the activity playbook for a platform category string."""
    return _PLAYBOOKS.get(category)


def get_playbook_for_platform(platform_id: str) -> ActivityPlaybook | None:
    """Get playbook using the platform's registered category.

    Returns None if the platform is unknown or has no matching playbook.
    """
    from openclaw.knowledge.platforms import get_platform

    platform = get_platform(platform_id)
    if not platform:
        return None
    return _PLAYBOOKS.get(platform.category.value)


def get_all_categories() -> list[str]:
    """Return all category strings that have a playbook defined."""
    return list(_PLAYBOOKS.keys())
