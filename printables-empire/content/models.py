"""Pydantic models for all content types."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    ARTICLE = "article"
    REVIEW = "review"
    LISTING = "listing"
    POST = "post"


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ContentStatus(str, Enum):
    DRAFT = "draft"
    SCORED = "scored"
    PUBLISHED = "published"
    FAILED = "failed"


class Section(BaseModel):
    heading: str
    body: str
    has_image: bool = False
    image_path: Optional[str] = None


class Article(BaseModel):
    title: str
    slug: str = ""
    content_type: ContentType = ContentType.ARTICLE
    difficulty: Difficulty = Difficulty.BEGINNER
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    intro: str = ""
    sections: list[Section] = Field(default_factory=list)
    conclusion: str = ""
    word_count: int = 0
    hero_image_path: Optional[str] = None
    step_image_paths: list[str] = Field(default_factory=list)
    score: float = 0.0
    status: ContentStatus = ContentStatus.DRAFT
    published_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def full_markdown(self) -> str:
        parts = [f"# {self.title}\n", self.intro, ""]
        for section in self.sections:
            parts.append(f"## {section.heading}\n")
            parts.append(section.body)
            parts.append("")
        if self.conclusion:
            parts.append("## Wrapping Up\n")
            parts.append(self.conclusion)
        return "\n".join(parts)

    def compute_word_count(self) -> int:
        text = self.full_markdown()
        self.word_count = len(text.split())
        return self.word_count

    def to_slug(self) -> str:
        import re
        slug = self.title.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s-]+", "-", slug).strip("-")
        self.slug = slug
        return slug


class Review(BaseModel):
    title: str
    slug: str = ""
    content_type: ContentType = ContentType.REVIEW
    product_name: str = ""
    product_id: str = ""  # key in printer_profiles.yaml
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    overview: str = ""
    specs_section: str = ""
    print_quality_section: str = ""
    ease_of_use_section: str = ""
    value_section: str = ""
    verdict: str = ""
    rating: float = 0.0  # 1-10
    best_for: str = ""
    skip_if: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    word_count: int = 0
    hero_image_path: Optional[str] = None
    comparison_image_path: Optional[str] = None
    score: float = 0.0
    status: ContentStatus = ContentStatus.DRAFT
    published_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def full_markdown(self) -> str:
        parts = [f"# {self.title}\n", self.overview, ""]
        if self.specs_section:
            parts.extend(["## Specs at a Glance\n", self.specs_section, ""])
        if self.print_quality_section:
            parts.extend(["## Print Quality\n", self.print_quality_section, ""])
        if self.ease_of_use_section:
            parts.extend(["## Ease of Use\n", self.ease_of_use_section, ""])
        if self.value_section:
            parts.extend(["## Value for Money\n", self.value_section, ""])
        if self.pros or self.cons:
            parts.append("## Pros & Cons\n")
            if self.pros:
                parts.append("**Pros:**")
                for p in self.pros:
                    parts.append(f"- {p}")
            if self.cons:
                parts.append("\n**Cons:**")
                for c in self.cons:
                    parts.append(f"- {c}")
            parts.append("")
        if self.best_for:
            parts.append(f"**Best For:** {self.best_for}\n")
        if self.skip_if:
            parts.append(f"**Skip If:** {self.skip_if}\n")
        if self.verdict:
            parts.extend(["## The Verdict\n", self.verdict])
        return "\n".join(parts)

    def compute_word_count(self) -> int:
        text = self.full_markdown()
        self.word_count = len(text.split())
        return self.word_count


class Listing(BaseModel):
    title: str
    slug: str = ""
    content_type: ContentType = ContentType.LISTING
    product_name: str = ""
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    print_settings: str = ""
    dimensions: str = ""
    tested_printers: list[str] = Field(default_factory=list)
    file_formats: list[str] = Field(default_factory=lambda: ["STL", "3MF"])
    word_count: int = 0
    score: float = 0.0
    status: ContentStatus = ContentStatus.DRAFT
    published_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def full_markdown(self) -> str:
        parts = [self.description, ""]
        if self.print_settings:
            parts.extend(["## Print Settings\n", self.print_settings, ""])
        if self.tested_printers:
            parts.append(f"**Tested on:** {', '.join(self.tested_printers)}\n")
        if self.file_formats:
            parts.append(f"**Formats:** {', '.join(self.file_formats)}\n")
        if self.dimensions:
            parts.append(f"**Dimensions:** {self.dimensions}\n")
        return "\n".join(parts)


class Post(BaseModel):
    title: str
    slug: str = ""
    content_type: ContentType = ContentType.POST
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    body: str = ""
    word_count: int = 0
    score: float = 0.0
    status: ContentStatus = ContentStatus.DRAFT
    published_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def full_markdown(self) -> str:
        return f"# {self.title}\n\n{self.body}"


class ContentPiece(BaseModel):
    """Wrapper for any content type with metadata."""
    content_type: ContentType
    article: Optional[Article] = None
    review: Optional[Review] = None
    listing: Optional[Listing] = None
    post: Optional[Post] = None
    voice_profile: str = "maker_mentor"
    cost_usd: float = 0.0
    generation_time_sec: float = 0.0

    @property
    def content(self) -> Article | Review | Listing | Post:
        if self.content_type == ContentType.ARTICLE:
            return self.article
        elif self.content_type == ContentType.REVIEW:
            return self.review
        elif self.content_type == ContentType.LISTING:
            return self.listing
        elif self.content_type == ContentType.POST:
            return self.post

    @property
    def title(self) -> str:
        return self.content.title if self.content else ""

    @property
    def score(self) -> float:
        return self.content.score if self.content else 0.0
