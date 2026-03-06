"""
Meta Optimizer — Title, description, and Open Graph tag optimization.
"""

import logging
import re
from typing import Dict, List, Optional

from systems.site_evolution.utils import load_site_config

log = logging.getLogger(__name__)


class MetaOptimizer:
    """Optimize meta tags for search engines and social sharing."""

    def optimize_title(self, title: str, site_name: str,
                       max_length: int = 60) -> str:
        """Optimize a page title for SEO (55-60 chars, keyword-front)."""
        # Remove site name suffix if present
        cleaned = re.sub(r'\s*[-|]\s*' + re.escape(site_name) + r'$', '', title)
        cleaned = cleaned.strip()

        # Truncate if needed (preserving whole words)
        suffix = f" | {site_name}"
        available = max_length - len(suffix)

        if len(cleaned) > available:
            words = cleaned.split()
            truncated = ""
            for word in words:
                if len(truncated) + len(word) + 1 <= available:
                    truncated = f"{truncated} {word}" if truncated else word
                else:
                    break
            cleaned = truncated

        return f"{cleaned}{suffix}"

    def generate_meta_description(self, title: str, content: str = "",
                                  keywords: List[str] = None,
                                  max_length: int = 160) -> str:
        """Generate a 150-160 char meta description with CTA."""
        # Strip HTML tags
        clean = re.sub(r'<[^>]+>', '', content)
        clean = re.sub(r'\s+', ' ', clean).strip()

        if not clean:
            # Generate from title
            return f"Discover {title.lower()}. Expert guide with actionable tips."[:max_length]

        # Extract first meaningful sentence
        sentences = re.split(r'[.!?]+', clean)
        desc = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
            if len(desc) + len(sentence) + 2 <= max_length - 15:
                desc = f"{desc}. {sentence}" if desc else sentence
            else:
                break

        if not desc:
            desc = clean[:max_length - 20]

        # Add CTA if space
        if len(desc) < max_length - 15:
            desc += " Learn more."

        return desc[:max_length]

    def generate_og_tags(self, site_slug: str, post: Dict) -> str:
        """Generate Open Graph + Twitter Card meta tags as HTML."""
        config = load_site_config(site_slug)
        domain = config.get("domain", "example.com")
        brand = config.get("name", site_slug)

        title = post.get("title", "")
        if isinstance(title, dict):
            title = title.get("rendered", "")

        description = post.get("excerpt", "")
        if isinstance(description, dict):
            description = description.get("rendered", "")
        description = re.sub(r'<[^>]+>', '', description).strip()[:200]

        image = post.get("featured_image_url", f"https://{domain}/wp-content/uploads/default-og.png")
        url = post.get("link", f"https://{domain}")

        tags = f"""<!-- Open Graph -->
<meta property="og:type" content="article">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:image" content="{image}">
<meta property="og:url" content="{url}">
<meta property="og:site_name" content="{brand}">

<!-- Twitter Card -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{description}">
<meta name="twitter:image" content="{image}">"""
        return tags

    def audit_meta(self, site_slug: str, post: Dict) -> Dict:
        """Audit meta tags on a post and return recommendations."""
        issues = []
        score = 100

        title = post.get("title", "")
        if isinstance(title, dict):
            title = title.get("rendered", "")

        # Title checks
        if not title:
            issues.append({"type": "critical", "msg": "Missing title"})
            score -= 30
        elif len(title) > 65:
            issues.append({"type": "warning", "msg": f"Title too long ({len(title)} chars)"})
            score -= 10
        elif len(title) < 20:
            issues.append({"type": "warning", "msg": "Title too short"})
            score -= 5

        # Description checks
        excerpt = post.get("excerpt", "")
        if isinstance(excerpt, dict):
            excerpt = excerpt.get("rendered", "")
        excerpt_clean = re.sub(r'<[^>]+>', '', excerpt).strip()

        if not excerpt_clean:
            issues.append({"type": "warning", "msg": "Missing meta description"})
            score -= 15
        elif len(excerpt_clean) > 160:
            issues.append({"type": "info", "msg": "Meta description could be shorter"})
            score -= 5

        # Featured image
        if not post.get("featured_media") and not post.get("featured_image_url"):
            issues.append({"type": "warning", "msg": "No featured image (affects OG sharing)"})
            score -= 10

        return {
            "score": max(0, score),
            "issues": issues,
            "title_length": len(title),
            "description_length": len(excerpt_clean),
        }

    def optimize_all_posts(self, site_slug: str,
                           dry_run: bool = True) -> Dict:
        """Batch audit all posts on a site and generate optimization queue."""
        from systems.site_evolution.deployer.wp_deployer import _wp_request

        try:
            posts = _wp_request(
                site_slug, "GET",
                "wp/v2/posts?per_page=100&status=publish"
                "&_fields=id,title,excerpt,link,featured_media,date,modified"
            )
            if not isinstance(posts, list):
                posts = []
        except Exception as e:
            return {"error": str(e), "posts_audited": 0}

        results = []
        for post in posts:
            audit = self.audit_meta(site_slug, post)
            if audit["score"] < 80:
                results.append({
                    "post_id": post.get("id"),
                    "title": post.get("title", {}).get("rendered", ""),
                    "score": audit["score"],
                    "issues": audit["issues"],
                })

        return {
            "site": site_slug,
            "posts_audited": len(posts),
            "posts_needing_work": len(results),
            "items": results[:50],
        }
