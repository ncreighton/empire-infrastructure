"""
Content Quality Scorer — Comprehensive per-post scoring: reading level,
heading hierarchy, media ratio, keyword density, link count, structure.
"""

import logging
import math
import re
from typing import Dict, List, Optional

from systems.site_evolution.utils import load_site_config, get_site_brand_name

log = logging.getLogger(__name__)


def _get_posts(site_slug: str, limit: int = 50) -> List[Dict]:
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        return _wp_request(
            site_slug, "GET",
            f"wp/v2/posts?per_page={limit}&status=publish"
            "&_fields=id,title,content,excerpt,link,featured_media,date,modified"
        ) or []
    except Exception as e:
        log.warning("Could not fetch posts for %s: %s", site_slug, e)
        return []


def _extract_title(post: Dict) -> str:
    t = post.get("title", {})
    return t.get("rendered", "") if isinstance(t, dict) else str(t)


def _clean_html(html: str) -> str:
    return re.sub(r'<[^>]+>', ' ', html).strip()


def _count_syllables(word: str) -> int:
    """Simple syllable counter for Flesch-Kincaid."""
    word = word.lower().strip(".,!?;:'\"")
    if len(word) <= 2:
        return 1
    # Count vowel groups
    count = len(re.findall(r'[aeiouy]+', word))
    # Subtract silent e
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


class ContentQualityScorer:
    """Score individual posts and entire sites on content quality."""

    def score_post(self, site_slug: str, post_id: int) -> Dict:
        """Comprehensive quality score for a single post.

        Scoring dimensions (0-100 total):
        - Reading level (0-15): Flesch-Kincaid readability
        - Heading hierarchy (0-15): Proper H1→H2→H3 nesting
        - Media ratio (0-15): Images per 500 words
        - Word count (0-15): Adequate length
        - Link count (0-10): Internal + external links
        - Structure (0-15): Paragraphs, lists, blockquotes
        - Meta quality (0-15): Excerpt, featured image
        """
        from systems.site_evolution.deployer.wp_deployer import _wp_request

        try:
            post = _wp_request(
                site_slug, "GET",
                f"wp/v2/posts/{post_id}?_fields=id,title,content,excerpt,featured_media,link"
            )
        except Exception:
            return {"post_id": post_id, "score": 0, "error": "Could not fetch post"}

        if not post:
            return {"post_id": post_id, "score": 0, "error": "Post not found"}

        content = post.get("content", {})
        if isinstance(content, dict):
            content = content.get("rendered", "")

        clean_text = _clean_html(content)
        words = clean_text.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        scores = {}

        # 1. Reading level (Flesch-Kincaid)
        reading = self.calculate_reading_level(clean_text)
        fk_grade = reading.get("fk_grade", 12)
        if 6 <= fk_grade <= 9:
            scores["reading_level"] = 15  # Ideal for web
        elif 9 < fk_grade <= 12:
            scores["reading_level"] = 10
        elif fk_grade < 6:
            scores["reading_level"] = 8  # Too simple
        else:
            scores["reading_level"] = 5   # Too complex

        # 2. Heading hierarchy
        heading_result = self.check_heading_hierarchy(content)
        scores["heading_hierarchy"] = heading_result.get("score", 0)

        # 3. Media ratio
        media_result = self.check_media_ratio(content, word_count)
        scores["media_ratio"] = media_result.get("score", 0)

        # 4. Word count
        if word_count >= 1500:
            scores["word_count"] = 15
        elif word_count >= 800:
            scores["word_count"] = 10
        elif word_count >= 400:
            scores["word_count"] = 5
        else:
            scores["word_count"] = 2

        # 5. Link count
        links = re.findall(r'<a\s[^>]*href', content, re.IGNORECASE)
        link_count = len(links)
        if link_count >= 5:
            scores["links"] = 10
        elif link_count >= 2:
            scores["links"] = 6
        elif link_count >= 1:
            scores["links"] = 3
        else:
            scores["links"] = 0

        # 6. Structure quality
        has_lists = bool(re.search(r'<[uo]l', content, re.IGNORECASE))
        has_blockquote = bool(re.search(r'<blockquote', content, re.IGNORECASE))
        paragraph_count = content.count('</p>')
        avg_paragraph_length = word_count / max(paragraph_count, 1)

        struct_score = 0
        if has_lists:
            struct_score += 4
        if has_blockquote:
            struct_score += 3
        if 30 <= avg_paragraph_length <= 100:
            struct_score += 4  # Good paragraph length
        if paragraph_count >= 5:
            struct_score += 4
        scores["structure"] = min(15, struct_score)

        # 7. Meta quality
        meta_score = 0
        excerpt = post.get("excerpt", {})
        if isinstance(excerpt, dict):
            excerpt = excerpt.get("rendered", "")
        if excerpt and len(_clean_html(excerpt)) > 30:
            meta_score += 7
        if post.get("featured_media", 0) > 0:
            meta_score += 8
        scores["meta_quality"] = min(15, meta_score)

        total = sum(scores.values())

        return {
            "post_id": post_id,
            "title": _extract_title(post),
            "url": post.get("link", ""),
            "score": total,
            "max_score": 100,
            "dimensions": scores,
            "word_count": word_count,
            "reading_level": reading,
            "heading_check": heading_result,
            "media_check": media_result,
        }

    def score_all_posts(self, site_slug: str, limit: int = 50) -> Dict:
        """Batch score all posts, return ranked list."""
        posts = _get_posts(site_slug, limit)
        scored = []

        for post in posts:
            post_id = post.get("id")
            if not post_id:
                continue

            content = post.get("content", {})
            if isinstance(content, dict):
                content = content.get("rendered", "")

            clean_text = _clean_html(content)
            words = clean_text.split()
            word_count = len(words)

            # Quick scoring (inline to avoid per-post API calls)
            score = 0

            # Word count
            if word_count >= 1500:
                score += 15
            elif word_count >= 800:
                score += 10
            elif word_count >= 400:
                score += 5
            else:
                score += 2

            # Headings
            h2_count = content.count('<h2')
            h3_count = content.count('<h3')
            if h2_count >= 3:
                score += 15
            elif h2_count >= 1:
                score += 8
            else:
                score += 2

            # Images
            img_count = len(re.findall(r'<img\s', content, re.IGNORECASE))
            ideal_images = max(1, word_count // 500)
            if img_count >= ideal_images:
                score += 15
            elif img_count >= 1:
                score += 8
            else:
                score += 0

            # Links
            link_count = len(re.findall(r'<a\s[^>]*href', content, re.IGNORECASE))
            if link_count >= 5:
                score += 10
            elif link_count >= 2:
                score += 5

            # Lists/structure
            if '<ul' in content or '<ol' in content:
                score += 5
            if content.count('</p>') >= 5:
                score += 5

            # Meta
            excerpt = post.get("excerpt", {})
            if isinstance(excerpt, dict):
                excerpt = excerpt.get("rendered", "")
            if excerpt and len(_clean_html(excerpt)) > 30:
                score += 7
            if post.get("featured_media", 0) > 0:
                score += 8

            # Reading level bonus
            reading = self.calculate_reading_level(clean_text)
            fk = reading.get("fk_grade", 12)
            if 6 <= fk <= 10:
                score += 10

            scored.append({
                "id": post.get("id"),
                "title": _extract_title(post),
                "url": post.get("link", ""),
                "score": min(100, score),
                "word_count": word_count,
                "images": img_count,
                "headings": h2_count + h3_count,
                "links": link_count,
            })

        scored.sort(key=lambda s: s["score"], reverse=True)

        avg_score = sum(s["score"] for s in scored) // max(len(scored), 1)

        return {
            "site_slug": site_slug,
            "total_posts": len(scored),
            "avg_score": avg_score,
            "posts": scored,
            "best": scored[:5] if scored else [],
            "worst": scored[-5:][::-1] if len(scored) >= 5 else [],
        }

    def calculate_reading_level(self, text: str) -> Dict:
        """Flesch-Kincaid readability analysis."""
        words = text.split()
        word_count = len(words)
        if word_count < 10:
            return {"fk_grade": 0, "fk_ease": 0, "word_count": word_count}

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)

        total_syllables = sum(_count_syllables(w) for w in words)

        # Flesch-Kincaid Grade Level
        fk_grade = (0.39 * (word_count / sentence_count) +
                    11.8 * (total_syllables / word_count) - 15.59)

        # Flesch Reading Ease
        fk_ease = (206.835 - 1.015 * (word_count / sentence_count) -
                   84.6 * (total_syllables / word_count))

        return {
            "fk_grade": round(max(0, fk_grade), 1),
            "fk_ease": round(max(0, min(100, fk_ease)), 1),
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": round(word_count / sentence_count, 1),
            "interpretation": (
                "Easy (6th grade)" if fk_grade <= 6 else
                "Standard (7-9th grade)" if fk_grade <= 9 else
                "Moderate (10-12th grade)" if fk_grade <= 12 else
                "Complex (college level)"
            ),
        }

    def check_heading_hierarchy(self, html: str) -> Dict:
        """Check proper H1→H2→H3 nesting."""
        headings = re.findall(r'<h([1-6])[^>]*>', html, re.IGNORECASE)
        levels = [int(h) for h in headings]
        score = 0
        issues = []

        if not levels:
            return {"score": 3, "issues": ["No headings found"], "headings": []}

        # H1 should appear 0-1 times (WordPress theme usually provides H1)
        h1_count = levels.count(1)
        if h1_count > 1:
            issues.append("Multiple H1 tags")
        elif h1_count == 0:
            score += 3  # Theme provides H1

        # Should have H2s
        h2_count = levels.count(2)
        if h2_count >= 3:
            score += 5
        elif h2_count >= 1:
            score += 3
        else:
            issues.append("No H2 headings")

        # Check for skipped levels (e.g., H2 → H4)
        skips = 0
        for i in range(1, len(levels)):
            if levels[i] > levels[i-1] + 1:
                skips += 1

        if skips == 0:
            score += 5
        elif skips <= 2:
            score += 2
            issues.append(f"{skips} heading level skips")
        else:
            issues.append(f"{skips} heading level skips (poor hierarchy)")

        # H2/H3 ratio
        h3_count = levels.count(3)
        if h2_count > 0 and h3_count > 0:
            score += 2  # Good sub-structure

        return {
            "score": min(15, score),
            "issues": issues,
            "h1": h1_count,
            "h2": h2_count,
            "h3": h3_count,
            "total": len(levels),
        }

    def check_media_ratio(self, html: str, word_count: int) -> Dict:
        """Check images per 500 words."""
        img_count = len(re.findall(r'<img\s', html, re.IGNORECASE))
        video_count = len(re.findall(r'<(video|iframe)', html, re.IGNORECASE))
        total_media = img_count + video_count

        ideal = max(1, word_count // 500)
        ratio = total_media / max(ideal, 1)

        if ratio >= 1.0:
            score = 15
        elif ratio >= 0.5:
            score = 10
        elif total_media >= 1:
            score = 5
        else:
            score = 0

        return {
            "score": score,
            "images": img_count,
            "videos": video_count,
            "total_media": total_media,
            "ideal_count": ideal,
            "word_count": word_count,
        }
