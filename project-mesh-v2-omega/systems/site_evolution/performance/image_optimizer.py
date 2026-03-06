"""
Image Optimizer — Audit media library, generate alt text, produce WebP/srcset snippets.
All fixes deployed as Code Snippets-compatible PHP.
"""

import logging
import re
from typing import Dict, List, Optional

from systems.site_evolution.utils import load_site_config, get_site_domain, get_site_brand_name

log = logging.getLogger(__name__)


def _get_media(site_slug: str, limit: int = 50) -> List[Dict]:
    """Fetch media library items."""
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        return _wp_request(
            site_slug, "GET",
            f"wp/v2/media?per_page={limit}&media_type=image"
            "&_fields=id,alt_text,source_url,media_details,title,post,caption"
        ) or []
    except Exception as e:
        log.warning("Could not fetch media for %s: %s", site_slug, e)
        return []


def _get_posts_with_media(site_slug: str, limit: int = 20) -> List[Dict]:
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        return _wp_request(
            site_slug, "GET",
            f"wp/v2/posts?per_page={limit}&status=publish"
            "&_fields=id,title,content,featured_media"
        ) or []
    except Exception as e:
        log.warning("Could not fetch posts for %s: %s", site_slug, e)
        return []


class ImageOptimizer:
    """Audit and optimize images across WordPress sites."""

    def audit_images(self, site_slug: str, limit: int = 50) -> Dict:
        """Check media library for missing alt text, oversized images, no WebP.

        Returns: {total, missing_alt, oversized, no_webp, score, issues}
        """
        media = _get_media(site_slug, limit)
        score = 50  # Base
        issues = []

        if not media:
            return {"total": 0, "score": 30, "issues": [{"type": "warning", "msg": "No media found"}]}

        missing_alt = []
        oversized = []
        has_webp = False

        for item in media:
            # Check alt text
            alt = item.get("alt_text", "").strip()
            if not alt:
                title = item.get("title", {})
                if isinstance(title, dict):
                    title = title.get("rendered", "")
                missing_alt.append({
                    "id": item["id"],
                    "url": item.get("source_url", ""),
                    "title": title,
                })

            # Check file size
            details = item.get("media_details", {})
            filesize = details.get("filesize", 0)
            if isinstance(filesize, str):
                try:
                    filesize = int(filesize)
                except ValueError:
                    filesize = 0

            if filesize > 500_000:  # >500KB
                oversized.append({
                    "id": item["id"],
                    "url": item.get("source_url", ""),
                    "size_kb": filesize // 1024,
                })

            # Check for WebP
            source_url = item.get("source_url", "")
            if source_url.endswith(".webp"):
                has_webp = True

        # Scoring
        alt_coverage = 1 - (len(missing_alt) / max(len(media), 1))
        if alt_coverage >= 0.9:
            score += 20
        elif alt_coverage >= 0.7:
            score += 10
        else:
            issues.append({"type": "warning", "msg": f"{len(missing_alt)}/{len(media)} images missing alt text"})

        if not oversized:
            score += 15
        else:
            issues.append({"type": "warning", "msg": f"{len(oversized)} oversized images (>500KB)"})

        if has_webp:
            score += 15
        else:
            issues.append({"type": "info", "msg": "No WebP images detected"})

        return {
            "site_slug": site_slug,
            "total": len(media),
            "missing_alt": missing_alt,
            "missing_alt_count": len(missing_alt),
            "oversized": oversized,
            "oversized_count": len(oversized),
            "has_webp": has_webp,
            "alt_coverage": round(alt_coverage * 100, 1),
            "score": min(100, score),
            "issues": issues,
        }

    def generate_alt_text(self, site_slug: str, post_id: int) -> List[Dict]:
        """Generate contextual alt text suggestions from surrounding content."""
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        brand = get_site_brand_name(site_slug)

        try:
            post = _wp_request(site_slug, "GET", f"wp/v2/posts/{post_id}?_fields=title,content")
        except Exception:
            return []

        if not post:
            return []

        title = post.get("title", {})
        if isinstance(title, dict):
            title = title.get("rendered", "")

        content = post.get("content", {})
        if isinstance(content, dict):
            content = content.get("rendered", "")

        # Find images without alt text
        img_pattern = re.compile(r'<img\s[^>]*?(?:alt=["\']([^"\']*)["\'])?\s[^>]*?src=["\']([^"\']+)["\'][^>]*?>', re.IGNORECASE)
        suggestions = []

        for match in img_pattern.finditer(content):
            alt = match.group(1) or ""
            src = match.group(2)

            if alt.strip():
                continue

            # Extract surrounding text for context
            start = max(0, match.start() - 200)
            end = min(len(content), match.end() + 200)
            context = re.sub(r'<[^>]+>', ' ', content[start:end]).strip()
            context_words = context.split()[:15]

            # Build descriptive alt text
            if context_words:
                alt_suggestion = f"{title} - {' '.join(context_words[:8])}"
            else:
                alt_suggestion = f"{title} - {brand}"

            # Truncate to 125 chars (accessibility best practice)
            alt_suggestion = alt_suggestion[:125]

            suggestions.append({
                "src": src,
                "suggested_alt": alt_suggestion,
                "context": " ".join(context_words[:20]),
            })

        return suggestions

    def fix_missing_alt_text(self, site_slug: str, dry_run: bool = True) -> Dict:
        """Batch fix missing alt text on media items."""
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        brand = get_site_brand_name(site_slug)

        media = _get_media(site_slug, 100)
        fixed = []
        errors = []

        for item in media:
            alt = item.get("alt_text", "").strip()
            if alt:
                continue

            title = item.get("title", {})
            if isinstance(title, dict):
                title = title.get("rendered", "")

            if not title:
                continue

            # Generate alt text from title + brand
            new_alt = f"{title} - {brand}"[:125]

            if dry_run:
                fixed.append({"id": item["id"], "alt": new_alt, "dry_run": True})
                continue

            try:
                _wp_request(site_slug, "POST", f"wp/v2/media/{item['id']}",
                            data={"alt_text": new_alt})
                fixed.append({"id": item["id"], "alt": new_alt})
            except Exception as e:
                errors.append({"id": item["id"], "error": str(e)})

        return {
            "site_slug": site_slug,
            "dry_run": dry_run,
            "fixed": len(fixed),
            "errors": len(errors),
            "details": fixed[:20],
        }

    def generate_webp_snippet(self) -> str:
        """PHP snippet to serve WebP when browser supports it."""
        return """<?php
/**
 * WebP Auto-Serve — Serve WebP versions when available and browser supports it.
 */
function evo_webp_serve($content) {
    if (!is_singular()) return $content;

    // Check if browser supports WebP
    $supports_webp = (isset($_SERVER['HTTP_ACCEPT']) && strpos($_SERVER['HTTP_ACCEPT'], 'image/webp') !== false);
    if (!$supports_webp) return $content;

    // Replace jpg/png with .webp if the file exists
    $content = preg_replace_callback(
        '/(src=["\'])([^"\']+\\.(jpg|jpeg|png))(["\'])/i',
        function($matches) {
            $webp_url = preg_replace('/\\.(jpg|jpeg|png)$/i', '.webp', $matches[2]);
            $upload_dir = wp_upload_dir();
            $webp_path = str_replace($upload_dir['baseurl'], $upload_dir['basedir'], $webp_url);
            if (file_exists($webp_path)) {
                return $matches[1] . $webp_url . $matches[4];
            }
            return $matches[0];
        },
        $content
    );

    return $content;
}
add_filter('the_content', 'evo_webp_serve', 99);
add_filter('post_thumbnail_html', 'evo_webp_serve', 99);
"""

    def generate_srcset_snippet(self) -> str:
        """PHP snippet to ensure proper srcset/sizes attributes."""
        return """<?php
/**
 * Image Srcset/Sizes Optimization — Proper responsive images.
 */
function evo_optimize_srcset_sizes($attr, $attachment, $size) {
    // Set optimal sizes attribute for common layouts
    if (!isset($attr['sizes'])) {
        $attr['sizes'] = '(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw';
    }

    // Add decoding async for non-critical images
    if (!isset($attr['decoding'])) {
        $attr['decoding'] = 'async';
    }

    return $attr;
}
add_filter('wp_get_attachment_image_attributes', 'evo_optimize_srcset_sizes', 10, 3);

// Increase srcset max width (default 2048px is too low for hero images)
function evo_max_srcset_width($max_width) {
    return 2560;
}
add_filter('max_srcset_image_width', 'evo_max_srcset_width');
"""

    def generate_placeholder_snippet(self) -> str:
        """PHP/CSS snippet for dominant-color placeholder during image load."""
        return """<?php
/**
 * Image Placeholder — Shows dominant-color background while images load.
 * Prevents layout shift (CLS) and improves perceived performance.
 */
function evo_image_placeholder_css() {
    echo '<style>
    .evo-img-placeholder {
        background: var(--color-bg-alt, #f0f0f0);
        aspect-ratio: 16/9;
        animation: evo-shimmer 1.5s ease-in-out infinite;
        background-image: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
        background-size: 200% 100%;
    }
    @keyframes evo-shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    img[loading="lazy"] {
        background: var(--color-bg-alt, #f0f0f0);
        transition: opacity 0.3s ease;
    }
    img[loading="lazy"].loaded { opacity: 1; }
    </style>';
}
add_action('wp_head', 'evo_image_placeholder_css');
"""

    def get_image_score(self, site_slug: str) -> Dict:
        """Score 0-100 for auditor integration."""
        audit = self.audit_images(site_slug)
        return {"score": audit.get("score", 0), "issues": audit.get("issues", [])}
