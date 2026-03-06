"""
Schema Generator — JSON-LD structured data for every page type.
Validates against Google Rich Results format.
"""

import json
import logging
from datetime import date
from typing import Dict, List, Optional

from systems.site_evolution.utils import load_site_config

log = logging.getLogger(__name__)


class SchemaGenerator:
    """Generate JSON-LD structured data for WordPress sites."""

    def generate_organization_schema(self, site_slug: str) -> Dict:
        """Organization schema with logo, social, sameAs."""
        config = load_site_config(site_slug)
        domain = config.get("domain", "example.com")
        brand = config.get("name", site_slug)

        return {
            "@context": "https://schema.org",
            "@type": "Organization",
            "name": brand,
            "url": f"https://{domain}",
            "logo": f"https://{domain}/wp-content/uploads/logo.png",
            "description": f"{brand} - Expert guidance and in-depth reviews",
            "sameAs": [],
        }

    def generate_website_schema(self, site_slug: str) -> Dict:
        """WebSite schema with SearchAction (sitelinks search box)."""
        config = load_site_config(site_slug)
        domain = config.get("domain", "example.com")
        brand = config.get("name", site_slug)

        return {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": brand,
            "url": f"https://{domain}",
            "potentialAction": {
                "@type": "SearchAction",
                "target": {
                    "@type": "EntryPoint",
                    "urlTemplate": f"https://{domain}/?s={{search_term_string}}"
                },
                "query-input": "required name=search_term_string"
            }
        }

    def generate_article_schema(self, site_slug: str, post: Dict) -> Dict:
        """Article schema with author, dates, image, wordCount."""
        config = load_site_config(site_slug)
        domain = config.get("domain", "example.com")
        brand = config.get("name", site_slug)

        title = post.get("title", {})
        if isinstance(title, dict):
            title = title.get("rendered", "")

        return {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "author": {
                "@type": "Organization",
                "name": brand,
                "url": f"https://{domain}/about"
            },
            "publisher": {
                "@type": "Organization",
                "name": brand,
                "logo": {
                    "@type": "ImageObject",
                    "url": f"https://{domain}/wp-content/uploads/logo.png"
                }
            },
            "datePublished": post.get("date", ""),
            "dateModified": post.get("modified", post.get("date", "")),
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": post.get("link", f"https://{domain}")
            },
            "wordCount": post.get("word_count", 0),
            "image": post.get("featured_image_url", f"https://{domain}/wp-content/uploads/default-og.png"),
        }

    def generate_faq_schema(self, questions: List[Dict]) -> Dict:
        """FAQPage schema from Q&A pairs.

        Args:
            questions: list of {question: str, answer: str}
        """
        return {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": q["question"],
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": q["answer"]
                    }
                }
                for q in questions
            ]
        }

    def generate_howto_schema(self, title: str, steps: List[str],
                              description: str = "") -> Dict:
        """HowTo schema with step-by-step instructions."""
        return {
            "@context": "https://schema.org",
            "@type": "HowTo",
            "name": title,
            "description": description,
            "step": [
                {
                    "@type": "HowToStep",
                    "position": i + 1,
                    "text": step,
                }
                for i, step in enumerate(steps)
            ]
        }

    def generate_product_review_schema(self, product_data: Dict) -> Dict:
        """Product + Review schema with ratings, price, availability."""
        return {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": product_data.get("name", ""),
            "description": product_data.get("description", ""),
            "brand": {
                "@type": "Brand",
                "name": product_data.get("brand", "")
            },
            "image": product_data.get("image", ""),
            "review": {
                "@type": "Review",
                "reviewRating": {
                    "@type": "Rating",
                    "ratingValue": product_data.get("rating", 4.0),
                    "bestRating": 5,
                    "worstRating": 1,
                },
                "author": {
                    "@type": "Organization",
                    "name": product_data.get("reviewer", "Editorial Team")
                }
            },
            "aggregateRating": {
                "@type": "AggregateRating",
                "ratingValue": product_data.get("rating", 4.0),
                "reviewCount": product_data.get("review_count", 1),
            },
            "offers": {
                "@type": "Offer",
                "price": product_data.get("price", ""),
                "priceCurrency": "USD",
                "availability": "https://schema.org/InStock",
                "url": product_data.get("url", ""),
            }
        }

    def generate_breadcrumb_schema(self, path: List[Dict]) -> Dict:
        """BreadcrumbList schema.

        Args:
            path: list of {name: str, url: str}
        """
        return {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": i + 1,
                    "name": item["name"],
                    "item": item["url"],
                }
                for i, item in enumerate(path)
            ]
        }

    def generate_recipe_schema(self, recipe_data: Dict) -> Dict:
        """Recipe schema for food/cooking content."""
        return {
            "@context": "https://schema.org",
            "@type": "Recipe",
            "name": recipe_data.get("name", ""),
            "description": recipe_data.get("description", ""),
            "image": recipe_data.get("image", ""),
            "author": {
                "@type": "Organization",
                "name": recipe_data.get("author", "Editorial Team")
            },
            "prepTime": recipe_data.get("prep_time", "PT15M"),
            "cookTime": recipe_data.get("cook_time", "PT30M"),
            "totalTime": recipe_data.get("total_time", "PT45M"),
            "recipeYield": recipe_data.get("servings", "4 servings"),
            "recipeCategory": recipe_data.get("category", ""),
            "recipeCuisine": recipe_data.get("cuisine", ""),
            "recipeIngredient": recipe_data.get("ingredients", []),
            "recipeInstructions": [
                {"@type": "HowToStep", "text": step}
                for step in recipe_data.get("instructions", [])
            ],
            "nutrition": {
                "@type": "NutritionInformation",
                "calories": recipe_data.get("calories", ""),
            } if recipe_data.get("calories") else {},
        }

    def generate_event_schema(self, event_data: Dict) -> Dict:
        """Event schema for event-related content."""
        return {
            "@context": "https://schema.org",
            "@type": "Event",
            "name": event_data.get("name", ""),
            "description": event_data.get("description", ""),
            "startDate": event_data.get("start_date", ""),
            "endDate": event_data.get("end_date", ""),
            "location": {
                "@type": event_data.get("location_type", "Place"),
                "name": event_data.get("location_name", ""),
                "address": event_data.get("address", ""),
            } if event_data.get("location_name") else {
                "@type": "VirtualLocation",
                "url": event_data.get("url", ""),
            },
            "organizer": {
                "@type": "Organization",
                "name": event_data.get("organizer", ""),
            },
            "image": event_data.get("image", ""),
            "eventStatus": "https://schema.org/EventScheduled",
        }

    def generate_video_schema(self, video_data: Dict) -> Dict:
        """VideoObject schema for video content."""
        return {
            "@context": "https://schema.org",
            "@type": "VideoObject",
            "name": video_data.get("name", ""),
            "description": video_data.get("description", ""),
            "thumbnailUrl": video_data.get("thumbnail", ""),
            "uploadDate": video_data.get("upload_date", ""),
            "duration": video_data.get("duration", ""),
            "contentUrl": video_data.get("content_url", ""),
            "embedUrl": video_data.get("embed_url", ""),
            "publisher": {
                "@type": "Organization",
                "name": video_data.get("publisher", ""),
            },
        }

    def generate_course_schema(self, course_data: Dict) -> Dict:
        """Course schema for educational content."""
        return {
            "@context": "https://schema.org",
            "@type": "Course",
            "name": course_data.get("name", ""),
            "description": course_data.get("description", ""),
            "provider": {
                "@type": "Organization",
                "name": course_data.get("provider", ""),
            },
            "courseMode": course_data.get("mode", "online"),
            "educationalLevel": course_data.get("level", "beginner"),
            "isAccessibleForFree": course_data.get("free", True),
            "hasCourseInstance": {
                "@type": "CourseInstance",
                "courseMode": "online",
            },
        }

    def generate_software_schema(self, software_data: Dict) -> Dict:
        """SoftwareApplication schema for tool/app reviews."""
        return {
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            "name": software_data.get("name", ""),
            "description": software_data.get("description", ""),
            "applicationCategory": software_data.get("category", "Utility"),
            "operatingSystem": software_data.get("os", "Web"),
            "offers": {
                "@type": "Offer",
                "price": software_data.get("price", "0"),
                "priceCurrency": "USD",
            },
            "aggregateRating": {
                "@type": "AggregateRating",
                "ratingValue": software_data.get("rating", 4.0),
                "reviewCount": software_data.get("review_count", 1),
            } if software_data.get("rating") else {},
        }

    def generate_site_schemas(self, site_slug: str) -> str:
        """Generate all site-wide schemas as a combined JSON-LD script tag."""
        schemas = [
            self.generate_organization_schema(site_slug),
            self.generate_website_schema(site_slug),
        ]
        combined = json.dumps(schemas, indent=2)
        return f'<script type="application/ld+json">\n{combined}\n</script>'
