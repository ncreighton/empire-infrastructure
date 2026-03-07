#!/usr/bin/env python3
"""
ZimmWriter Bulk CSV Generator
Generates properly formatted CSVs for ZimmWriter bulk blog writer.

ZimmWriter expects the Sheet2 format from its Google Sheet template:
- NO header row
- Each cell is wrapped as {fieldname=valueZW}
- Empty cells are left empty (no tag wrapper)
- 7 columns: title, outline_focus, background, outline, keywords, category, slug
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional


# ZimmWriter Sheet2 field names (column order matters)
ZW_FIELDS = ['title', 'outline_focus', 'background', 'outline',
             'keywords', 'category', 'slug']


def _expand_auto_h3(outline: str) -> str:
    """Expand {auto_h3_#} tags into explicit H3 placeholder lines.

    ZimmWriter's SEO CSV format does not support {auto_h3_#} tags (they work
    only in the direct UI outline input).  Replace each occurrence with the
    corresponding number of H3 placeholder lines that ZimmWriter will
    rewrite via {optimize_title}.
    """
    def _replace(m):
        count = int(m.group(1))
        lines = '\n'.join(f'- Subtopic {i+1}{{optimize_title}}' for i in range(count))
        return '\n' + lines
    return re.sub(r'\{auto_h3_(\d+)\}', _replace, outline)


def _zw_wrap(field_name: str, value: str) -> str:
    """Wrap a value in ZimmWriter's {field=valueZW} format.

    Returns empty string when value is empty/None.
    """
    if not value:
        return ''
    return '{' + field_name + '=' + value + 'ZW}'


def load_site_config(site_domain: str) -> Dict:
    """Load configuration for a specific site"""
    config_path = Path(__file__).parent.parent / "configs" / "site-configs.json"
    with open(config_path, 'r') as f:
        configs = json.load(f)

    if site_domain in configs['sites']:
        return configs['sites'][site_domain]
    else:
        raise ValueError(f"Site {site_domain} not found in configurations")

def generate_bulk_csv(
    articles: List[Dict],
    output_path: str,
    site_domain: Optional[str] = None
) -> str:
    """
    Generate a ZimmWriter bulk CSV file in Sheet2 format.

    Each cell is wrapped as {fieldname=valueZW}. No header row.

    Args:
        articles: List of article dictionaries with keys:
            - title (required)
            - outline_focus (optional)
            - background (optional) - text or URLs (one per line)
            - outline (optional) - custom outline format
            - seo_keywords (optional) - list or newline-separated string
            - wordpress_category (optional)
            - slug (optional)
        output_path: Path where CSV will be saved
        site_domain: Domain to load default settings from

    Returns:
        Path to generated CSV file
    """

    # Load site config if provided
    site_config = load_site_config(site_domain) if site_domain else None

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        for article in articles:
            # Process SEO keywords
            keywords = article.get('seo_keywords', '')
            if isinstance(keywords, list):
                keywords = '\n'.join(keywords)

            # Process background
            background = article.get('background', '')
            if isinstance(background, list):
                background = '\n'.join(background)

            # Use site default category if not specified
            category = article.get('wordpress_category', '')
            if not category and site_config:
                category = site_config.get('wordpress', {}).get('default_category', '')

            # Generate slug if not provided
            slug = article.get('slug', '')
            if not slug and article.get('title'):
                slug = article['title'].lower().replace(' ', '-')
                slug = ''.join(c for c in slug if c.isalnum() or c == '-')

            row = [
                _zw_wrap('title', article.get('title', '')),
                _zw_wrap('outline_focus', article.get('outline_focus', '')),
                _zw_wrap('background', background),
                _zw_wrap('outline', _expand_auto_h3(article.get('outline', ''))),
                _zw_wrap('keywords', keywords),
                _zw_wrap('category', category),
                _zw_wrap('slug', slug),
            ]

            writer.writerow(row)

    return output_path

def generate_csv_from_titles(
    titles: List[str],
    output_path: str,
    site_domain: Optional[str] = None,
    add_outline_focus: Optional[str] = None,
    wordpress_category: Optional[str] = None
) -> str:
    """
    Quick CSV generation from just titles

    Args:
        titles: List of article titles
        output_path: Where to save CSV
        site_domain: Site domain for defaults
        add_outline_focus: Optional outline focus to apply to all
        wordpress_category: Optional category for all articles
    """
    articles = []
    for title in titles:
        article = {'title': title}
        if add_outline_focus:
            article['outline_focus'] = add_outline_focus
        if wordpress_category:
            article['wordpress_category'] = wordpress_category
        articles.append(article)

    return generate_bulk_csv(articles, output_path, site_domain)

# Example usage
if __name__ == "__main__":
    # Example 1: Simple title list
    simple_titles = [
        "How to Set Up Your First Smart Home Hub",
        "Best Smart Thermostats of 2025",
        "Zigbee vs Z-Wave: Which Protocol is Better?"
    ]

    generate_csv_from_titles(
        titles=simple_titles,
        output_path="/tmp/smart_home_articles.csv",
        site_domain="smarthomewizards.com",
        wordpress_category="Smart Home Guides"
    )

    print("Generated smart_home_articles.csv")

    # Example 2: Full article details
    detailed_articles = [
        {
            'title': 'Complete Guide to Home Assistant',
            'outline_focus': 'Cover installation, basic configuration, popular integrations, and automation examples',
            'background': 'https://www.home-assistant.io/getting-started/',
            'seo_keywords': ['home assistant', 'smart home automation', 'home assistant setup', 'smart home hub'],
            'wordpress_category': 'Software Guides',
            'slug': 'home-assistant-complete-guide'
        }
    ]

    generate_bulk_csv(
        articles=detailed_articles,
        output_path="/tmp/detailed_articles.csv",
        site_domain="smarthomewizards.com"
    )

    print("Generated detailed_articles.csv")
