#!/usr/bin/env python3
"""
ZimmWriter Bulk CSV Generator
Generates properly formatted CSVs for ZimmWriter bulk blog writer
"""

import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

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
    Generate a ZimmWriter bulk CSV file
    
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
    
    # CSV Header
    fieldnames = [
        'ARTICLE TITLE',
        'OUTLINE FOCUS',
        'BACKGROUND',
        'OUTLINE',
        'SEO KEYWORDS',
        'ONE WORDPRESS CATEGORY',
        'SLUG'
    ]
    
    # Add directions header
    directions = '''Directions: Follow the format below. When complete, Click on Sheet2 → Click on File → Click on Download → Select CSV.  Load the CSV in the Bulk writer inside ZimmWriter and it will override any manual titles you enter. You can tell it's loaded correctly when the manual titles vanish and the SEO CSV button turns green.

Best Practices:
1. Each row requires an Article Title, but the other columns are optional.
2. Double click on some of the cells (e.g., Background, Outline, SEO Keywords) to expand them and see that they are more than they appear.
3. Outline Focus (optional) only affects the outline that is generated and is therefore unnecessary if you're using a custom outline in the Outline column.
4. Background (optional) length limit is about 1,200 words per article title. You can alternatively add 1-3 URLs to scrape. Each URL should be on a new line within the same cell.
5. SEO Keywords (optional) can be comma separated or on a new line. Limited to a max of 150 keywords.
6. Outline (optional) must be in the format described in the SEO Writer exhaustive guide. Variables: {list}, {table}, {yt}, {tpl}, {url=}
7. One WordPress Category allows you to set the WP category for the article. ZimmWriter will auto create the category if it does not already exist.
8. The slug is used when you want to specifically define the slug that is used when ZimmWriter uploads the article to WordPress.
9. Avoid deleting rows and/or inserting new rows as that can screw up the Sheet2 formulas.'''
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write directions in first row
        csvfile.write(f'"{directions}"\n')
        csvfile.write(',,,,,,\n')  # Empty row
        
        # Write header
        writer.writeheader()
        
        # Write articles
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
                # Remove special characters
                slug = ''.join(c for c in slug if c.isalnum() or c == '-')
            
            row = {
                'ARTICLE TITLE': article.get('title', ''),
                'OUTLINE FOCUS': article.get('outline_focus', ''),
                'BACKGROUND': background,
                'OUTLINE': article.get('outline', ''),
                'SEO KEYWORDS': keywords,
                'ONE WORDPRESS CATEGORY': category,
                'SLUG': slug
            }
            
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
    
    print("✓ Generated smart_home_articles.csv")
    
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
    
    print("✓ Generated detailed_articles.csv")
