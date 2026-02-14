# ZimmWriter Outline Patterns

Common outline structures for different content types.

## Basic Article Outline

```
Introduction
Main Topic 1
- Subtopic A
- Subtopic B
- Subtopic C
Main Topic 2
- Subtopic A
- Subtopic B
Main Topic 3{list}
Conclusion
```

## Product Roundup Outline

```
Introduction
Best [Product Category]
- Product 1 Name{url=https://amazon.com/dp/ID}{tpl}
- Product 2 Name{url=https://amazon.com/dp/ID}{tpl}
- Product 3 Name{url=https://amazon.com/dp/ID}{tpl}
- Product 4 Name{url=https://amazon.com/dp/ID}{tpl}
- Product 5 Name{url=https://amazon.com/dp/ID}{tpl}
Factors to Consider When Buying
- Feature Comparison{table}
- Price Ranges{list}
- Key Specifications
Buying Guide{list}
Conclusion
```

## Product Comparison (2 Products)

```
Introduction
[Product A] Overview{url=https://amazon.com/dp/ID}{tpl}
[Product B] Overview{url=https://amazon.com/dp/ID}{tpl}
Feature Comparison{table}
Performance Comparison
- Speed and Efficiency
- Reliability and Durability
- User Experience
Price and Value Analysis
Which Should You Choose?{list}
Conclusion
```

## How-To Guide Outline

```
Introduction
What You'll Need{list}
Step-by-Step Instructions
- Step 1: [Action]{url=https://reference-url.com}
-- Detailed Sub-step A
-- Detailed Sub-step B
- Step 2: [Action]
-- Detailed Sub-step A
-- Detailed Sub-step B
- Step 3: [Action]
Common Mistakes to Avoid{list}
Troubleshooting{table}
Conclusion
```

## Comparison/Versus Article

```
Introduction
[Option A] Explained{yt}
- Key Features{list}
- Pros and Cons{table}
- Best Use Cases
[Option B] Explained{yt}
- Key Features{list}
- Pros and Cons{table}
- Best Use Cases
Head-to-Head Comparison{table}
Which is Right for You?
- Choose [A] if...{list}
- Choose [B] if...{list}
Conclusion
```

## Ultimate Guide/Pillar Content

```
Introduction
What is [Topic]?{yt}
History and Evolution
Core Concepts
- Concept 1{url=https://reference.com}
-- Deep Dive A{table}
-- Deep Dive B
- Concept 2
-- Deep Dive A
-- Deep Dive B
- Concept 3
Practical Applications{list}
Advanced Techniques
- Technique 1
- Technique 2
- Technique 3
Common Questions{list}
Tools and Resources{table}
Conclusion
```

## List/Listicle Article

```
Introduction
[Number] [Things]
- Item 1: [Title]{yt}
-- Description and details
-- Why it matters{list}
- Item 2: [Title]
-- Description and details
-- Why it matters{list}
- Item 3: [Title]
-- Description and details
-- Why it matters{list}
- Item 4: [Title]
-- Description and details
-- Why it matters{list}
- Item 5: [Title]
-- Description and details
-- Why it matters{list}
Final Thoughts
Conclusion
```

## Smart Home Setup Guide

```
Introduction
What You'll Need{list}
Compatibility Check{table}
Step 1: Install the Hub
- Unboxing and Contents
- Physical Installation
- Power and Connectivity
Step 2: Download the App
- iOS Setup
- Android Setup
- Initial Configuration
Step 3: Add Devices
- Pairing Process{url=https://manufacturer-guide.com}
- Naming Conventions
- Room Assignments
Step 4: Create Automations{list}
Troubleshooting Common Issues{table}
Advanced Tips{list}
Conclusion
```

## Variable Usage Guide

### {list}
Generates a bulleted list for that section. Use when you want:
- Multiple quick points
- Features enumeration
- Quick tips or recommendations

Example: `Common Mistakes{list}`

### {table}
Creates a comparison or specification table. Use for:
- Product comparisons
- Spec sheets
- Feature matrices
- Pros vs Cons

Example: `Feature Comparison{table}`

### {yt}
Suggests/embeds a relevant YouTube video. Use for:
- Visual demonstrations
- Tutorial videos
- Product reviews

Example: `How to Install{yt}`

### {tpl}
Product template - generates structured product review section. Use with {url=}:
- Product roundups
- Best-of lists
- Product comparisons

Example: `- Nest Thermostat{url=https://amazon.com/dp/ID}{tpl}`

### {url=https://...}
Scrapes specific URL for content. Use for:
- Manufacturer specifications
- Official documentation
- Detailed references

Example: `Installation Guide{url=https://manufacturer.com/guide}`

## Outline Hierarchy

```
Main Heading (H2)
- Subheading (H3)
-- Sub-subheading (H4)
--- Sub-sub-subheading (H5)
```

Maximum recommended depth: H4 (two dashes)
Most articles: Stick to H2 and H3 (main topics and subtopics)

## Best Practices

1. **Keep it balanced** - 6-10 H2 sections for most articles
2. **Use variables strategically** - Don't overuse {list} and {table}
3. **Hierarchy matters** - Maintain logical structure
4. **URL variables** - Simplify Amazon URLs to /dp/PRODUCT_ID format
5. **Product templates** - Always pair {tpl} with {url=}
6. **Scraping limits** - Don't overload URLs (max 3-5 per article)
7. **Variable placement** - Place at end of heading (e.g., "Topic{list}")

## Common Mistakes

❌ **Bad**: `{list}Common Mistakes` - Variable before text
✓ **Good**: `Common Mistakes{list}` - Variable after text

❌ **Bad**: `Product Name{tpl}` - Missing URL
✓ **Good**: `Product Name{url=...}{tpl}` - URL included

❌ **Bad**: Long complex Amazon URL with tracking
✓ **Good**: Simplified to amazon.com/dp/PRODUCTID

❌ **Bad**: Too many H2s (15+)
✓ **Good**: Focused 6-10 H2 sections

❌ **Bad**: Every section has {list} or {table}
✓ **Good**: Strategic use where it adds value
