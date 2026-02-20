"""
ZimmWriter Outline Template Library.

Provides rotatable outline templates for different article types.
Each type has 2-3 variants selected randomly or rotated through
to prevent structural repetition across articles.

ZimmWriter outline syntax:
    - Lines are H2 headings by default
    - Lines starting with "- " are H3 subheadings
    - Lines starting with "-- " are H4
    - {list} generates a bulleted list for that section
    - {table} generates a comparison/data table
    - {yt} embeds a relevant YouTube video
    - {img} inserts an AI-generated image
    - {auto_h3_#} auto-generates # H3 subheadings
    - {optimize_title} rewrites the heading based on article title context
    - {research} enables Deep Research for that section
    - {cp_name} applies a saved custom prompt to that section
    - [lp_packname] loads a specific link pack for internal linking

Design principles:
    - Use {optimize_title} on generic/placeholder H2s so AI adapts them
    - Place {list} where quick-scan content makes sense
    - Place {table} where comparison data adds value
    - Place {img} at visual break points (every 2-3 H2s)
    - Place {yt} once per article where video adds value
    - Keep templates niche-agnostic — niche voice comes from custom prompts
    - Vary structure between variants to prevent repetitive articles
"""

import random
from typing import Dict, List, Optional


# ═══════════════════════════════════════════
# TEMPLATE DEFINITIONS
# ═══════════════════════════════════════════

OUTLINE_TEMPLATES: Dict[str, List[str]] = {

    # ─── HOW-TO — 3 variants ───
    "how_to": [
        # Variant A: Classic step-by-step with materials list
        """\
Introduction
What You Need Before Starting{list}
Step 1: Prepare Your Setup{optimize_title}
- Initial Configuration
- Verifying Everything Works
Step 2: Execute the Core Process{optimize_title}{img}
- Detailed Walkthrough
-- Handling Common Variations
-- Adjusting for Your Situation
Step 3: Fine-Tune Your Results{optimize_title}
- Testing and Validation
- Making Final Adjustments
Common Mistakes to Avoid{list}
Troubleshooting Guide{table}
Quick Reference Checklist{list}""",

        # Variant B: Problem-solution structure
        """\
Introduction
Understanding the Problem{optimize_title}
- Why This Matters
- What Most People Get Wrong
The Solution: Step by Step{optimize_title}{yt}
- Phase 1: Foundation{optimize_title}
-- Key Setup Details
- Phase 2: Implementation{optimize_title}{img}
-- Critical Checkpoints
- Phase 3: Verification{optimize_title}
-- How to Know It Worked
Pro Tips From Experience{list}
Before and After: What to Expect{table}
Your Next Steps""",

        # Variant C: Quick-win tutorial (shorter, tighter)
        """\
Introduction
Prerequisites and Tools{list}
The Quick Method{optimize_title}
- Step 1{optimize_title}
- Step 2{optimize_title}{img}
- Step 3{optimize_title}
- Step 4{optimize_title}
The Advanced Method{optimize_title}
- When to Use This Instead
- Detailed Walkthrough{auto_h3_3}
Tips That Save Time{list}
What Can Go Wrong{table}
Try This Today""",
    ],

    # ─── LISTICLE — 3 variants ───
    "listicle": [
        # Variant A: Detailed items with pros/context
        """\
Introduction
How We Evaluated These{optimize_title}
{optimize_title}{img}
- Why It Stands Out
- Best For
- Key Drawback
{optimize_title}
- Why It Stands Out
- Best For
- Key Drawback
{optimize_title}{img}
- Why It Stands Out
- Best For
- Key Drawback
{optimize_title}
- Why It Stands Out
- Best For
- Key Drawback
{optimize_title}
- Why It Stands Out
- Best For
- Key Drawback
Side-by-Side Comparison{table}
How to Choose the Right One{list}""",

        # Variant B: Quick-scan list (more items, less depth)
        """\
Introduction
{optimize_title}
{optimize_title}
{optimize_title}{img}
{optimize_title}
{optimize_title}
{optimize_title}
{optimize_title}{img}
Honorable Mentions{list}
Quick Comparison{table}
The Bottom Line""",

        # Variant C: Grouped categories
        """\
Introduction
Best Overall Picks{optimize_title}
- Top Pick{optimize_title}
- Runner Up{optimize_title}
Best Budget Options{optimize_title}{img}
- Budget Pick{optimize_title}
- Value Pick{optimize_title}
Best for Specific Needs{optimize_title}
- Specialized Pick{optimize_title}
- Niche Pick{optimize_title}{img}
Complete Comparison{table}
How We Tested{list}
Our Recommendation""",
    ],

    # ─── REVIEW — 3 variants ───
    "review": [
        # Variant A: Single product deep review
        """\
Introduction
Specifications and Pricing{table}
Design and Build Quality{optimize_title}{img}
- Materials and Construction
- Ergonomics and Comfort
Key Features{optimize_title}
- Standout Feature 1{optimize_title}
- Standout Feature 2{optimize_title}
- Standout Feature 3{optimize_title}
Real-World Performance{optimize_title}{img}
- Testing Methodology
- Daily Use Results
- Battery Life Under Load
What the Marketing Doesn't Tell You
- Honest Limitations{list}
- Deal-Breakers vs Minor Annoyances
Alternatives to Consider{table}
Final Verdict""",

        # Variant B: Comparison review (vs format)
        """\
Introduction
Quick Specs Comparison{table}
{optimize_title}: Overview{img}
- Design and Build
- Key Strengths{list}
- Weaknesses
{optimize_title}: Overview{img}
- Design and Build
- Key Strengths{list}
- Weaknesses
Head-to-Head Performance{optimize_title}
- Speed and Responsiveness
- Accuracy and Reliability
- Battery and Endurance{table}
Value for Money{optimize_title}
- Price-to-Feature Analysis
Which One Should You Buy{list}
Final Verdict""",

        # Variant C: Long-term review (after X days/months)
        """\
Introduction
First Impressions vs Reality{optimize_title}
- What I Expected
- What Actually Happened
The Good: What Held Up{optimize_title}{img}
- Standout Performance Areas{list}
- Surprising Strengths
The Bad: What Didn't{optimize_title}
- Where It Falls Short{list}
- The Deal-Breaker Moment
Durability After Extended Use{optimize_title}{img}
- Wear and Tear Report
- Battery Degradation
Who This Is Actually For{list}
Better Alternatives?{table}
Updated Verdict""",
    ],

    # ─── GUIDE — 3 variants ───
    "guide": [
        # Variant A: Comprehensive pillar (deep, authoritative)
        """\
Introduction
What Is {optimize_title}{yt}
- Definition and Core Concepts
- Why It Matters Right Now
History and Background{optimize_title}
- Origins and Evolution
- Key Milestones{list}
Foundational Concepts{optimize_title}{img}
- Core Principle 1{optimize_title}
-- Deep Dive
-- Real-World Examples
- Core Principle 2{optimize_title}
-- Deep Dive
-- Case Studies
- Core Principle 3{optimize_title}
-- Deep Dive
-- Expert Perspectives
Practical Application{optimize_title}{img}
- Getting Started
- Intermediate Techniques
- Advanced Strategies{list}
Tools and Resources{table}
Common Mistakes to Avoid{list}
Frequently Asked Questions{auto_h3_5}""",

        # Variant B: Beginner-friendly progressive guide
        """\
Introduction
Who This Guide Is For{list}
The Basics: What You Need to Know{optimize_title}
- Key Terminology{list}
- The Foundation Explained{optimize_title}
- How Everything Connects{img}
Getting Started: Your First Steps{optimize_title}
- Phase 1: Setup{optimize_title}
- Phase 2: Practice{optimize_title}
- Phase 3: Confidence{optimize_title}{img}
Level Up: Intermediate Skills{optimize_title}
- Technique 1{optimize_title}
- Technique 2{optimize_title}
- Technique 3{optimize_title}
Resources and Tools{table}
Mistakes Every Beginner Makes{list}
Your 30-Day Action Plan{list}""",

        # Variant C: Reference guide (scannable, lookup-friendly)
        """\
Introduction
Quick Reference Summary{table}
Section 1{optimize_title}{img}
- Key Points{list}
- When to Use This
- Common Variations
Section 2{optimize_title}
- Key Points{list}
- When to Use This
- Common Variations{img}
Section 3{optimize_title}
- Key Points{list}
- When to Use This
- Common Variations
Section 4{optimize_title}
- Key Points{list}
- When to Use This
Advanced Topics{optimize_title}{auto_h3_4}
Comparison of Approaches{table}
Next Steps and Further Reading{list}""",
    ],

    # ─── NEWS — 2 variants ───
    "news": [
        # Variant A: Breaking/announcement coverage
        """\
Introduction
Key Details{optimize_title}
- What Happened
- Who Is Involved
- Official Statements
Why This Matters{optimize_title}{img}
- Immediate Impact
- Broader Implications
Industry Reaction{optimize_title}
- Expert Commentary
- Community Response
What This Means for You{list}
What Happens Next{optimize_title}
- Expected Timeline
- What to Watch For""",

        # Variant B: Analysis/context piece
        """\
Introduction
The News: What We Know{optimize_title}
- Key Facts{list}
- Timeline of Events
Background and Context{optimize_title}
- How We Got Here
- Previous Related Developments{img}
Impact Analysis{optimize_title}
- Who Wins
- Who Loses
- The Ripple Effects{list}
Expert Takes{optimize_title}
- Optimistic View
- Skeptical View
Looking Ahead{optimize_title}
- Short-Term Expectations
- Long-Term Significance""",
    ],

    # ─── INFORMATIONAL — 3 variants ───
    "informational": [
        # Variant A: Explainer (what/how/why)
        """\
Introduction
What Is {optimize_title}
- Clear Definition
- Key Characteristics{list}
How It Works{optimize_title}{img}
- The Process Explained
- Step-by-Step Breakdown
Why It Matters{optimize_title}
- Key Benefits{list}
- Real-World Impact
Common Misconceptions{table}
Practical Tips{optimize_title}{img}
- Getting the Most From It
- Avoiding Common Pitfalls{list}
Related Topics to Explore""",

        # Variant B: Deep explainer with history
        """\
Introduction
Overview{optimize_title}
- What You Need to Know
- Why People Are Talking About This{img}
History and Origins{optimize_title}
- Early Developments
- How It Evolved Over Time
How It Actually Works{optimize_title}
- The Core Mechanism
- Key Components{list}
- Under the Hood{img}
Applications and Use Cases{optimize_title}{table}
Advantages and Limitations{table}
The Future{optimize_title}
- Emerging Trends
- What Experts Predict""",

        # Variant C: Problem-focused informational
        """\
Introduction
The Problem{optimize_title}
- Why This Matters
- Who It Affects{img}
The Explanation{optimize_title}
- Root Causes
- Contributing Factors{list}
What the Research Says{optimize_title}
- Key Findings
- Where Experts Agree
- Where They Disagree
Practical Implications{optimize_title}{img}
- What You Can Do{list}
- What to Avoid
Comparison of Approaches{table}
Key Takeaways{list}""",
    ],
}


# ═══════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════

def get_template(article_type: str, variant: int = None) -> str:
    """Return a template for the given article type.

    Args:
        article_type: Key from OUTLINE_TEMPLATES (e.g. "how_to", "listicle").
        variant: Zero-based index. If None, selects randomly.

    Returns:
        The outline template string.
    """
    templates = OUTLINE_TEMPLATES.get(article_type)
    if not templates:
        templates = OUTLINE_TEMPLATES["informational"]
    if variant is None:
        return random.choice(templates)
    return templates[variant % len(templates)]


def get_random_template(article_type: str) -> str:
    """Return a random variant for the given article type."""
    return get_template(article_type, variant=None)


def rotate_template(article_type: str, index: int) -> str:
    """Return a variant by rotating through available templates.

    Uses modulo so index can grow without bound and wraps around.
    """
    templates = OUTLINE_TEMPLATES.get(article_type, OUTLINE_TEMPLATES["informational"])
    return templates[index % len(templates)]


def get_template_for_title(title: str, index: int = None) -> str:
    """Classify a title and return an appropriate outline template.

    Convenience function that combines article_types.classify_title()
    with template selection.

    Args:
        title: The article title to classify and get a template for.
        index: If provided, uses rotation instead of random selection.

    Returns:
        An outline template string appropriate for the article type.
    """
    from .article_types import classify_title
    article_type = classify_title(title)
    if index is not None:
        return rotate_template(article_type, index)
    return get_random_template(article_type)


def get_all_types() -> list:
    """Return all available article type keys."""
    return list(OUTLINE_TEMPLATES.keys())


def get_variant_count(article_type: str) -> int:
    """Return the number of variants for an article type."""
    return len(OUTLINE_TEMPLATES.get(article_type, []))
