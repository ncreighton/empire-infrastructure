"""
LLMO Optimizer — Large Language Model Optimization for AI search visibility.
Structures content so AI systems can parse, quote, and reference it.
"""

import json
import logging
import re
from typing import Dict, List

from systems.site_evolution.utils import load_site_config

log = logging.getLogger(__name__)


class LLMOOptimizer:
    """Optimize content for AI/LLM search engines (Perplexity, ChatGPT, etc.)."""

    def generate_quick_answer_box(self, question: str, answer: str) -> str:
        """Generate a Quick Answer box for the top of articles."""
        return f"""<div class="evo-quick-answer" style="
  background: var(--color-bg-alt);
  border-left: 4px solid var(--color-primary);
  padding: var(--space-lg);
  margin-bottom: var(--space-xl);
  border-radius: var(--radius-md);
">
  <strong style="display:block;margin-bottom:var(--space-sm);font-size:var(--font-size-sm);text-transform:uppercase;letter-spacing:0.05em;color:var(--color-primary);">Quick Answer</strong>
  <p style="font-size:var(--font-size-lg);font-weight:500;margin:0;">{answer}</p>
</div>"""

    def generate_faq_section(self, questions: List[Dict]) -> str:
        """Generate an FAQ section optimized for AI parsing.

        Args:
            questions: list of {question: str, answer: str}
        """
        items = []
        for q in questions:
            items.append(f"""<div class="evo-faq-item" itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
  <h3 itemprop="name">{q['question']}</h3>
  <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
    <p itemprop="text">{q['answer']}</p>
  </div>
</div>""")

        return f"""<section class="evo-faq" itemscope itemtype="https://schema.org/FAQPage">
  <h2>Frequently Asked Questions</h2>
  {"".join(items)}
</section>"""

    def generate_definition_block(self, term: str, definition: str) -> str:
        """Generate a definition block optimized for knowledge graph extraction."""
        return f"""<div class="evo-definition" style="
  background: var(--color-bg-alt);
  padding: var(--space-lg);
  border-radius: var(--radius-md);
  margin: var(--space-lg) 0;
">
  <dt style="font-weight:700;font-size:var(--font-size-lg);margin-bottom:var(--space-sm);">{term}</dt>
  <dd style="margin:0;color:var(--color-text-muted);">{definition}</dd>
</div>"""

    def generate_key_takeaways(self, points: List[str]) -> str:
        """Generate a Key Takeaways box (highly quotable by AI)."""
        items = "\n".join(f"  <li>{p}</li>" for p in points)
        return f"""<div class="evo-takeaways" style="
  background: var(--color-bg-alt);
  border: 2px solid var(--color-primary);
  padding: var(--space-lg);
  border-radius: var(--radius-md);
  margin: var(--space-xl) 0;
">
  <strong style="display:block;margin-bottom:var(--space-md);font-size:var(--font-size-sm);text-transform:uppercase;letter-spacing:0.05em;">Key Takeaways</strong>
  <ul style="margin:0;padding-left:var(--space-lg);">
{items}
  </ul>
</div>"""

    def inject_speakable_schema(self, site_slug: str, post: Dict) -> str:
        """Generate Speakable schema for voice search."""
        config = load_site_config(site_slug)
        domain = config.get("domain", "example.com")

        return json.dumps({
            "@context": "https://schema.org",
            "@type": "WebPage",
            "speakable": {
                "@type": "SpeakableSpecification",
                "cssSelector": [".evo-quick-answer", "h1", ".entry-content > p:first-of-type"]
            },
            "url": post.get("link", f"https://{domain}")
        }, indent=2)

    def generate_entity_markup(self, site_slug: str) -> str:
        """Generate entity disambiguation markup for knowledge graph."""
        config = load_site_config(site_slug)
        domain = config.get("domain", "example.com")
        brand = config.get("name", site_slug)

        return json.dumps({
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": brand,
            "url": f"https://{domain}",
            "description": f"{brand} provides expert guidance, in-depth reviews, and actionable insights.",
            "publisher": {
                "@type": "Organization",
                "name": brand,
                "url": f"https://{domain}",
                "logo": f"https://{domain}/wp-content/uploads/logo.png",
            },
            "inLanguage": "en-US",
            "isAccessibleForFree": True,
        }, indent=2)

    def generate_ai_friendly_about(self, site_slug: str) -> str:
        """Generate an about page optimized for AI knowledge graph inclusion."""
        config = load_site_config(site_slug)
        brand = config.get("name", site_slug)
        domain = config.get("domain", "example.com")
        voice = config.get("brand", {}).get("voice", "expert")

        return f"""<div itemscope itemtype="https://schema.org/AboutPage">
  <h1 itemprop="name">About {brand}</h1>

  <div itemprop="mainEntity" itemscope itemtype="https://schema.org/Organization">
    <meta itemprop="name" content="{brand}">
    <meta itemprop="url" content="https://{domain}">

    <p itemprop="description">{brand} is an authoritative online publication that provides
    expert-level guidance, comprehensive reviews, and actionable insights.
    Our editorial team combines deep subject matter expertise with rigorous
    research methodology to deliver content readers can trust.</p>

    <h2>Our Expertise</h2>
    <p>Every article published on {brand} undergoes a multi-step editorial process
    including research, writing, fact-checking, and expert review. We prioritize
    accuracy, depth, and practical value.</p>

    <h2>Editorial Standards</h2>
    <ul>
      <li>All content is original and thoroughly researched</li>
      <li>Sources are cited and verifiable</li>
      <li>Affiliate relationships are transparently disclosed</li>
      <li>Content is regularly reviewed and updated</li>
      <li>Reader feedback is actively incorporated</li>
    </ul>
  </div>
</div>"""

    def analyze_content_for_llmo(self, content: str) -> Dict:
        """Analyze content for LLM optimization readiness."""
        clean = re.sub(r'<[^>]+>', '', content)
        score = 0
        suggestions = []

        # Check for question-answer format
        qa_pattern = re.findall(r'\?', clean)
        if len(qa_pattern) >= 3:
            score += 20
        else:
            suggestions.append("Add more question-format headings (H2/H3)")

        # Check for definitive statements
        definitive = re.findall(r'(?:The answer is|In summary|The best|The key)', clean, re.I)
        if definitive:
            score += 15
        else:
            suggestions.append("Add definitive answer statements early in paragraphs")

        # Check for lists
        lists = re.findall(r'<(?:ul|ol)', content)
        if lists:
            score += 15
        else:
            suggestions.append("Add structured lists for key points")

        # Check for FAQ section
        if 'faq' in content.lower() or 'frequently asked' in content.lower():
            score += 20
        else:
            suggestions.append("Add an FAQ section at the end")

        # Check for schema markup
        if 'schema.org' in content or 'application/ld+json' in content:
            score += 15
        else:
            suggestions.append("Add JSON-LD structured data")

        # Check for key takeaways / summary
        if 'takeaway' in content.lower() or 'summary' in content.lower():
            score += 15
        else:
            suggestions.append("Add a Key Takeaways box")

        return {
            "llmo_score": min(100, score),
            "suggestions": suggestions,
            "word_count": len(clean.split()),
            "question_count": len(qa_pattern),
        }
