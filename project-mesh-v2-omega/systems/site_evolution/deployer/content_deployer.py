"""
Content Deployer — WordPress page content deployment.
Creates Gutenberg block content for homepages, standard pages, and schema injection.
"""

import json
import logging
from typing import Dict, List, Optional

from systems.site_evolution.deployer.wp_deployer import WPDeployer, _load_sites

log = logging.getLogger(__name__)


class ContentDeployer:
    """Deploy page content and Gutenberg blocks to WordPress sites."""

    def __init__(self):
        self.deployer = WPDeployer()

    def _get_site_config(self, site_slug: str) -> Dict:
        sites = _load_sites()
        return sites.get(site_slug, {})

    def _to_gutenberg_html(self, html: str) -> str:
        """Wrap raw HTML in a Gutenberg custom HTML block."""
        return f'<!-- wp:html -->\n{html}\n<!-- /wp:html -->'

    def _to_gutenberg_blocks(self, sections: List[Dict]) -> str:
        """Convert a list of section dicts to Gutenberg block content.

        Each section: {type: 'html'|'paragraph'|'heading'|'group', content: str, level: int}
        """
        blocks = []
        for section in sections:
            stype = section.get("type", "html")
            content = section.get("content", "")

            if stype == "html":
                blocks.append(f'<!-- wp:html -->\n{content}\n<!-- /wp:html -->')
            elif stype == "paragraph":
                blocks.append(f'<!-- wp:paragraph -->\n<p>{content}</p>\n<!-- /wp:paragraph -->')
            elif stype == "heading":
                level = section.get("level", 2)
                blocks.append(
                    f'<!-- wp:heading {{"level":{level}}} -->\n'
                    f'<h{level}>{content}</h{level}>\n'
                    f'<!-- /wp:heading -->'
                )
            elif stype == "group":
                inner = section.get("inner_blocks", "")
                blocks.append(
                    f'<!-- wp:group -->\n<div class="wp-block-group">\n'
                    f'{inner}\n</div>\n<!-- /wp:group -->'
                )

        return "\n\n".join(blocks)

    def deploy_homepage(self, site_slug: str, html_sections: List[str],
                        dry_run: bool = False) -> Dict:
        """Deploy homepage content as Gutenberg blocks."""
        blocks = []
        for html in html_sections:
            blocks.append(self._to_gutenberg_html(html))

        full_content = "\n\n".join(blocks)

        if dry_run:
            return {
                "site": site_slug,
                "action": "deploy_homepage",
                "dry_run": True,
                "content_length": len(full_content),
                "block_count": len(blocks),
                "preview": full_content[:500],
            }

        return self.deployer.deploy_page(
            site_slug,
            title="Home",
            content=full_content,
            slug="home",
            template="page-templates/full-width.php",
        )

    def deploy_standard_pages(self, site_slug: str,
                              dry_run: bool = False) -> Dict:
        """Deploy all 7 standard pages customized per brand."""
        config = self._get_site_config(site_slug)
        brand_name = config.get("name", site_slug)
        domain = config.get("domain", "example.com")

        pages = {
            "about": {
                "title": f"About {brand_name}",
                "content": self._generate_about_page(config),
            },
            "contact": {
                "title": "Contact Us",
                "content": self._generate_contact_page(config),
            },
            "privacy-policy": {
                "title": "Privacy Policy",
                "content": self._generate_privacy_policy(brand_name, domain),
            },
            "terms-of-service": {
                "title": "Terms of Service",
                "content": self._generate_terms(brand_name, domain),
            },
            "affiliate-disclosure": {
                "title": "Affiliate Disclosure",
                "content": self._generate_affiliate_disclosure(brand_name),
            },
            "cookie-policy": {
                "title": "Cookie Policy",
                "content": self._generate_cookie_policy(brand_name, domain),
            },
            "404": {
                "title": "Page Not Found",
                "content": self._generate_404_page(config),
            },
        }

        results = {}
        for slug, page_data in pages.items():
            if dry_run:
                results[slug] = {
                    "title": page_data["title"],
                    "content_length": len(page_data["content"]),
                    "dry_run": True,
                }
            else:
                try:
                    result = self.deployer.deploy_page(
                        site_slug,
                        title=page_data["title"],
                        content=self._to_gutenberg_html(page_data["content"]),
                        slug=slug,
                    )
                    results[slug] = {"status": "deployed", "id": result.get("id")}
                except Exception as e:
                    results[slug] = {"status": "error", "error": str(e)}

        return {"site": site_slug, "pages": results}

    def _generate_about_page(self, config: Dict) -> str:
        brand = config.get("name", "Our Site")
        voice = config.get("brand", {}).get("voice", "expert")
        tagline = config.get("brand", {}).get("tagline", "")
        return f"""<div class="about-page">
<h1>About {brand}</h1>
<p class="lead">{tagline or f'Welcome to {brand} — your trusted source for expert guidance.'}</p>

<h2>Our Mission</h2>
<p>At {brand}, we believe everyone deserves access to high-quality, well-researched information.
Our team of passionate writers and subject matter experts works tirelessly to bring you
actionable insights you can trust.</p>

<h2>What We Cover</h2>
<p>Every article on {brand} is meticulously researched, fact-checked, and written
with our readers' needs in mind. We prioritize accuracy, depth, and practical value
in everything we publish.</p>

<h2>Our Editorial Standards</h2>
<ul>
<li>All content is original and thoroughly researched</li>
<li>We disclose affiliate relationships transparently</li>
<li>Our reviews are honest and based on real experience</li>
<li>We regularly update articles to keep information current</li>
</ul>

<h2>Contact Us</h2>
<p>Have a question, suggestion, or just want to say hello?
We'd love to hear from you. Visit our <a href="/contact">contact page</a>.</p>
</div>"""

    def _generate_contact_page(self, config: Dict) -> str:
        brand = config.get("name", "Our Site")
        return f"""<div class="contact-page">
<h1>Contact {brand}</h1>
<p>We'd love to hear from you! Whether you have a question, feedback, or a partnership inquiry,
please don't hesitate to reach out.</p>

<h2>Get in Touch</h2>
<p>The best way to reach us is via email. We typically respond within 24-48 hours.</p>

<h2>Partnership & Collaboration</h2>
<p>Interested in working with us? We're always open to collaborations that benefit our readers.
Please include details about your proposal in your message.</p>

<h2>Report an Issue</h2>
<p>Found an error or outdated information? We appreciate your help keeping our content accurate.
Please let us know which article and what needs updating.</p>
</div>"""

    def _generate_privacy_policy(self, brand: str, domain: str) -> str:
        return f"""<div class="legal-page">
<h1>Privacy Policy</h1>
<p><strong>Last updated:</strong> {__import__('datetime').date.today().isoformat()}</p>

<p>{brand} ("{domain}") respects your privacy. This policy explains what information
we collect, how we use it, and your rights.</p>

<h2>Information We Collect</h2>
<p>We may collect: email addresses (when voluntarily provided), usage data via analytics,
and cookies for site functionality.</p>

<h2>How We Use Information</h2>
<ul>
<li>To improve our content and user experience</li>
<li>To send newsletters (only with your consent)</li>
<li>To analyze site traffic and performance</li>
</ul>

<h2>Third-Party Services</h2>
<p>We use Google Analytics, Google AdSense, and affiliate networks (Amazon Associates).
These services may collect data per their own privacy policies.</p>

<h2>Your Rights</h2>
<p>You may request access to, correction, or deletion of your personal data
by contacting us. You can opt out of email communications at any time.</p>

<h2>Cookies</h2>
<p>See our <a href="/cookie-policy">Cookie Policy</a> for details.</p>
</div>"""

    def _generate_terms(self, brand: str, domain: str) -> str:
        return f"""<div class="legal-page">
<h1>Terms of Service</h1>
<p><strong>Last updated:</strong> {__import__('datetime').date.today().isoformat()}</p>

<p>By using {brand} ("{domain}"), you agree to these terms.</p>

<h2>Content</h2>
<p>All content is for informational purposes only. We strive for accuracy but make no
warranties. Always consult relevant professionals for specific advice.</p>

<h2>Intellectual Property</h2>
<p>All content, images, and branding on {domain} are owned by {brand}.
You may not reproduce our content without permission.</p>

<h2>Affiliate Links</h2>
<p>Some links on our site are affiliate links. See our
<a href="/affiliate-disclosure">Affiliate Disclosure</a> for details.</p>

<h2>Limitation of Liability</h2>
<p>{brand} is not liable for any damages arising from the use of our content
or linked third-party services.</p>
</div>"""

    def _generate_affiliate_disclosure(self, brand: str) -> str:
        return f"""<div class="legal-page">
<h1>Affiliate Disclosure</h1>
<p>{brand} is a participant in the Amazon Services LLC Associates Program and other
affiliate advertising programs. This means we may earn commissions on qualifying
purchases made through our links.</p>

<h2>How It Works</h2>
<p>When you click on product links and make a purchase, we may receive a small commission
at no additional cost to you. This helps us maintain our site and continue producing
high-quality content.</p>

<h2>Our Promise</h2>
<ul>
<li>Affiliate relationships never influence our editorial content or reviews</li>
<li>We only recommend products we genuinely believe in</li>
<li>All affiliate links are clearly identified</li>
<li>Our opinions are always our own</li>
</ul>
</div>"""

    def _generate_cookie_policy(self, brand: str, domain: str) -> str:
        return f"""<div class="legal-page">
<h1>Cookie Policy</h1>
<p><strong>Last updated:</strong> {__import__('datetime').date.today().isoformat()}</p>

<p>{brand} uses cookies to improve your experience.</p>

<h2>Types of Cookies We Use</h2>
<ul>
<li><strong>Essential:</strong> Required for site functionality</li>
<li><strong>Analytics:</strong> Google Analytics to understand site usage</li>
<li><strong>Advertising:</strong> Google AdSense for relevant ads</li>
<li><strong>Preferences:</strong> Remember your settings</li>
</ul>

<h2>Managing Cookies</h2>
<p>You can control cookies through your browser settings. Disabling cookies
may affect site functionality.</p>
</div>"""

    def _generate_404_page(self, config: Dict) -> str:
        brand = config.get("name", "Our Site")
        ctas = config.get("ctas", ["Go Home"])
        return f"""<div class="error-404-page" style="text-align:center;padding:60px 20px;">
<h1 style="font-size:4rem;margin-bottom:10px;">404</h1>
<h2>Page Not Found</h2>
<p>Looks like this page has vanished. Don't worry — there's plenty more to explore on {brand}.</p>
<div style="margin-top:30px;">
<a href="/" class="btn btn-primary" style="padding:12px 30px;text-decoration:none;">{ctas[0] if ctas else 'Go Home'}</a>
</div>
</div>"""
