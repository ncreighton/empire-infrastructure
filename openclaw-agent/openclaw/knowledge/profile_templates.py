"""Profile content template pools — bio, tagline, description variations by category.

Pure data module. No AI calls — all templates are curated strings with
{brand_name} placeholders for runtime interpolation.
"""

import random
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Tagline pools by platform category (short, punchy, 60-120 chars)
# ─────────────────────────────────────────────────────────────────────────────

TAGLINES: dict[str, list[str]] = {
    "ai_marketplace": [
        "Building AI tools that automate the mundane | {brand_name}",
        "Custom AI agents & automation workflows",
        "{brand_name} — Your AI productivity partner",
        "Practical AI solutions for real-world problems",
        "AI agents that work while you sleep | {brand_name}",
        "From idea to automation in minutes",
        "Open-source AI tools built for builders",
        "Shipping AI agents weekly | {brand_name}",
        "{brand_name}: AI tools that actually save you time",
        "Making AI accessible, one agent at a time",
    ],
    "workflow_marketplace": [
        "Automation workflows that just work | {brand_name}",
        "Connecting apps, eliminating busywork",
        "{brand_name} — Battle-tested automation templates",
        "n8n & Make workflows for teams that ship fast",
        "Workflow automation for the no-code generation",
        "Pre-built integrations, zero setup headaches",
        "Automate everything. Ship faster. | {brand_name}",
        "Workflow blueprints from the automation trenches",
        "{brand_name}: Plug-and-play automation for every stack",
        "Less clicking, more building",
    ],
    "digital_product": [
        "Digital tools & templates for modern creators | {brand_name}",
        "Premium digital products, instant delivery",
        "{brand_name} — Tools that amplify your output",
        "Creator-built resources for ambitious professionals",
        "Digital assets designed to save you hours",
        "Templates, tools & systems for 10x productivity",
        "{brand_name}: Ship better work, faster",
        "Practical digital products for serious builders",
        "Built by creators, for creators | {brand_name}",
        "High-quality digital resources, no fluff",
    ],
    "education": [
        "Learn AI & automation — practical, hands-on courses | {brand_name}",
        "Teaching the skills that future-proof your career",
        "{brand_name} — From zero to automated in one course",
        "Real-world AI skills, not academic theory",
        "Courses built by practitioners, not professors",
        "Master automation with step-by-step guidance",
        "{brand_name}: The shortest path from learning to earning",
        "Hands-on courses that get you building on day one",
        "AI education for the builder mindset | {brand_name}",
        "Stop watching tutorials. Start building. | {brand_name}",
    ],
    "prompt_marketplace": [
        "Engineered prompts for consistent AI output | {brand_name}",
        "Prompt engineering meets real-world use cases",
        "{brand_name} — Prompts that work on the first try",
        "AI configurations tuned for production quality",
        "Copy, paste, profit. Premium AI prompts.",
        "System prompts & configs for ChatGPT, Claude & more",
        "{brand_name}: Tested prompts, proven results",
        "Stop guessing. Use prompts that deliver.",
        "AI prompt templates for every workflow | {brand_name}",
        "Crafted prompts for reliable, repeatable AI output",
    ],
    "3d_models": [
        "3D models & printable designs for makers | {brand_name}",
        "Print-ready 3D assets, designed for quality",
        "{brand_name} — Functional 3D models, ready to print",
        "From digital design to physical reality",
        "Premium STL files & 3D assets for every project",
        "3D printable designs crafted with precision",
        "{brand_name}: Models that work on the first print",
        "Bringing ideas to life, one layer at a time",
        "Maker-tested 3D designs | {brand_name}",
        "Quality 3D models for hobbyists and pros alike",
    ],
    "general": [
        "Building tools for builders | {brand_name}",
        "{brand_name} — Shipping useful things on the internet",
        "Creator, builder, automation enthusiast",
        "Making technology work harder so you don't have to",
        "Digital tools & resources for ambitious people",
        "Practical solutions from {brand_name}",
        "{brand_name}: Build more. Click less.",
        "Tools, templates & automations for modern work",
        "Helping you work smarter | {brand_name}",
        "Less grind, more output. That's the mission.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Bio pools (short-form, 150-300 chars)
# ─────────────────────────────────────────────────────────────────────────────

BIOS: dict[str, list[str]] = {
    "ai_marketplace": [
        "I build AI agents and automation tools that solve real problems. Every tool I ship is something I use daily in my own workflows. {brand_name} is where I share what works.",
        "Software developer and AI enthusiast creating practical agents for content, analytics, and workflow automation. Focused on tools that save hours, not minutes.",
        "Building at the intersection of AI and automation. I create agents, plugins, and integrations that help teams ship faster. Everything is battle-tested in production.",
        "{brand_name} creates AI-powered tools for creators and small teams. From content generation to data pipelines, every product is built to be useful on day one.",
        "Full-stack developer turned AI builder. I publish open-source agents, custom GPTs, and automation workflows. Currently shipping new tools weekly.",
        "AI tools for people who build things. I create agents that handle the repetitive work so you can focus on what matters. New releases every week.",
        "Automation-first developer building AI agents for the modern creator stack. My tools are used by hundreds of teams to streamline their daily operations.",
        "I write code that writes code. {brand_name} builds practical AI tools — agents, workflows, and integrations — for people who value their time.",
    ],
    "workflow_marketplace": [
        "Automation architect building production-ready workflows for n8n, Make, and custom integrations. Every template is tested and documented for easy deployment.",
        "I create workflow automations that eliminate busywork. Specializing in multi-step integrations between SaaS tools, databases, and AI services.",
        "{brand_name} publishes plug-and-play automation templates. Each workflow is battle-tested across real business operations before listing.",
        "Building the automation layer between your favorite tools. I specialize in n8n workflows, API integrations, and data pipeline templates.",
        "Workflow designer focused on reliability and clarity. Every automation I publish includes documentation, error handling, and real-world examples.",
        "No-code and low-code automation expert. I build workflow templates that connect 50+ services and handle edge cases gracefully.",
        "I automate business processes so teams can focus on creative work. My workflows handle everything from lead capture to content distribution.",
        "Automation consultant turned template creator. {brand_name} publishes the workflows I wish existed when I started automating.",
    ],
    "digital_product": [
        "Creator of digital tools, templates, and systems for productive professionals. Everything I sell is something I built for my own use first.",
        "{brand_name} creates premium digital products for creators and solopreneurs. Templates, toolkits, and systems designed to save you hours each week.",
        "I build digital products that help people work smarter. From Notion templates to automation toolkits, every product solves a specific problem.",
        "Digital product creator focused on quality over quantity. Each resource is designed, tested, and refined before it reaches your hands.",
        "Building practical digital resources for ambitious professionals. My products span productivity tools, content templates, and business systems.",
        "I create digital tools for people who take their craft seriously. Every product is documented, maintained, and updated regularly.",
        "{brand_name} — creating digital assets that earn their price on the first use. Templates, tools, and frameworks for modern professionals.",
        "Maker of digital products that actually get used. I focus on solving one problem well, not shipping everything at once.",
    ],
    "education": [
        "Teaching AI and automation through hands-on projects. My courses focus on building real tools, not watching slides. Learn by shipping.",
        "Instructor and builder creating courses on AI, automation, and modern development. Every lesson comes with working code and practical exercises.",
        "{brand_name} teaches the skills that matter. My courses cover AI development, workflow automation, and the tools that top builders use daily.",
        "I believe the best way to learn is to build. My courses take you from concept to deployed project in hours, not months.",
        "Experienced developer turned educator. I create project-based courses that teach you to build AI tools, not just understand them.",
        "Teaching the next generation of AI builders. My courses are practical, project-focused, and updated monthly to match the latest tools.",
        "I teach automation and AI development for people who learn by doing. Each course is built around a real project you can ship.",
        "Educator focused on practical AI skills. {brand_name} courses are designed for builders who want to create, not just consume.",
    ],
    "prompt_marketplace": [
        "Prompt engineer creating production-tested AI configurations. Every prompt is refined through hundreds of iterations for consistent, reliable output.",
        "I design system prompts and AI configurations that deliver professional results. Specializing in ChatGPT, Claude, and image generation workflows.",
        "{brand_name} publishes prompt templates that work on the first try. Each prompt includes usage guides, example outputs, and tuning instructions.",
        "Building the best prompt library on the internet. Every template is tested across models, documented with examples, and optimized for your use case.",
        "AI configuration specialist creating prompts for content, code, analysis, and creative work. Tested on real projects, not synthetic benchmarks.",
        "I turn vague AI requests into precise, repeatable prompts. My templates help you get consistent output without prompt engineering expertise.",
        "Prompt designer focused on production quality. Every prompt I sell has been through dozens of refinement cycles with real users.",
        "Creating AI prompts that save you the trial-and-error phase. {brand_name} templates get you professional output from your first message.",
    ],
    "3d_models": [
        "3D designer creating print-ready models for makers, hobbyists, and professionals. Every design is tested on real printers before listing.",
        "{brand_name} publishes functional 3D models and printable designs. From home decor to mechanical parts, all models are print-tested and optimized.",
        "I design 3D models that work in the real world. Every STL is optimized for FDM and resin printing with detailed print settings included.",
        "Maker and 3D designer creating practical printable models. My designs focus on function, fit, and printability above all else.",
        "3D print designer specializing in functional models, articulated designs, and home decor. Print-tested on multiple machines for reliable results.",
        "Creating 3D printable designs for the maker community. {brand_name} models include supports guidance, material recommendations, and assembly instructions.",
        "I design models that bridge digital and physical. Every file is tested, documented, and optimized for successful first prints.",
        "3D artist and maker publishing print-ready designs. My models are used by thousands of makers worldwide across every printer type.",
    ],
    "general": [
        "Creator and builder shipping digital products, AI tools, and automation workflows. {brand_name} is where I publish what works.",
        "I build useful things on the internet. From AI agents to digital products, everything I create solves a real problem.",
        "{brand_name} — building tools, templates, and automations for people who value their time. New products shipped regularly.",
        "Maker, builder, and automation enthusiast. I create digital products that help people work smarter and ship faster.",
        "Building at the intersection of AI, automation, and practical utility. My products are designed for people who build things.",
        "I create tools for creators. {brand_name} publishes AI agents, workflow templates, and digital products used by thousands.",
        "Full-stack creator building across AI, automation, and digital products. Everything I ship is something I use myself.",
        "Digital builder focused on practical utility. I publish tools, templates, and systems that earn back your investment on day one.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Description pools (long-form, 500-2000 chars)
# ─────────────────────────────────────────────────────────────────────────────

DESCRIPTIONS: dict[str, list[str]] = {
    "ai_marketplace": [
        "{brand_name} builds AI agents and automation tools for creators, developers, and small teams.\n\nEvery tool starts as a solution to a real problem I face in my own work. If it saves me time, I package it up and share it. That means no vaporware — everything listed here is actively used in production.\n\nWhat you'll find:\n- AI agents for content creation, data analysis, and workflow automation\n- Pre-built integrations with popular tools and APIs\n- Open-source projects with commercial support options\n- Detailed documentation and setup guides\n\nI ship new tools weekly and update existing ones based on user feedback. If something breaks, I fix it. If something could be better, I improve it.\n\nBuilt for builders. No fluff, no hype — just tools that work.",
        "Welcome to {brand_name} — where practical AI meets real-world automation.\n\nI'm a developer who got tired of seeing AI demos that never become useful products. So I started building the tools I wished existed: agents that handle repetitive tasks, integrations that connect your stack, and workflows that run reliably without babysitting.\n\nMy focus areas:\n- Custom AI agents (Claude, GPT, and open-source models)\n- Multi-step automation workflows\n- API integrations and data pipelines\n- Developer tools and CLI utilities\n\nEvery product is:\n- Battle-tested in my own operations\n- Documented with setup guides and examples\n- Maintained and updated regularly\n- Backed by responsive support\n\nWhether you're automating content creation, streamlining operations, or building your own AI toolkit, you'll find something useful here.",
    ],
    "workflow_marketplace": [
        "{brand_name} creates production-ready automation workflows for teams that value reliability over novelty.\n\nEvery template in my catalog has been built, broken, debugged, and refined in real business environments. I don't publish workflows based on tutorials — I publish the ones that survived contact with messy real-world data.\n\nWhat I specialize in:\n- n8n workflow templates with full error handling\n- Make (Integromat) scenarios with retry logic\n- Multi-platform integrations (CRM, email, analytics, AI)\n- Data pipeline and ETL automation\n\nEvery workflow includes:\n- Step-by-step setup documentation\n- Environment variable configuration guides\n- Error handling and fallback logic\n- Test data and validation steps\n\nI update templates when APIs change and respond to issues quickly. Automation should reduce your workload, not add to it.",
        "Automation shouldn't be fragile. {brand_name} publishes workflow templates built for the real world.\n\nAfter years of building custom automations for businesses, I've packaged the most useful patterns into reusable templates. Each one handles the edge cases that tutorials skip: API rate limits, data validation, retry logic, and graceful error recovery.\n\nMy workflow catalog covers:\n- Lead capture and CRM automation\n- Content distribution and scheduling\n- Data synchronization between platforms\n- Reporting and analytics pipelines\n- AI-enhanced processing workflows\n\nEvery template is documented, tested, and includes configuration notes for your specific stack. I also provide setup support for complex integrations.\n\nBrowse the catalog, grab a template, and have your automation running in minutes — not days.",
    ],
    "digital_product": [
        "{brand_name} creates digital products for professionals who refuse to waste time on busy work.\n\nEvery product in this store started as a tool I built to solve my own problem. If it worked well enough that I kept using it, I polished it and made it available to others. That's my quality bar.\n\nWhat you'll find here:\n- Productivity templates and systems\n- Automation toolkits and configurations\n- Business frameworks and planning tools\n- Creator resources and content templates\n\nMy philosophy:\n- Every product should pay for itself on the first use\n- Documentation matters as much as the product\n- Updates are free, forever\n- If it's not useful, it doesn't get listed\n\nI'm a builder who respects your time. No upsell funnels, no subscription traps. Buy the tool, use the tool, save the time.",
        "Welcome to {brand_name} — premium digital products built by a creator who actually uses them.\n\nI believe the best digital products are the ones their creators can't live without. That's why everything in this store is extracted from my own daily workflows. If I stopped using it, I'd pull it from the store.\n\nProduct categories:\n- Notion templates and workspace systems\n- Content creation toolkits\n- Business operations frameworks\n- Automation configurations and guides\n- Developer productivity tools\n\nEvery product includes detailed documentation, video walkthroughs where appropriate, and lifetime updates. I actively maintain everything I sell and incorporate user feedback into updates.\n\nMy goal is simple: build things that save you more time than they cost. If a product doesn't deliver on that promise, reach out and I'll make it right.",
    ],
    "education": [
        "{brand_name} teaches AI and automation through building, not lecturing.\n\nEvery course is structured around a real project you'll complete by the end. No abstract theory slides — just hands-on coding, building, and deploying. By the time you finish, you'll have a working tool, not just a certificate.\n\nWhat I teach:\n- AI agent development (Claude, GPT, open-source models)\n- Workflow automation (n8n, Make, custom scripts)\n- API integration and data pipelines\n- Digital product creation and monetization\n\nCourse philosophy:\n- Project-first: you build something real in every course\n- Updated monthly: AI moves fast, courses keep up\n- Community support: Discord access for Q&A and feedback\n- Practical focus: every skill taught is one I use professionally\n\nI've been building AI tools and automations professionally for years. These courses are the shortcut I wish I had when I started.",
        "Learn to build, not just understand. {brand_name} courses are designed for people who want to create real AI tools and automations.\n\nI created these courses because I kept seeing the same gap: tons of conceptual AI content, but very little that teaches you to ship a working product. My courses fill that gap.\n\nEach course includes:\n- A real project from start to finish\n- Working source code and starter templates\n- Step-by-step video instruction\n- Text-based reference guides for quick review\n- Community access for questions and collaboration\n\nTopics covered:\n- Building AI agents from scratch\n- Automating business operations\n- Creating and selling digital products\n- API development and integration\n- Deployment and monitoring\n\nCourses are updated regularly as tools and best practices evolve. Your purchase includes all future updates at no extra cost.",
    ],
    "prompt_marketplace": [
        "{brand_name} creates production-quality AI prompts that deliver consistent, professional results.\n\nEvery prompt in this store has been through extensive testing and refinement. I don't publish first drafts — each template goes through dozens of iterations across multiple models before it's listed.\n\nWhat makes my prompts different:\n- Tested on GPT-4, Claude, and open-source models\n- Include example outputs and edge case handling\n- Come with tuning guides for your specific use case\n- Updated when model behavior changes\n\nPrompt categories:\n- Content creation (articles, social media, email)\n- Code generation and review\n- Data analysis and summarization\n- Creative writing and brainstorming\n- Business and marketing copy\n\nEach prompt includes:\n- The system prompt / configuration\n- Usage instructions and best practices\n- 3-5 example inputs and outputs\n- Tips for customization\n\nStop spending hours tweaking prompts. Use tested templates and get back to your actual work.",
    ],
    "3d_models": [
        "{brand_name} designs functional 3D models that work in the real world, not just in the slicer preview.\n\nEvery model is designed for successful printing. That means proper wall thickness, appropriate tolerances, smart support placement, and tested print settings. I don't publish models that only look good in renders.\n\nWhat you'll find:\n- Functional home and office accessories\n- Articulated and mechanical designs\n- Organizers, holders, and storage solutions\n- Decorative models optimized for print quality\n\nEvery listing includes:\n- Print-tested STL files\n- Recommended print settings (layer height, infill, supports)\n- Material recommendations\n- Assembly instructions where applicable\n- Multiple size variants when useful\n\nI test every model on FDM printers (PLA and PETG) before listing. Resin-optimized versions are noted separately. If a model doesn't print well for you, contact me and I'll help troubleshoot or update the design.",
    ],
    "general": [
        "{brand_name} builds digital products, AI tools, and automation workflows for people who value quality and practicality.\n\nI'm a creator who believes in shipping useful things. Every product in my catalog exists because it solved a real problem — either for me or for someone who asked for it. No theoretical products, no vaporware.\n\nWhat I create:\n- AI agents and automation tools\n- Workflow templates and integrations\n- Digital products and templates\n- Courses and educational resources\n\nMy principles:\n- Everything is tested before publishing\n- Documentation is part of the product\n- Updates are free for the life of the product\n- Support is responsive and human\n\nI ship regularly and improve constantly. If you find something useful, great. If something doesn't work, let me know and I'll fix it. That's the deal.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Username patterns
# ─────────────────────────────────────────────────────────────────────────────

USERNAME_PATTERNS: list[str] = [
    "{brand_slug}",
    "{brand_slug}_official",
    "{brand_slug}ai",
    "{brand_slug}_ai",
    "{brand_slug}dev",
    "{brand_slug}_dev",
    "{brand_slug}hq",
    "{brand_slug}_hq",
    "{brand_slug}tools",
    "{brand_slug}_tools",
    "{brand_slug}io",
    "{brand_slug}labs",
    "{brand_slug}_labs",
    "the_{brand_slug}",
    "get{brand_slug}",
    "use{brand_slug}",
    "{brand_slug}studio",
    "{brand_slug}_studio",
    "hey{brand_slug}",
    "{brand_slug}works",
]


# ─────────────────────────────────────────────────────────────────────────────
# SEO keyword pools
# ─────────────────────────────────────────────────────────────────────────────

SEO_KEYWORDS: dict[str, list[str]] = {
    "ai_marketplace": [
        "AI agents",
        "automation tools",
        "AI tools",
        "custom GPT",
        "AI automation",
        "workflow automation",
        "AI productivity",
        "machine learning tools",
        "no-code AI",
        "AI integrations",
        "LLM tools",
        "AI assistants",
        "chatbot builder",
        "AI plugins",
        "Claude tools",
        "GPT tools",
    ],
    "workflow_marketplace": [
        "n8n workflows",
        "automation templates",
        "Make scenarios",
        "workflow automation",
        "API integration",
        "no-code automation",
        "Zapier alternative",
        "data pipeline",
        "business automation",
        "integration templates",
        "ETL workflows",
        "CRM automation",
        "email automation",
        "webhook workflows",
        "process automation",
    ],
    "digital_product": [
        "digital products",
        "Notion templates",
        "productivity tools",
        "digital downloads",
        "business templates",
        "creator tools",
        "content templates",
        "digital assets",
        "online business tools",
        "SaaS templates",
        "startup toolkit",
        "solopreneur resources",
        "productivity systems",
        "digital resources",
    ],
    "education": [
        "AI courses",
        "automation courses",
        "learn AI",
        "coding courses",
        "developer education",
        "hands-on courses",
        "project-based learning",
        "AI development",
        "workflow courses",
        "online courses",
        "tech education",
        "programming courses",
        "no-code courses",
        "AI certification",
    ],
    "prompt_marketplace": [
        "AI prompts",
        "ChatGPT prompts",
        "Claude prompts",
        "prompt engineering",
        "system prompts",
        "prompt templates",
        "AI configurations",
        "GPT prompts",
        "Midjourney prompts",
        "AI writing prompts",
        "prompt library",
        "production prompts",
        "tested prompts",
        "prompt marketplace",
    ],
    "3d_models": [
        "3D models",
        "STL files",
        "3D printing",
        "printable models",
        "3D print designs",
        "maker models",
        "FDM models",
        "resin models",
        "3D assets",
        "print-ready designs",
        "functional prints",
        "3D printable",
        "CAD models",
        "maker community",
    ],
    "general": [
        "AI tools",
        "automation",
        "digital products",
        "creator economy",
        "productivity",
        "templates",
        "workflows",
        "developer tools",
        "no-code",
        "online business",
        "digital assets",
        "solopreneur tools",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def get_tagline(category: str, **kwargs: Any) -> str:
    """Return a random tagline for the given category, formatted with kwargs."""
    pool = TAGLINES.get(category, TAGLINES["general"])
    template = random.choice(pool)
    try:
        return template.format(**kwargs) if kwargs else template
    except KeyError:
        return template


def get_bio(category: str, **kwargs: Any) -> str:
    """Return a random bio for the given category, formatted with kwargs."""
    pool = BIOS.get(category, BIOS["general"])
    template = random.choice(pool)
    try:
        return template.format(**kwargs) if kwargs else template
    except KeyError:
        return template


def get_description(category: str, **kwargs: Any) -> str:
    """Return a random description for the given category, formatted with kwargs."""
    pool = DESCRIPTIONS.get(category, DESCRIPTIONS["general"])
    template = random.choice(pool)
    try:
        return template.format(**kwargs) if kwargs else template
    except KeyError:
        return template


def get_username(brand_slug: str, max_length: int = 30) -> str:
    """Generate a username from brand_slug using a random pattern.

    Args:
        brand_slug: The base brand slug (lowercase, no spaces).
        max_length: Maximum allowed username length on the target platform.

    Returns:
        A username string that fits within the max_length constraint.
    """
    # Normalize the slug
    slug = brand_slug.lower().strip().replace(" ", "").replace("-", "_")

    # Try patterns in random order, return first that fits
    patterns = USERNAME_PATTERNS.copy()
    random.shuffle(patterns)

    for pattern in patterns:
        candidate = pattern.format(brand_slug=slug)
        if len(candidate) <= max_length:
            return candidate

    # Fallback: truncate the bare slug
    return slug[:max_length]


def get_seo_keywords(category: str, count: int = 5) -> list[str]:
    """Return a random sample of SEO keywords for the given category.

    Args:
        category: Platform category key.
        count: Number of keywords to return.

    Returns:
        A list of keyword strings.
    """
    pool = SEO_KEYWORDS.get(category, SEO_KEYWORDS["general"])
    return random.sample(pool, min(count, len(pool)))


def get_all_categories() -> list[str]:
    """Return all category keys that have templates defined."""
    return list(TAGLINES.keys())


def get_template_counts() -> dict[str, dict[str, int]]:
    """Return counts of templates per category per type for diagnostics."""
    result: dict[str, dict[str, int]] = {}
    all_cats = set(TAGLINES.keys()) | set(BIOS.keys()) | set(DESCRIPTIONS.keys())
    for cat in sorted(all_cats):
        result[cat] = {
            "taglines": len(TAGLINES.get(cat, [])),
            "bios": len(BIOS.get(cat, [])),
            "descriptions": len(DESCRIPTIONS.get(cat, [])),
            "seo_keywords": len(SEO_KEYWORDS.get(cat, [])),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Expansion: Code Repository & Social Platform categories
# ─────────────────────────────────────────────────────────────────────────────

TAGLINES["code_repository"] = [
    "Open-source tools for modern developers | {brand_name}",
    "Ship better code, faster",
    "Developer tools that eliminate boilerplate | {brand_name}",
    "{brand_name} — Code templates & deploy blueprints",
    "From prototype to production in one click",
    "Building the tools I wish existed | {brand_name}",
    "Open-source automation & developer utilities",
    "Code less, build more | {brand_name}",
]

TAGLINES["social_platform"] = [
    "Building in public | {brand_name}",
    "Indie maker shipping AI tools & automation",
    "{brand_name} — Maker of things that save time",
    "Turning ideas into products, one ship at a time",
    "AI tools, automation, and the journey of building | {brand_name}",
    "Solo founder building useful tools",
    "Shipping weekly, learning daily | {brand_name}",
    "Creating the future of work, one tool at a time",
]

BIOS["code_repository"] = [
    "Building open-source developer tools and deploy templates. Focused on making cloud infrastructure accessible to everyone.",
    "I create code templates, automation scripts, and developer utilities. Shipping tools that eliminate repetitive work.",
    "Full-stack developer building open-source tools. Templates, blueprints, and automation for the modern stack.",
    "Developer and maker at {brand_name}. I build tools that help other developers ship faster.",
    "Open-source contributor focused on automation and AI tooling. Code templates and deploy blueprints for modern apps.",
    "Turning common development patterns into reusable templates. Making deployment and infrastructure simple.",
]

BIOS["social_platform"] = [
    "Building AI tools and automation that help people work smarter. Sharing the journey openly.",
    "Indie maker focused on AI, automation, and digital products. Building in public, learning from every launch.",
    "Creator at {brand_name}. I build practical tools using AI and automation. Follow along for behind-the-scenes content.",
    "Solo founder shipping AI agents, automation workflows, and digital tools. Building useful things and sharing how.",
    "Maker of AI tools and digital products. Documenting the indie maker journey — the wins, the fails, all of it.",
    "I build things at the intersection of AI and automation. Sharing tools, learnings, and the real numbers.",
]

DESCRIPTIONS["code_repository"] = [
    "I build open-source developer tools, deploy templates, and automation scripts at {brand_name}. My focus is on eliminating boilerplate and making complex infrastructure accessible through simple, well-documented templates. Every tool is battle-tested in production before being shared. Check out my repositories for ready-to-use code.",
    "{brand_name} is where I publish developer tools and code templates. From CI/CD pipelines to full-stack deploy blueprints, everything here is designed to help you ship faster. All code is open-source and actively maintained.",
]

DESCRIPTIONS["social_platform"] = [
    "I'm the maker behind {brand_name}, where I build AI tools, automation workflows, and digital products. I believe in building in public — sharing not just the finished products, but the process, the mistakes, and the real numbers. Follow along for weekly updates on what I'm building and launching.",
    "Creator and solo founder building at {brand_name}. I focus on practical AI applications and automation that saves real time. When I'm not coding, I'm writing about the indie maker journey, product launches, and the tools that make it all possible.",
]

SEO_KEYWORDS["code_repository"] = [
    "open source", "developer tools", "code templates",
    "deploy blueprints", "automation scripts", "CI/CD",
    "full-stack templates", "infrastructure as code",
    "developer utilities", "cloud deploy", "boilerplate",
    "starter templates",
]

SEO_KEYWORDS["social_platform"] = [
    "indie maker", "building in public", "solo founder",
    "product launches", "AI tools", "automation",
    "digital products", "maker journey", "startup",
    "bootstrapped", "side projects", "indie hacker",
]


# ─────────────────────────────────────────────────────────────────────────────
# A/B Variant Tracking — track which templates perform best
# ─────────────────────────────────────────────────────────────────────────────

class TemplateTracker:
    """Track which template variants are used and their outcomes.

    Used by the VariationEngine to prefer better-performing templates.
    """

    def __init__(self):
        self._usage: dict[str, dict[int, int]] = {}  # "category:type" -> {index: use_count}
        self._scores: dict[str, dict[int, list[float]]] = {}  # "category:type" -> {index: [sentinel_scores]}

    def record_usage(self, category: str, template_type: str, index: int) -> None:
        """Record that a template was used."""
        key = f"{category}:{template_type}"
        if key not in self._usage:
            self._usage[key] = {}
        self._usage[key][index] = self._usage[key].get(index, 0) + 1

    def record_score(self, category: str, template_type: str, index: int, score: float) -> None:
        """Record the sentinel score achieved with this template."""
        key = f"{category}:{template_type}"
        if key not in self._scores:
            self._scores[key] = {}
        if index not in self._scores[key]:
            self._scores[key][index] = []
        self._scores[key][index].append(score)

    def get_best_indices(self, category: str, template_type: str, top_n: int = 3) -> list[int]:
        """Get the top N best-performing template indices by average score."""
        key = f"{category}:{template_type}"
        scores = self._scores.get(key, {})
        if not scores:
            return []
        avg_scores = {
            idx: sum(s) / len(s) for idx, s in scores.items() if s
        }
        sorted_indices = sorted(avg_scores, key=avg_scores.get, reverse=True)
        return sorted_indices[:top_n]

    def get_stats(self) -> dict[str, Any]:
        """Get usage and performance statistics."""
        stats = {}
        for key, usage in self._usage.items():
            scores = self._scores.get(key, {})
            stats[key] = {
                "total_uses": sum(usage.values()),
                "unique_templates_used": len(usage),
                "avg_score": (
                    sum(sum(s) / len(s) for s in scores.values() if s) / len(scores)
                    if scores else 0
                ),
            }
        return stats

    def reset(self) -> None:
        """Reset all tracking data."""
        self._usage.clear()
        self._scores.clear()


# Global tracker instance
template_tracker = TemplateTracker()
