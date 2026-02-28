"""Script frameworks — 6 narrative structures matched to content types.

Each framework defines a scene structure, prompt instructions, and best-fit
content types. The ScriptEngine selects the right framework based on the
video topic and niche category.
"""

SCRIPT_FRAMEWORKS = {
    "hook_problem_solution_cta": {
        "name": "Hook → Problem → Solution → CTA",
        "scene_structure": [
            "hook — pattern interrupt, specific claim or question",
            "problem — name the pain, make them feel it",
            "solution_1 — first fix, concrete and specific",
            "solution_2 — second fix, builds on the first",
            "solution_3 — third fix, the big one",
            "cta — direct ask, tie back to the hook",
        ],
        "prompt_instruction": (
            "Structure: HOOK (pattern interrupt with a specific claim) → "
            "PROBLEM (name the exact pain point — make the viewer feel it) → "
            "SOLUTION 1-3 (each one concrete and specific, building in impact) → "
            "CTA (direct ask that ties back to the hook). "
            "The hook must earn the next five seconds. The problem must feel personal. "
            "Each solution must be actionable in under sixty seconds."
        ),
        "best_for": ["educational", "tutorial", "listicle", "how_to"],
    },
    "pas": {
        "name": "Problem → Agitation → Solution",
        "scene_structure": [
            "hook — state the problem as a question or frustration",
            "agitate — make it worse, show what happens if they ignore it",
            "twist — reveal the real cause most people miss",
            "solution — the fix, with proof or specifics",
            "cta — urgency, act now",
        ],
        "prompt_instruction": (
            "Structure: PROBLEM (open with the exact frustration) → "
            "AGITATION (make it worse — what happens if they do nothing? what are they losing?) → "
            "TWIST (reveal the real cause nobody talks about) → "
            "SOLUTION (the fix, with specific proof or numbers) → "
            "CTA (create urgency). "
            "The agitation phase is the engine. Make it uncomfortable. "
            "The solution should feel like relief."
        ),
        "best_for": ["review", "comparison", "problem_solving"],
    },
    "bab": {
        "name": "Before → After → Bridge",
        "scene_structure": [
            "hook — paint the before picture vividly",
            "before — the old way, the struggle, the frustration",
            "after — the transformed result, specific and visual",
            "bridge — how to get from before to after, step by step",
            "cta — invitation to start the transformation",
        ],
        "prompt_instruction": (
            "Structure: HOOK (paint the 'before' picture — make it visceral) → "
            "BEFORE (the struggle, the old way, the frustration everyone knows) → "
            "AFTER (the transformation — be specific about the result, use numbers or visuals) → "
            "BRIDGE (how to get there — concrete steps, not vague advice) → "
            "CTA (invite them to start). "
            "The contrast between before and after is everything. "
            "Make the before feel real. Make the after feel inevitable."
        ),
        "best_for": ["transformation", "inspirational", "motivation", "before_after"],
    },
    "loop": {
        "name": "Open Loop → Tease → Deliver → Loop Back",
        "scene_structure": [
            "hook — open the loop with an impossible-sounding claim",
            "context — set the scene, build the world",
            "escalation — raise the stakes, add details",
            "revelation — deliver the payoff, close the first loop",
            "twist — open a new loop or reframe everything",
            "cta — leave them wanting more",
        ],
        "prompt_instruction": (
            "Structure: HOOK (open a loop — tease something impossible or shocking) → "
            "CONTEXT (set the scene, make them care about the characters or stakes) → "
            "ESCALATION (raise the tension, add surprising details, delay the payoff) → "
            "REVELATION (deliver — close the loop with a satisfying payoff) → "
            "TWIST (reframe everything or open a new question) → "
            "CTA (leave them wanting more). "
            "Never close a loop without opening another. "
            "The delay between tease and payoff is where retention lives."
        ),
        "best_for": ["story", "mythology", "folklore", "narrative", "history"],
    },
    "reverse_tell": {
        "name": "Result First → Work Backward",
        "scene_structure": [
            "hook — show the result, the headline, the number",
            "impact — why this matters, who it affects",
            "backstory — how we got here, the key events",
            "insight — the part nobody is talking about",
            "cta — what to do with this information",
        ],
        "prompt_instruction": (
            "Structure: HOOK (lead with the result — the number, the headline, the outcome) → "
            "IMPACT (why should they care? who does this affect?) → "
            "BACKSTORY (how did we get here? what were the key events?) → "
            "INSIGHT (the angle nobody else is covering) → "
            "CTA (what should they do with this information?). "
            "Start with the ending. Work backward. "
            "The hook IS the conclusion — everything after is the why and how."
        ),
        "best_for": ["news", "breaking", "analysis", "tech_update"],
    },
    "psp": {
        "name": "Problem → Solution → Proof",
        "scene_structure": [
            "hook — name the specific problem with a product or situation",
            "solution — what fixes it, be specific",
            "proof_1 — evidence, numbers, comparison",
            "proof_2 — second layer of proof, different angle",
            "verdict — clear recommendation",
            "cta — where to get it or what to do next",
        ],
        "prompt_instruction": (
            "Structure: HOOK (name the exact problem with a specific product or situation) → "
            "SOLUTION (what fixes it — name names, give specifics) → "
            "PROOF 1 (evidence: numbers, benchmarks, real comparison) → "
            "PROOF 2 (different angle: durability, user experience, value) → "
            "VERDICT (clear yes/no/maybe recommendation, no hedging) → "
            "CTA (where to get it or what to do next). "
            "This is a courtroom. Present evidence, not opinions. "
            "Every claim needs a specific number or comparison."
        ),
        "best_for": ["product_review", "gear_review", "comparison", "buyer_guide"],
    },
}


# Map content types to their best framework
CONTENT_TYPE_TO_FRAMEWORK = {
    "educational": "hook_problem_solution_cta",
    "tutorial": "hook_problem_solution_cta",
    "listicle": "hook_problem_solution_cta",
    "how_to": "hook_problem_solution_cta",
    "review": "psp",
    "product_review": "psp",
    "gear_review": "psp",
    "comparison": "pas",
    "buyer_guide": "psp",
    "story": "loop",
    "mythology": "loop",
    "folklore": "loop",
    "narrative": "loop",
    "history": "loop",
    "news": "reverse_tell",
    "breaking": "reverse_tell",
    "analysis": "reverse_tell",
    "tech_update": "reverse_tell",
    "transformation": "bab",
    "inspirational": "bab",
    "motivation": "bab",
    "before_after": "bab",
    "problem_solving": "pas",
}


# Per-category framework preference (first = default, rest = alternatives)
NICHE_FRAMEWORK_RANKING = {
    "witchcraft": ["hook_problem_solution_cta", "loop", "bab"],
    "mythology": ["loop", "reverse_tell", "hook_problem_solution_cta"],
    "tech": ["hook_problem_solution_cta", "psp", "pas"],
    "ai_news": ["reverse_tell", "hook_problem_solution_cta", "pas"],
    "lifestyle": ["hook_problem_solution_cta", "bab", "pas"],
    "fitness": ["psp", "pas", "hook_problem_solution_cta"],
    "business": ["hook_problem_solution_cta", "reverse_tell", "pas"],
}


def get_framework(content_type: str) -> dict:
    """Get the best framework for a content type."""
    key = CONTENT_TYPE_TO_FRAMEWORK.get(content_type, "hook_problem_solution_cta")
    return SCRIPT_FRAMEWORKS.get(key, SCRIPT_FRAMEWORKS["hook_problem_solution_cta"])


def get_framework_for_niche(category: str) -> dict:
    """Get the default framework for a niche category."""
    ranking = NICHE_FRAMEWORK_RANKING.get(category, ["hook_problem_solution_cta"])
    key = ranking[0]
    return SCRIPT_FRAMEWORKS.get(key, SCRIPT_FRAMEWORKS["hook_problem_solution_cta"])


def get_framework_key(content_type: str = None, category: str = None) -> str:
    """Get the framework key, preferring content_type match over category default."""
    if content_type and content_type in CONTENT_TYPE_TO_FRAMEWORK:
        return CONTENT_TYPE_TO_FRAMEWORK[content_type]
    if category and category in NICHE_FRAMEWORK_RANKING:
        return NICHE_FRAMEWORK_RANKING[category][0]
    return "hook_problem_solution_cta"
