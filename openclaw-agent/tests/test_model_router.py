"""Tests for ModelRouter — intelligent model routing and cost optimization."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.vibecoder.forge.model_router import (
    ModelRouter,
    ModelTier,
    TaskCategory,
)


def test_routing():
    """Test that tasks route to the correct model tier."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        router = ModelRouter(db_path=db_path, monthly_budget=100.0)

        # Haiku tasks
        haiku_tasks = [
            "Classify this email as spam or not spam",
            "Extract the names from this document",
            "Convert this JSON data to CSV format",
            "Generate a commit message for this diff",
            "Summarize this paragraph in one sentence",
        ]
        for task in haiku_tasks:
            d = router.route(task)
            assert d.model_spec.tier == ModelTier.HAIKU, (
                f"Expected haiku for '{task}', got {d.model_spec.tier.value}"
            )
            print(f"  HAIKU: {task[:50]}")

        # Sonnet tasks
        sonnet_tasks = [
            "Review this Python function for security issues and code quality",
            "Write a compelling blog post about smart home automation",
            "Debug this traceback and find the root cause of the error",
            "Edit the file to fix the import statement",
            "Refactor this code across multiple files in the project",
        ]
        for task in sonnet_tasks:
            d = router.route(task)
            assert d.model_spec.tier == ModelTier.SONNET, (
                f"Expected sonnet for '{task}', got {d.model_spec.tier.value}"
            )
            print(f"  SONNET: {task[:50]}")

        # Opus tasks
        opus_tasks = [
            "Design the system architecture for a distributed database with precise ACID compliance",
        ]
        for task in opus_tasks:
            d = router.route(task)
            assert d.model_spec.tier == ModelTier.OPUS, (
                f"Expected opus for '{task}', got {d.model_spec.tier.value}"
            )
            print(f"  OPUS: {task[:50]}")

        print("  Routing tests: PASSED")

    finally:
        os.unlink(db_path)


def test_cost_savings():
    """Test that cheaper tiers actually save money vs Opus."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        router = ModelRouter(db_path=db_path)

        d_haiku = router.route("Classify intent")
        d_sonnet = router.route("Review code quality of this function")
        d_opus = router.route("Design precise production system architecture")

        assert d_haiku.savings_vs_opus > 0, "Haiku should save vs Opus"
        assert d_sonnet.savings_vs_opus > 0, "Sonnet should save vs Opus"
        assert d_opus.savings_vs_opus == 0, "Opus saves nothing vs itself"

        print(f"  Haiku savings: ${d_haiku.savings_vs_opus:.6f}")
        print(f"  Sonnet savings: ${d_sonnet.savings_vs_opus:.6f}")
        print("  Cost savings tests: PASSED")

    finally:
        os.unlink(db_path)


def test_outcome_recording():
    """Test that outcomes are recorded and affect future routing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        router = ModelRouter(db_path=db_path, monthly_budget=100.0)

        # Simulate 10 API calls
        test_cases = [
            ("Classify email intent", 150, 12, 0.98),
            ("Extract names from document", 500, 80, 0.95),
            ("Format data as JSON", 200, 150, 0.99),
            ("Generate commit message for diff", 300, 25, 0.92),
            ("Review Python function security", 2000, 400, 0.88),
            ("Write blog post about AI tools", 1500, 2000, 0.91),
            ("Debug traceback error", 1000, 500, 0.85),
            ("Summarize meeting notes", 800, 200, 0.94),
            ("Edit file to fix import", 600, 300, 0.97),
            ("Classify sentiment of review", 100, 8, 0.99),
        ]

        for task, in_tok, out_tok, quality in test_cases:
            decision = router.route(task)
            router.record_outcome(
                decision=decision,
                actual_input_tokens=in_tok,
                actual_output_tokens=out_tok,
                quality_score=quality,
            )

        # Check spend report
        report = router.get_spend_report(1)
        assert report.total_requests == 10
        assert report.total_cost > 0
        assert report.avg_quality_score > 0.8
        print(f"  Total cost: ${report.total_cost:.6f}")
        print(f"  Savings vs Opus: ${report.savings_vs_opus:.6f}")
        print(f"  Avg quality: {report.avg_quality_score:.2f}")
        print(f"  By tier: {report.requests_by_tier}")

        # Check optimization tips
        tips = router.get_optimization_tips()
        assert len(tips) > 0
        print(f"  Optimization tips: {len(tips)}")

        # Check budget
        assert router._get_budget_pressure() > 0
        print(f"  Budget pressure: {router._get_budget_pressure():.4f}")

        print("  Outcome recording tests: PASSED")

    finally:
        os.unlink(db_path)


def test_prompt_compression():
    """Test prompt compression reduces token count."""
    router = ModelRouter.__new__(ModelRouter)  # No DB needed

    long_prompt = """
    # System Prompt

    You are a helpful assistant.

    ═══════════════════════════════════════════

    ## Instructions

    Follow these rules carefully.

    ---

    ## Examples

    Example 1: Do this thing.
    Example 2: Do that thing.
    Example 3: Do another thing.
    Example 4: One more thing.
    Example 5: Final thing.

    ---

    ## More Rules

    Be concise and helpful.
    """ * 5  # Make it big

    compressed = ModelRouter.compress_prompt(long_prompt, max_tokens_approx=200)
    assert len(compressed) < len(long_prompt)
    print(f"  Original: {len(long_prompt)} chars")
    print(f"  Compressed: {len(compressed)} chars")
    print(f"  Reduction: {(1 - len(compressed)/len(long_prompt)):.0%}")
    print("  Prompt compression tests: PASSED")


def test_max_tokens():
    """Test that max_tokens are set appropriately per category."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        router = ModelRouter(db_path=db_path)

        # Classification should have low max_tokens
        d = router.route("Classify this as spam")
        assert d.max_tokens <= 200, f"Classification max_tokens too high: {d.max_tokens}"

        # Code generation should have higher max_tokens
        d2 = router.route("Edit the file to add error handling")
        assert d2.max_tokens >= 2000, f"Code edit max_tokens too low: {d2.max_tokens}"

        print(f"  Classification max_tokens: {d.max_tokens}")
        print(f"  Code edit max_tokens: {d2.max_tokens}")
        print("  Max tokens tests: PASSED")

    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    print("ModelRouter Tests")
    print("=" * 60)
    test_routing()
    test_cost_savings()
    test_outcome_recording()
    test_prompt_compression()
    test_max_tokens()
    print("=" * 60)
    print("ALL TESTS PASSED")
