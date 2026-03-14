"""Tests for StepRouter — intelligent Haiku/Sonnet model routing per browser step."""

import gc
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.browser.step_router import StepRouter, HAIKU, SONNET
from openclaw.forge.platform_codex import PlatformCodex
from openclaw.models import SignupStep, StepType


def _safe_unlink(path: str) -> None:
    """Unlink a temp DB file, tolerating Windows file locks from SQLite WAL."""
    gc.collect()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except (PermissionError, FileNotFoundError):
            pass


def _make_step(step_type: StepType, target: str = "", value: str = "") -> SignupStep:
    """Helper to create a SignupStep for testing."""
    return SignupStep(
        step_number=1,
        step_type=step_type,
        description=f"Test {step_type.value}",
        target=target,
        value=value,
    )


def test_default_routing():
    """Verify each StepType maps to the expected default model."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = PlatformCodex(db_path=db_path)
        router = StepRouter(codex)

        # Haiku steps
        haiku_types = [
            StepType.NAVIGATE,
            StepType.DISMISS_MODAL,
            StepType.CLICK,
            StepType.ACCEPT_TERMS,
            StepType.SELECT_DROPDOWN,
            StepType.FILL_TEXTAREA,
            StepType.UPLOAD_FILE,
        ]
        for st in haiku_types:
            step = _make_step(st, target="some_field")
            model = router.get_model(step, "test_platform")
            assert model == HAIKU, f"Expected HAIKU for {st.value}, got {model}"

        # Sonnet steps
        sonnet_types = [
            StepType.SUBMIT_FORM,
            StepType.SOLVE_CAPTCHA,
            StepType.OAUTH_LOGIN,
        ]
        for st in sonnet_types:
            step = _make_step(st, target="some_field")
            model = router.get_model(step, "test_platform")
            assert model == SONNET, f"Expected SONNET for {st.value}, got {model}"

        # No-LLM steps
        no_llm_types = [
            StepType.WAIT_FOR_NAVIGATION,
            StepType.SCREENSHOT,
            StepType.VERIFY_EMAIL,
        ]
        for st in no_llm_types:
            step = _make_step(st)
            model = router.get_model(step, "test_platform")
            assert model is None, f"Expected None for {st.value}, got {model}"

        print("  Default routing: PASSED")
    finally:
        _safe_unlink(db_path)


def test_email_field_uses_sonnet():
    """Email FILL_FIELD should always use Sonnet (may need to reveal form)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = PlatformCodex(db_path=db_path)
        router = StepRouter(codex)

        step = _make_step(StepType.FILL_FIELD, target="email", value="test@example.com")
        model = router.get_model(step, "test_platform")
        assert model == SONNET, f"Email field should use SONNET, got {model}"

        # Also test mixed case
        step2 = _make_step(StepType.FILL_FIELD, target="Email Address", value="x@y.com")
        model2 = router.get_model(step2, "test_platform")
        assert model2 == SONNET, f"Email Address field should use SONNET, got {model2}"

        print("  Email field routing: PASSED")
    finally:
        _safe_unlink(db_path)


def test_password_field_uses_haiku():
    """Password FILL_FIELD should use Haiku (JS injection is primary, agent is fallback)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = PlatformCodex(db_path=db_path)
        router = StepRouter(codex)

        step = _make_step(StepType.FILL_FIELD, target="password", value="secret123")
        model = router.get_model(step, "test_platform")
        assert model == HAIKU, f"Password field should use HAIKU, got {model}"

        print("  Password field routing: PASSED")
    finally:
        _safe_unlink(db_path)


def test_non_email_fill_uses_haiku():
    """Non-email, non-password FILL_FIELD should use Haiku."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = PlatformCodex(db_path=db_path)
        router = StepRouter(codex)

        step = _make_step(StepType.FILL_FIELD, target="username", value="testuser")
        model = router.get_model(step, "test_platform")
        assert model == HAIKU, f"Username field should use HAIKU, got {model}"

        step2 = _make_step(StepType.FILL_FIELD, target="display_name", value="Test User")
        model2 = router.get_model(step2, "test_platform")
        assert model2 == HAIKU, f"Display name field should use HAIKU, got {model2}"

        print("  Non-email fill routing: PASSED")
    finally:
        _safe_unlink(db_path)


def test_promotion_on_failure():
    """Record failure should promote a Haiku step to Sonnet for next run."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = PlatformCodex(db_path=db_path)
        router = StepRouter(codex)

        platform = "test_platform"
        step = _make_step(StepType.NAVIGATE, target="https://example.com")

        # Initially should be Haiku
        model = router.get_model(step, platform)
        assert model == HAIKU, f"NAVIGATE should start as HAIKU, got {model}"

        # Record failure with Haiku
        router.record_failure(platform, StepType.NAVIGATE, HAIKU)

        # Now should be promoted to Sonnet
        model2 = router.get_model(step, platform)
        assert model2 == SONNET, f"NAVIGATE should be promoted to SONNET, got {model2}"

        # Different platform should still be Haiku
        model3 = router.get_model(step, "other_platform")
        assert model3 == HAIKU, f"Other platform should still be HAIKU, got {model3}"

        print("  Promotion on failure: PASSED")
    finally:
        _safe_unlink(db_path)


def test_sonnet_failure_no_promotion():
    """Failing with Sonnet should NOT create a promotion (already at highest tier)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = PlatformCodex(db_path=db_path)
        router = StepRouter(codex)

        router.record_failure("test", StepType.SUBMIT_FORM, SONNET)
        promotions = codex.get_step_promotions("test")
        assert len(promotions) == 0, "Sonnet failure should not create promotion"

        print("  Sonnet failure no-promotion: PASSED")
    finally:
        _safe_unlink(db_path)


def test_retry_model_upgrade():
    """get_model_for_retry should promote Haiku to Sonnet."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = PlatformCodex(db_path=db_path)
        router = StepRouter(codex)

        step = _make_step(StepType.CLICK, target="submit_button")

        # Retry from Haiku → Sonnet
        retry_model = router.get_model_for_retry(step, "test_platform", HAIKU)
        assert retry_model == SONNET, f"Retry from HAIKU should be SONNET, got {retry_model}"

        # Retry from Sonnet → stays Sonnet
        retry_model2 = router.get_model_for_retry(step, "test_platform", SONNET)
        assert retry_model2 == SONNET, f"Retry from SONNET should stay SONNET, got {retry_model2}"

        print("  Retry model upgrade: PASSED")
    finally:
        _safe_unlink(db_path)


def test_cost_tracking():
    """record_step and get_cost_report should track costs correctly."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = PlatformCodex(db_path=db_path)
        router = StepRouter(codex)

        # Simulate a signup with mixed models
        steps = [
            (StepType.NAVIGATE, HAIKU, True, 500, 50),
            (StepType.DISMISS_MODAL, HAIKU, True, 300, 30),
            (StepType.FILL_FIELD, HAIKU, True, 600, 80),
            (StepType.FILL_FIELD, SONNET, True, 800, 100),  # email field
            (StepType.ACCEPT_TERMS, HAIKU, True, 200, 20),
            (StepType.SUBMIT_FORM, SONNET, True, 1500, 300),
        ]

        for st, model, success, in_tok, out_tok in steps:
            router.record_step(
                platform_id="test_platform",
                step_type=st,
                model_used=model,
                success=success,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )

        report = router.get_cost_report(days=1)
        assert report["total_steps"] == 6, f"Expected 6 steps, got {report['total_steps']}"
        assert report["successful_steps"] == 6
        assert report["total_cost_usd"] > 0
        assert report["counterfactual_cost_usd"] > 0
        assert report["savings_usd"] > 0, "Should save money with Haiku mix"
        assert report["savings_pct"] > 0
        assert HAIKU in report["by_model"]
        assert SONNET in report["by_model"]
        assert report["by_model"][HAIKU]["steps"] == 4
        assert report["by_model"][SONNET]["steps"] == 2

        print(f"  Cost tracking: ${report['total_cost_usd']:.6f} actual, "
              f"${report['counterfactual_cost_usd']:.6f} counterfactual, "
              f"${report['savings_usd']:.6f} saved ({report['savings_pct']:.1f}%)")
        print("  Cost tracking: PASSED")
    finally:
        _safe_unlink(db_path)


def test_promotion_expiry():
    """Promotions older than 7 days should be expired."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = PlatformCodex(db_path=db_path)
        router = StepRouter(codex)

        # Insert a promotion with an old timestamp
        from datetime import datetime, timedelta
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        with codex._connect() as conn:
            conn.execute(
                "INSERT INTO step_model_promotions (platform_id, step_type, promoted_at, reason) "
                "VALUES (?, ?, ?, ?)",
                ("test_platform", "navigate", old_time, "old test"),
            )

        # Insert a fresh promotion
        codex.upsert_step_promotion("test_platform", "click", "fresh test")

        # Expire old
        expired = router.expire_promotions(days=7)
        assert expired == 1, f"Expected 1 expired, got {expired}"

        # Fresh one should still exist
        promotions = codex.get_step_promotions("test_platform")
        assert "click" in promotions, "Fresh promotion should survive"
        assert "navigate" not in promotions, "Old promotion should be expired"

        print("  Promotion expiry: PASSED")
    finally:
        _safe_unlink(db_path)


def test_routing_summary():
    """get_routing_summary should return all step types with their tier."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        codex = PlatformCodex(db_path=db_path)
        router = StepRouter(codex)

        summary = router.get_routing_summary()
        assert "navigate" in summary
        assert summary["navigate"] == "haiku"
        assert summary["submit_form"] == "sonnet"
        assert summary["screenshot"] == "none"
        assert summary["verify_email"] == "none"

        print("  Routing summary: PASSED")
    finally:
        _safe_unlink(db_path)


def test_cost_estimation():
    """_estimate_cost should calculate correct USD amounts."""
    # Haiku: 1000 input tokens + 100 output tokens
    # = 1000 * 0.80 / 1M + 100 * 4.00 / 1M
    # = 0.0008 + 0.0004 = 0.0012
    cost = StepRouter._estimate_cost(HAIKU, 1000, 100)
    assert abs(cost - 0.0012) < 0.00001, f"Expected 0.0012, got {cost}"

    # Sonnet: 1000 input tokens + 100 output tokens
    # = 1000 * 3.00 / 1M + 100 * 15.00 / 1M
    # = 0.003 + 0.0015 = 0.0045
    cost2 = StepRouter._estimate_cost(SONNET, 1000, 100)
    assert abs(cost2 - 0.0045) < 0.00001, f"Expected 0.0045, got {cost2}"

    # Haiku is 3.75x cheaper
    ratio = cost2 / cost
    assert abs(ratio - 3.75) < 0.01, f"Expected 3.75x ratio, got {ratio}"

    print(f"  Cost estimation: Haiku={cost:.6f}, Sonnet={cost2:.6f}, ratio={ratio:.2f}x")
    print("  Cost estimation: PASSED")


if __name__ == "__main__":
    print("StepRouter Tests")
    print("=" * 60)
    test_default_routing()
    test_email_field_uses_sonnet()
    test_password_field_uses_haiku()
    test_non_email_fill_uses_haiku()
    test_promotion_on_failure()
    test_sonnet_failure_no_promotion()
    test_retry_model_upgrade()
    test_cost_tracking()
    test_promotion_expiry()
    test_routing_summary()
    test_cost_estimation()
    print("=" * 60)
    print("ALL TESTS PASSED")
