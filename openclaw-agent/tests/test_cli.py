"""Tests for cli.py — CLI argument parsing and command dispatch."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from openclaw.models import (
    AccountStatus,
    DashboardStats,
    OraclePriority,
    OracleRecommendation,
    PlatformCategory,
    ProfileContent,
    QualityGrade,
    ScoutResult,
    SentinelScore,
    SignupComplexity,
    CaptchaType,
)


@pytest.fixture
def mock_engine():
    """Create a fully mocked OpenClawEngine."""
    engine = MagicMock()

    # generate_profile
    engine.generate_profile.return_value = ProfileContent(
        platform_id="gumroad",
        username="testuser",
        display_name="Test User",
        email="test@example.com",
        bio="A test bio for the profile.",
        tagline="Test tagline here",
        description="Test description text.",
        website_url="https://example.com",
        social_links={"github": "https://github.com/testuser"},
        seo_keywords=["ai", "automation"],
    )

    # score_profile
    engine.score_profile.return_value = SentinelScore(
        platform_id="gumroad",
        total_score=82.0,
        grade=QualityGrade.B,
        completeness=18.0,
        seo_quality=16.0,
        brand_consistency=12.0,
        link_presence=14.0,
        bio_quality=12.0,
        avatar_quality=10.0,
        feedback=["Good bio length", "Add more SEO keywords"],
        enhancements=["Include banner image"],
    )

    # analyze_platform
    engine.analyze_platform.return_value = ScoutResult(
        platform_id="gumroad",
        complexity=SignupComplexity.SIMPLE,
        estimated_minutes=5,
        captcha_type=CaptchaType.NONE,
        required_fields=["email", "password"],
        optional_fields=["bio", "avatar"],
        readiness_checklist=[
            {"item": "Email", "ready": True, "note": "Configured"},
            {"item": "Password", "ready": True, "note": "Will be provided"},
        ],
        risks=[],
        tips=["Use a strong password"],
        completeness_score=85.0,
    )

    # prioritize
    engine.prioritize.return_value = [
        OracleRecommendation(
            platform_id="gumroad",
            platform_name="Gumroad",
            category=PlatformCategory.DIGITAL_PRODUCT,
            priority=OraclePriority.HIGH,
            score=80.0,
            monetization_score=35.0,
            audience_score=20.0,
            seo_score=15.0,
            effort_score=5.0,
            reasoning="High monetization potential.",
        ),
    ]

    # get_dashboard
    engine.get_dashboard.return_value = DashboardStats(
        total_platforms=46,
        active_accounts=5,
        pending_signups=2,
        failed_signups=1,
    )

    # get_platform_status
    engine.get_platform_status.return_value = {
        "platform": {
            "id": "gumroad",
            "name": "Gumroad",
            "category": "digital_product",
            "complexity": "simple",
        },
        "account": None,
        "profile": None,
        "signup_log": [],
    }

    # codex mock for status/health commands
    engine.codex = MagicMock()
    engine.codex.get_all_accounts.return_value = []
    engine.codex.get_all_profiles.return_value = []
    engine.codex.db_path = "/tmp/test_openclaw.db"
    engine.codex.get_stats.return_value = {
        "total_accounts": 0,
        "active_accounts": 0,
        "avg_sentinel_score": 0,
    }

    # sentinel mock
    engine.sentinel = MagicMock()

    return engine


class TestHealthCommand:
    def test_health_runs_without_error(self, mock_engine, capsys):
        """Health command should complete without exceptions."""
        with patch("cli.OpenClawEngine", return_value=mock_engine):
            with patch("sys.argv", ["cli.py", "health"]):
                # Import and run
                import cli
                cli.cmd_health(mock_engine, MagicMock(command="health"))

        captured = capsys.readouterr()
        assert "Health Check" in captured.out


class TestPlatformsCommand:
    def test_platforms_lists_output(self, mock_engine, capsys):
        """Platforms command should list platforms."""
        args = MagicMock()
        args.category = None

        import cli
        cli.cmd_platforms(mock_engine, args)

        captured = capsys.readouterr()
        assert "Supported Platforms" in captured.out
        # Should contain at least some platform data
        assert "platforms" in captured.out.lower() or "Total" in captured.out

    def test_platforms_with_category(self, mock_engine, capsys):
        """Platforms command with category filter."""
        args = MagicMock()
        args.category = "digital_product"

        import cli
        cli.cmd_platforms(mock_engine, args)

        captured = capsys.readouterr()
        assert "digital_product" in captured.out.lower() or "Filtering" in captured.out


class TestGenerateCommand:
    def test_generate_produces_output(self, mock_engine, capsys):
        """Generate command should show profile content."""
        args = MagicMock()
        args.platform_id = "gumroad"

        import cli
        cli.cmd_generate(mock_engine, args)

        captured = capsys.readouterr()
        assert "testuser" in captured.out
        assert "Test User" in captured.out
        assert "example.com" in captured.out


class TestAnalyzeCommand:
    def test_analyze_produces_output(self, mock_engine, capsys):
        """Analyze command should show scout analysis."""
        args = MagicMock()
        args.platform_id = "gumroad"

        import cli
        cli.cmd_analyze(mock_engine, args)

        captured = capsys.readouterr()
        assert "Scout Analysis" in captured.out
        assert "simple" in captured.out.lower()
        assert "email" in captured.out.lower()


class TestScoreCommand:
    def test_score_produces_output(self, mock_engine, capsys):
        """Score command should show sentinel score."""
        args = MagicMock()
        args.platform_id = "gumroad"

        import cli
        cli.cmd_score(mock_engine, args)

        captured = capsys.readouterr()
        assert "Profile Score" in captured.out
        assert "82" in captured.out

    def test_score_missing_profile(self, mock_engine, capsys):
        """Score command with no stored profile should warn."""
        mock_engine.score_profile.return_value = None
        args = MagicMock()
        args.platform_id = "gumroad"

        import cli
        with pytest.raises(SystemExit):
            cli.cmd_score(mock_engine, args)


class TestPrioritizeCommand:
    def test_prioritize_runs(self, mock_engine, capsys):
        """Prioritize command should list recommendations."""
        args = MagicMock()

        import cli
        cli.cmd_prioritize(mock_engine, args)

        captured = capsys.readouterr()
        assert "Recommendations" in captured.out
        assert "Gumroad" in captured.out


class TestStatusCommand:
    def test_status_all(self, mock_engine, capsys):
        """Status command without platform_id shows all accounts."""
        args = MagicMock()
        args.platform_id = None
        args.format = "table"

        import cli
        cli.cmd_status(mock_engine, args)

        captured = capsys.readouterr()
        assert "All Accounts" in captured.out

    def test_status_single_platform(self, mock_engine, capsys):
        """Status command with platform_id shows details."""
        args = MagicMock()
        args.platform_id = "gumroad"
        args.format = "table"

        import cli
        cli.cmd_status(mock_engine, args)

        captured = capsys.readouterr()
        assert "Status" in captured.out
        assert "Gumroad" in captured.out

    def test_status_json_format(self, mock_engine, capsys):
        """Status command with json format outputs JSON."""
        args = MagicMock()
        args.platform_id = "gumroad"
        args.format = "json"

        import cli
        cli.cmd_status(mock_engine, args)

        captured = capsys.readouterr()
        assert '"platform"' in captured.out or '"id"' in captured.out


class TestMainEntryPoint:
    def test_no_command_shows_help(self, capsys):
        """Running without a command should show help."""
        with patch("sys.argv", ["cli.py"]):
            import cli
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 0

    def test_unknown_command_shows_help(self, capsys):
        """Unknown command should show help or error."""
        with patch("sys.argv", ["cli.py", "nonexistent_command"]):
            import cli
            with pytest.raises(SystemExit):
                cli.main()
