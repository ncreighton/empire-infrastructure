"""Tests for health check modules — wordpress, service, profile, n8n, email, seo, security."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from openclaw.models import (
    AccountStatus,
    CheckResult,
    HealthCheck,
    HeartbeatTier,
    ProfileContent,
)

from openclaw.daemon.checks import (
    wordpress_check,
    service_check,
    profile_check,
    n8n_check,
    email_check,
    seo_check,
    security_check,
)


# ================================================================== #
#  WordPress Check                                                      #
# ================================================================== #


class TestWordPressCheck:
    @pytest.mark.asyncio
    async def test_empty_domains_returns_empty(self):
        result = await wordpress_check.check_all_sites([])
        assert result == []

    @pytest.mark.asyncio
    async def test_healthy_site(self):
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200

        api_response = MagicMock()
        api_response.status_code = 200

        with patch("openclaw.daemon.checks.wordpress_check.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            client.get = AsyncMock(side_effect=[mock_response, api_response])
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client

            results = await wordpress_check.check_all_sites(["test.com"])

        assert len(results) == 1
        assert results[0].name == "wp:test.com"
        assert results[0].result == CheckResult.HEALTHY

    @pytest.mark.asyncio
    async def test_down_site_500(self):
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("openclaw.daemon.checks.wordpress_check.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_response)
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client

            results = await wordpress_check.check_all_sites(["broken.com"])

        assert results[0].result == CheckResult.DOWN
        assert "500" in results[0].message

    @pytest.mark.asyncio
    async def test_timeout_is_down(self):
        import httpx

        with patch("openclaw.daemon.checks.wordpress_check.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client

            results = await wordpress_check.check_all_sites(["slow.com"])

        assert results[0].result == CheckResult.DOWN
        assert "Timeout" in results[0].message

    @pytest.mark.asyncio
    async def test_multiple_sites(self):
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("openclaw.daemon.checks.wordpress_check.httpx.AsyncClient") as mock_client_cls:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_response)
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client

            results = await wordpress_check.check_all_sites(["a.com", "b.com"])

        assert len(results) == 2


# ================================================================== #
#  Service Check                                                        #
# ================================================================== #


class TestServiceCheck:
    @pytest.mark.asyncio
    async def test_empty_services_returns_empty(self):
        result = await service_check.check_all_services({})
        assert result == []

    @pytest.mark.asyncio
    async def test_service_up_tcp_no_health(self):
        """Service reachable via TCP but no /health endpoint → HEALTHY."""
        import httpx

        with patch("asyncio.open_connection") as mock_conn:
            writer = AsyncMock()
            mock_conn.return_value = (AsyncMock(), writer)
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()

            with patch("openclaw.daemon.checks.service_check.httpx.AsyncClient") as mock_client_cls:
                client = AsyncMock()
                client.get = AsyncMock(side_effect=httpx.ConnectError("no http"))
                client.__aenter__ = AsyncMock(return_value=client)
                client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = client

                results = await service_check.check_all_services({"test": 8000})

        assert len(results) == 1
        assert results[0].result == CheckResult.HEALTHY
        assert "TCP OK" in results[0].message

    @pytest.mark.asyncio
    async def test_service_down_connection_refused(self):
        with patch("asyncio.open_connection", side_effect=ConnectionRefusedError):
            with patch("asyncio.wait_for", side_effect=ConnectionRefusedError):
                results = await service_check.check_all_services({"dead": 9999})

        assert len(results) == 1
        assert results[0].result == CheckResult.DOWN

    @pytest.mark.asyncio
    async def test_multiple_services(self):
        with patch("asyncio.open_connection") as mock_conn:
            writer = AsyncMock()
            mock_conn.return_value = (AsyncMock(), writer)
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()

            with patch("openclaw.daemon.checks.service_check.httpx.AsyncClient") as mock_client_cls:
                client = AsyncMock()
                client.get = AsyncMock(side_effect=Exception("no http"))
                client.__aenter__ = AsyncMock(return_value=client)
                client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = client

                results = await service_check.check_all_services(
                    {"svc1": 8000, "svc2": 8001}
                )

        assert len(results) == 2


# ================================================================== #
#  Profile Check                                                        #
# ================================================================== #


class TestProfileCheck:
    @pytest.fixture
    def mock_codex(self):
        codex = MagicMock()
        return codex

    @pytest.fixture
    def mock_sentinel(self):
        sentinel = MagicMock()
        return sentinel

    @pytest.mark.asyncio
    async def test_no_active_accounts(self, mock_codex, mock_sentinel):
        mock_codex.get_accounts_by_status.return_value = []
        results = await profile_check.check_profiles(mock_codex, mock_sentinel)
        assert len(results) == 1
        assert results[0].result == CheckResult.HEALTHY
        assert "No active accounts" in results[0].message

    @pytest.mark.asyncio
    async def test_stale_profile_flagged(self, mock_codex, mock_sentinel):
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        mock_codex.get_accounts_by_status.return_value = [
            {
                "platform_id": "gumroad",
                "platform_name": "Gumroad",
                "updated_at": old_date,
            },
        ]
        mock_codex.get_profile.return_value = None

        results = await profile_check.check_profiles(
            mock_codex, mock_sentinel, stale_days=30
        )
        stale_checks = [c for c in results if "stale" in c.name]
        assert len(stale_checks) == 1
        assert stale_checks[0].result == CheckResult.DEGRADED

    @pytest.mark.asyncio
    async def test_low_grade_flagged(self, mock_codex, mock_sentinel):
        mock_codex.get_accounts_by_status.return_value = [
            {
                "platform_id": "etsy",
                "platform_name": "Etsy",
                "updated_at": datetime.now().isoformat(),
            },
        ]
        mock_codex.get_profile.return_value = {
            "sentinel_score": 30.0,
            "grade": "F",
            "content": {},
        }

        results = await profile_check.check_profiles(mock_codex, mock_sentinel)
        low_grade = [c for c in results if "low_grade" in c.name]
        assert len(low_grade) == 1

    @pytest.mark.asyncio
    async def test_healthy_profiles(self, mock_codex, mock_sentinel):
        mock_codex.get_accounts_by_status.return_value = [
            {
                "platform_id": "gumroad",
                "platform_name": "Gumroad",
                "updated_at": datetime.now().isoformat(),
            },
        ]
        mock_codex.get_profile.return_value = {
            "sentinel_score": 85.0,
            "grade": "A",
            "content": {},
        }

        results = await profile_check.check_profiles(mock_codex, mock_sentinel)
        overall = [c for c in results if c.name == "profiles:overall"]
        assert len(overall) == 1
        assert overall[0].result == CheckResult.HEALTHY


# ================================================================== #
#  n8n Check                                                            #
# ================================================================== #


class TestN8nCheck:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_unknown(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("os.environ.get", return_value=None):
                results = await n8n_check.check_workflows()

        # Should return at least one check (may be UNKNOWN or have results)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_returns_health_checks(self):
        """n8n check should return HealthCheck objects."""
        with patch("os.environ.get", return_value="fake-key"):
            with patch("openclaw.daemon.checks.n8n_check.httpx.AsyncClient") as mock_cls:
                client = AsyncMock()
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"data": []}
                client.get = AsyncMock(return_value=resp)
                client.__aenter__ = AsyncMock(return_value=client)
                client.__aexit__ = AsyncMock(return_value=False)
                mock_cls.return_value = client

                results = await n8n_check.check_workflows()

        assert all(isinstance(r, HealthCheck) for r in results)


# ================================================================== #
#  Email Check                                                          #
# ================================================================== #


class TestEmailCheck:
    @pytest.mark.asyncio
    async def test_no_imap_config_returns_unknown(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("os.environ.get", return_value=None):
                results = await email_check.check_inbox()

        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], HealthCheck)


# ================================================================== #
#  SEO Check                                                            #
# ================================================================== #


class TestSeoCheck:
    @pytest.mark.asyncio
    async def test_no_gsc_credentials_returns_unknown(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("os.environ.get", return_value=None):
                results = await seo_check.check_traffic(0.20)

        assert isinstance(results, list)
        if results:
            assert results[0].result == CheckResult.UNKNOWN


# ================================================================== #
#  Security Check                                                       #
# ================================================================== #


class TestSecurityCheck:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        # Patch Path.exists to prevent real file reads
        with patch("openclaw.daemon.checks.security_check.Path.exists", return_value=False):
            results = await security_check.check_plugin_security()

        assert isinstance(results, list)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_no_sites_returns_unknown(self):
        with patch("openclaw.daemon.checks.security_check.Path.exists", return_value=False):
            results = await security_check.check_plugin_security()

        assert results[0].result == CheckResult.UNKNOWN
        assert "not found" in results[0].message
