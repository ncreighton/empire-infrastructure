"""Test email_list_builder — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.email_list_builder import (
        EmailListBuilder,
        EmailList,
        Subscriber,
        CaptureForm,
        NurtureSequence,
        EmailCampaign,
        Segment,
        ListType,
        SubscriberStatus,
        SegmentCriteria,
        FormType,
        SequenceStatus,
        EmailStatus,
        get_list_builder,
        _validate_email,
        _validate_site_id,
        _load_json,
        _save_json,
        _make_id,
        ALL_SITE_IDS,
        SITE_NICHES,
        SITE_VOICES,
        MAX_SUBSCRIBERS_PER_LIST,
        ENGAGEMENT_OPEN_WEIGHT,
        ENGAGEMENT_CLICK_WEIGHT,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="email_list_builder module not available"
)


# ===================================================================
# Enum tests
# ===================================================================

class TestEnums:
    """Verify enum values."""

    def test_list_types(self):
        assert ListType.NEWSLETTER.value == "newsletter"
        assert ListType.LEAD_MAGNET.value == "lead_magnet"
        assert ListType.COURSE.value == "course"
        assert ListType.GENERAL.value == "general"

    def test_subscriber_statuses(self):
        assert SubscriberStatus.ACTIVE.value == "active"
        assert SubscriberStatus.UNSUBSCRIBED.value == "unsubscribed"
        assert SubscriberStatus.BOUNCED.value == "bounced"
        assert SubscriberStatus.PENDING.value == "pending"

    def test_segment_criteria(self):
        assert SegmentCriteria.SITE.value == "site"
        assert SegmentCriteria.ENGAGEMENT.value == "engagement"
        assert SegmentCriteria.TAG.value == "tag"

    def test_form_types(self):
        assert FormType.POPUP.value == "popup"
        assert FormType.INLINE.value == "inline"
        assert FormType.EXIT_INTENT.value == "exit_intent"
        assert FormType.SIDEBAR.value == "sidebar"

    def test_sequence_statuses(self):
        assert SequenceStatus.DRAFT.value == "draft"
        assert SequenceStatus.ACTIVE.value == "active"
        assert SequenceStatus.PAUSED.value == "paused"

    def test_email_statuses(self):
        assert EmailStatus.SCHEDULED.value == "scheduled"
        assert EmailStatus.SENT.value == "sent"
        assert EmailStatus.BOUNCED.value == "bounced"


# ===================================================================
# Helper functions
# ===================================================================

class TestHelpers:
    """Test module-level helper functions."""

    def test_validate_email_valid(self):
        assert _validate_email("alice@example.com") is True
        assert _validate_email("Bob.Smith+tag@domain.co.uk") is True

    def test_validate_email_invalid(self):
        assert _validate_email("not-an-email") is False
        assert _validate_email("@domain.com") is False
        assert _validate_email("user@") is False
        assert _validate_email("") is False

    def test_validate_site_id_valid(self):
        _validate_site_id("witchcraft")  # should not raise
        _validate_site_id("smarthome")

    def test_validate_site_id_invalid(self):
        with pytest.raises(ValueError):
            _validate_site_id("nonexistent_site")

    def test_make_id(self):
        uid = _make_id()
        assert len(uid) > 0
        assert isinstance(uid, str)

    def test_all_site_ids(self):
        assert len(ALL_SITE_IDS) == 16
        assert "witchcraft" in ALL_SITE_IDS

    def test_site_niches(self):
        assert "witchcraft" in SITE_NICHES
        assert "smarthome" in SITE_NICHES

    def test_site_voices(self):
        assert "witchcraft" in SITE_VOICES
        assert "Mystical" in SITE_VOICES["witchcraft"]


# ===================================================================
# Dataclass tests
# ===================================================================

class TestEmailList:
    """Test EmailList dataclass."""

    def test_default_list(self):
        lst = EmailList()
        assert lst.list_id
        assert lst.subscriber_count == 0
        assert lst.double_optin is True

    def test_with_fields(self):
        lst = EmailList(
            name="Moon Magic Newsletter",
            site_id="witchcraft",
            list_type=ListType.NEWSLETTER.value,
        )
        assert lst.name == "Moon Magic Newsletter"
        assert lst.list_type == "newsletter"


class TestSubscriber:
    """Test Subscriber dataclass."""

    def test_default_subscriber(self):
        sub = Subscriber()
        assert sub.subscriber_id
        assert sub.status == "pending"
        assert sub.engagement_score == 0.0

    def test_with_fields(self):
        sub = Subscriber(
            email="alice@example.com",
            name="Alice",
            status=SubscriberStatus.ACTIVE.value,
            engagement_score=85.0,
        )
        assert sub.email == "alice@example.com"
        assert sub.engagement_score == 85.0


class TestCaptureForm:
    """Test CaptureForm dataclass."""

    def test_default_form(self):
        form = CaptureForm()
        assert form.form_id
        assert form.form_type == "inline"
        assert form.active is True

    def test_conversion_rate(self):
        form = CaptureForm(
            impressions=1000,
            conversions=50,
            conversion_rate=0.05,
        )
        assert form.conversion_rate == 0.05


class TestNurtureSequence:
    """Test NurtureSequence dataclass."""

    def test_default_sequence(self):
        seq = NurtureSequence()
        assert seq.sequence_id
        assert seq.status == "draft"
        assert seq.trigger == "on_subscribe"


class TestSegment:
    """Test Segment dataclass."""

    def test_default_segment(self):
        seg = Segment()
        assert seg.segment_id
        assert seg.criteria == "tag"


# ===================================================================
# EmailListBuilder — list management
# ===================================================================

class TestEmailListBuilderLists:
    """Test list CRUD operations."""

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_create_list(self, mock_load, mock_save):
        builder = EmailListBuilder()
        lst = builder.create_list(
            site_id="witchcraft",
            name="Moon Magic Newsletter",
            list_type=ListType.NEWSLETTER,
        )
        assert isinstance(lst, EmailList)
        assert lst.name == "Moon Magic Newsletter"
        assert lst.site_id == "witchcraft"

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_create_list_invalid_site(self, mock_load, mock_save):
        builder = EmailListBuilder()
        with pytest.raises(ValueError, match="Unknown site_id"):
            builder.create_list(site_id="invalid_site", name="Test")

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_create_list_empty_name(self, mock_load, mock_save):
        builder = EmailListBuilder()
        with pytest.raises(ValueError, match="empty"):
            builder.create_list(site_id="witchcraft", name="")

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_create_duplicate_list(self, mock_load, mock_save):
        builder = EmailListBuilder()
        builder.create_list(site_id="witchcraft", name="Unique List")
        with pytest.raises(ValueError, match="already exists"):
            builder.create_list(site_id="witchcraft", name="Unique List")

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_get_list(self, mock_load, mock_save):
        builder = EmailListBuilder()
        lst = builder.create_list(site_id="witchcraft", name="Test List")
        fetched = builder.get_list(lst.list_id)
        assert fetched is not None
        assert fetched.name == "Test List"

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_get_nonexistent_list(self, mock_load, mock_save):
        builder = EmailListBuilder()
        assert builder.get_list("nonexistent") is None


# ===================================================================
# Subscriber add/remove
# ===================================================================

class TestEmailListBuilderSubscribers:
    """Test subscriber management."""

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_add_subscriber(self, mock_load, mock_save):
        builder = EmailListBuilder()
        lst = builder.create_list(site_id="witchcraft", name="NL")
        sub = builder.add_subscriber(
            email="alice@example.com",
            list_id=lst.list_id,
            site_id="witchcraft",
            name="Alice",
        )
        assert isinstance(sub, Subscriber)
        assert sub.email == "alice@example.com"
        assert lst.list_id in sub.lists

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_remove_subscriber(self, mock_load, mock_save):
        builder = EmailListBuilder()
        lst = builder.create_list(site_id="witchcraft", name="NL")
        sub = builder.add_subscriber(
            email="bob@example.com",
            list_id=lst.list_id,
            site_id="witchcraft",
        )
        result = builder.remove_subscriber(sub.subscriber_id, lst.list_id)
        assert result is True


# ===================================================================
# Segment creation and evaluation
# ===================================================================

class TestEmailListBuilderSegments:
    """Test segment management."""

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_create_segment(self, mock_load, mock_save):
        builder = EmailListBuilder()
        seg = builder.create_segment(
            name="High Engagement",
            criteria=SegmentCriteria.ENGAGEMENT,
            value="high",
        )
        assert isinstance(seg, Segment)
        assert seg.name == "High Engagement"
        assert seg.criteria == "engagement"


# ===================================================================
# Email sequence generation (mocked AI)
# ===================================================================

class TestSequenceGeneration:
    """Test email sequence generation with mocked AI."""

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_create_sequence(self, mock_load, mock_save):
        builder = EmailListBuilder()
        lst = builder.create_list(site_id="witchcraft", name="NL")
        seq = builder.create_sequence(
            name="Welcome Series",
            list_id=lst.list_id,
        )
        assert isinstance(seq, NurtureSequence)
        assert seq.name == "Welcome Series"
        assert seq.list_id == lst.list_id

    @pytest.mark.asyncio
    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    async def test_generate_sequence_emails(self, mock_load, mock_save):
        builder = EmailListBuilder()
        lst = builder.create_list(site_id="witchcraft", name="NL")
        seq = builder.create_sequence(name="Welcome", list_id=lst.list_id)

        with patch.object(builder, "generate_sequence_emails", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = {
                "sequence_id": seq.sequence_id,
                "emails_generated": 5,
                "emails": [
                    {"subject": f"Day {i} Welcome", "body": f"Content {i}"}
                    for i in range(1, 6)
                ],
            }
            result = await builder.generate_sequence_emails(
                sequence_id=seq.sequence_id,
                site_id="witchcraft",
                email_count=5,
            )
            assert result["emails_generated"] == 5


# ===================================================================
# Form creation and tracking
# ===================================================================

class TestFormTracking:
    """Test capture form management."""

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_create_form(self, mock_load, mock_save):
        builder = EmailListBuilder()
        lst = builder.create_list(site_id="witchcraft", name="NL")
        form = builder.create_form(
            site_id="witchcraft",
            form_type=FormType.POPUP,
            title="Join Our Coven",
            target_list_id=lst.list_id,
        )
        assert isinstance(form, CaptureForm)
        assert form.form_type == "popup"
        assert form.title == "Join Our Coven"
        assert form.target_list_id == lst.list_id


# ===================================================================
# Lead magnet management
# ===================================================================

class TestLeadMagnetManagement:
    """Test lead magnet related functionality."""

    @pytest.mark.asyncio
    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    async def test_generate_lead_magnet_ideas(self, mock_load, mock_save):
        builder = EmailListBuilder()
        with patch.object(builder, "generate_lead_magnet_ideas", new_callable=AsyncMock) as mock_ideas:
            mock_ideas.return_value = {
                "site_id": "witchcraft",
                "ideas": [
                    {"title": "Moon Ritual Checklist", "type": "pdf_checklist"},
                    {"title": "Crystal Healing Mini Course", "type": "email_course"},
                ],
            }
            result = await builder.generate_lead_magnet_ideas("witchcraft")
            assert len(result["ideas"]) == 2


# ===================================================================
# Engagement scoring
# ===================================================================

class TestEngagementScoring:
    """Test subscriber engagement score logic."""

    def test_engagement_weights(self):
        assert ENGAGEMENT_OPEN_WEIGHT == 2.0
        assert ENGAGEMENT_CLICK_WEIGHT == 5.0

    def test_max_subscribers_constant(self):
        assert MAX_SUBSCRIBERS_PER_LIST == 50000


# ===================================================================
# List health report
# ===================================================================

class TestListHealth:
    """Test list health reporting."""

    @patch("src.email_list_builder._save_json")
    @patch("src.email_list_builder._load_json", return_value={})
    def test_multiple_lists(self, mock_load, mock_save):
        builder = EmailListBuilder()
        lst1 = builder.create_list(site_id="witchcraft", name="NL1")
        lst2 = builder.create_list(site_id="witchcraft", name="NL2")

        # Add subscribers to both
        builder.add_subscriber(
            email="a@example.com", list_id=lst1.list_id, site_id="witchcraft"
        )
        builder.add_subscriber(
            email="b@example.com", list_id=lst1.list_id, site_id="witchcraft"
        )
        builder.add_subscriber(
            email="c@example.com", list_id=lst2.list_id, site_id="witchcraft"
        )

        all_lists = builder.list_lists(site_id="witchcraft")
        assert len(all_lists) >= 2


# ===================================================================
# Persistence
# ===================================================================

class TestPersistence:
    """Test data persistence helpers."""

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "lists.json"
        _save_json(path, {"lists": {}, "segments": {}})
        loaded = _load_json(path)
        assert "lists" in loaded

    def test_load_missing_default(self, tmp_path):
        result = _load_json(tmp_path / "absent.json", {})
        assert result == {}

    def test_load_corrupt_default(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json!", encoding="utf-8")
        result = _load_json(path, {"fallback": True})
        assert result == {"fallback": True}


# ===================================================================
# EmailCampaign dataclass
# ===================================================================

class TestEmailCampaign:
    """Test EmailCampaign dataclass."""

    def test_default_campaign(self):
        camp = EmailCampaign()
        assert camp.campaign_id
        assert camp.status == "scheduled"
        assert camp.recipients == 0

    def test_with_fields(self):
        camp = EmailCampaign(
            name="February Promo",
            subject="Special Moon Ritual",
            body="Dear reader...",
            recipients=500,
        )
        assert camp.name == "February Promo"
        assert camp.recipients == 500
