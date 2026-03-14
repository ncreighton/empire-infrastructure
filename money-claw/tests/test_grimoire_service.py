"""Tests for the Grimoire Service — Luna's knowledge base adapter."""

from moneyclaw.services.luna.grimoire_service import GrimoireService


def test_grimoire_service_init():
    gs = GrimoireService()
    # Should initialize without error regardless of grimoire-intelligence availability
    assert isinstance(gs.available, bool)


def test_get_current_energy():
    gs = GrimoireService()
    energy = gs.get_current_energy()
    assert "moon" in energy
    assert "sabbat" in energy
    assert "element" in energy
    assert "season" in energy
    assert energy["moon"]["phase"]  # Should have a phase name


def test_lookup_herb():
    gs = GrimoireService()
    herb = gs.lookup_herb("lavender")
    assert herb is not None
    assert herb["name"] == "Lavender"
    assert "uses" in herb
    assert "peace" in herb["uses"]


def test_lookup_herb_not_found():
    gs = GrimoireService()
    herb = gs.lookup_herb("nonexistent_herb_xyz")
    assert herb is None


def test_lookup_crystal():
    gs = GrimoireService()
    crystal = gs.lookup_crystal("amethyst")
    assert crystal is not None
    assert crystal["name"] == "Amethyst"
    assert "uses" in crystal


def test_get_correspondences():
    gs = GrimoireService()
    result = gs.get_correspondences("protection")
    assert result["intention"] == "protection"
    assert len(result["herbs"]) > 0
    assert len(result["crystals"]) > 0


def test_get_correspondences_fuzzy():
    gs = GrimoireService()
    # "sleep" isn't a direct key but "peace" herbs include sleep uses
    result = gs.get_correspondences("sleep")
    assert result["intention"] == "sleep"
    # Should find some matches via fuzzy search
    assert isinstance(result["herbs"], list)


def test_get_daily_practice():
    gs = GrimoireService()
    practice = gs.get_daily_practice()
    assert "practice" in practice
    assert "recommended_herb" in practice
    assert "recommended_crystal" in practice
    assert "moon" in practice
    assert practice["practice"]  # Non-empty string


def test_craft_spell():
    gs = GrimoireService()
    spell = gs.craft_spell("protection")
    assert spell["intention"] == "protection"
    assert spell["title"] == "Spell for Protection"
    assert "ingredients" in spell
    assert "steps" in spell
    assert len(spell["steps"]) >= 3
    assert "herbs" in spell["ingredients"]
    assert "crystals" in spell["ingredients"]


def test_craft_ritual():
    gs = GrimoireService()
    ritual = gs.craft_ritual("love")
    assert ritual["intention"] == "love"
    assert ritual["title"] == "Ritual for Love"
    assert "preparation" in ritual
    assert "steps" in ritual
    assert "closing" in ritual
    assert len(ritual["steps"]) >= 5
