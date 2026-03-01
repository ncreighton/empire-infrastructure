# Grimoire Intelligence

Witchcraft practice companion with algorithmic intelligence. Provides spell crafting, ritual design, moon phase guidance, tarot readings, and practice tracking — all without AI API costs.

## Trigger Phrases

- "Consult the grimoire about [topic]"
- "Craft a spell for [intention]"
- "Create a ritual for [purpose]"
- "What's the current moon energy?"
- "Give me a tarot reading"
- "Log my practice session"
- "Show my practice journey"
- "Weekly energy forecast"

## API Endpoints

| Method | Path | Handler | File |
|--------|------|---------|------|
| GET | `/` | `root` | `api\app.py` |
| POST | `/consult` | `consult` | `api\app.py` |
| POST | `/craft/meditation` | `craft_meditation` | `api\app.py` |
| POST | `/craft/ritual` | `craft_ritual` | `api\app.py` |
| POST | `/craft/spell` | `craft_spell` | `api\app.py` |
| GET | `/daily` | `daily` | `api\app.py` |
| GET | `/energy` | `energy` | `api\app.py` |
| GET | `/forecast` | `forecast` | `api\app.py` |
| GET | `/health` | `health` | `api\app.py` |
| GET | `/journey` | `journey` | `api\app.py` |
| GET | `/knowledge/colors` | `knowledge_colors` | `api\app.py` |
| GET | `/knowledge/crystals` | `knowledge_crystals` | `api\app.py` |
| GET | `/knowledge/elements` | `knowledge_elements` | `api\app.py` |
| GET | `/knowledge/herbs` | `knowledge_herbs` | `api\app.py` |
| GET | `/knowledge/intentions` | `knowledge_intentions` | `api\app.py` |
| GET | `/knowledge/moon-phases` | `knowledge_moon_phases` | `api\app.py` |
| GET | `/knowledge/planetary-hours` | `knowledge_planetary_hours` | `api\app.py` |
| GET | `/knowledge/sabbats` | `knowledge_sabbats` | `api\app.py` |
| GET | `/knowledge/spell-types` | `knowledge_spell_types` | `api\app.py` |
| GET | `/knowledge/tarot` | `knowledge_tarot` | `api\app.py` |
| POST | `/log` | `log_practice` | `api\app.py` |
| POST | `/tarot/log` | `tarot_log` | `api\app.py` |
| POST | `/tarot/spread` | `tarot_spread` | `api\app.py` |

## Key Components

- **PracticeCodex** (`grimoire\forge\practice_codex.py`) — 36 methods: SQLite-backed learning engine that grows with the practitioner.  Every practice session, tarot readi
- **AmplifyPipeline** (`grimoire\amplify\amplify_pipeline.py`) — 22 methods: Six-stage enhancement pipeline for RitualPlan objects.  Usage:     pipeline = AmplifyPipeline()     
- **GrimoireEngine** (`grimoire\grimoire_engine.py`) — 21 methods: Master orchestrator for the Grimoire Intelligence System.  Wires together FORGE (5 intelligence modu
- **RitualSentinel** (`grimoire\forge\ritual_sentinel.py`) — 17 methods: Scores ritual/spell plans on 6 criteria (100 points) and auto-enhances.  Usage::      sentinel = Rit
- **MoonOracle** (`grimoire\forge\moon_oracle.py`) — 14 methods: Timing intelligence engine for magical practice.  Combines lunar, planetary, zodiacal, and seasonal 
- **TestCorrespondences** (`tests\test_knowledge.py`) — 13 methods
- **SpellSmith** (`grimoire\forge\spell_smith.py`) — 13 methods: Template-based generator for spells, rituals, meditations, daily practices, tarot spreads, and journ
- **MysticEnhancer** (`grimoire\enhancer\mystic_enhancer.py`) — 12 methods: Auto-detects query type and enriches every spiritual query.  Layers applied to each query:     1. Kn
- **TestPoolIntegrity** (`tests\test_variation_engine.py`) — 11 methods
- **SpellScout** (`grimoire\forge\spell_scout.py`) — 10 methods: Intention analysis and correspondence recommendation engine.  The SpellScout reads any intention str
- **TestTarot** (`tests\test_knowledge.py`) — 8 methods
- **CodexAdvisor** (`grimoire\forge\codex_advisor.py`) — 8 methods: Bridges PracticeCodex user history into generation logic.
- **TestWithData** (`tests\test_codex_advisor.py`) — 7 methods
- **VariationEngine** (`grimoire\forge\variation_engine.py`) — 7 methods: Weighted pool selection with SQLite-backed anti-repetition tracking.
- **TestMoonPhases** (`tests\test_knowledge.py`) — 6 methods

## Key Functions

- `root()` (`api\app.py`)
- `health()` (`api\app.py`)
- `consult(req)` (`api\app.py`)
- `energy(lat, lon)` (`api\app.py`)
- `forecast()` (`api\app.py`)
- `craft_spell(req)` (`api\app.py`)
- `craft_ritual(req)` (`api\app.py`)
- `craft_meditation(req)` (`api\app.py`)
- `daily()` (`api\app.py`)
- `log_practice(req)` (`api\app.py`)
- `journey()` (`api\app.py`)
- `tarot_spread(req)` (`api\app.py`)
- `tarot_log(req)` (`api\app.py`)
- `knowledge_herbs(q)` (`api\app.py`)
- `knowledge_crystals(q)` (`api\app.py`)
- `knowledge_colors()` (`api\app.py`)
- `knowledge_elements()` (`api\app.py`)
- `knowledge_intentions()` (`api\app.py`)
- `knowledge_tarot(q)` (`api\app.py`)
- `knowledge_moon_phases()` (`api\app.py`)

## Stats

- **Functions**: 401
- **Classes**: 60
- **Endpoints**: 23
- **Files**: 41
- **Category**: witchcraft-sites
- **Tech Stack**: python, claude-code
