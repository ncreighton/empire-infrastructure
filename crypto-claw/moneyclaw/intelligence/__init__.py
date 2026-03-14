"""
MoneyClaw Intelligence Layer — Autonomous learning brain that grows smarter
with every trade.

Public API:
    from moneyclaw.intelligence import IntelligenceBrain
    brain = IntelligenceBrain(db)
    brain.tick()
    delta = brain.get_skill_adjustments(product_id, strategy, regime, indicators)
"""

from moneyclaw.intelligence.brain import IntelligenceBrain

__all__ = ["IntelligenceBrain"]
