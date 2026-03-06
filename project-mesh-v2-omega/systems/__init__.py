"""
Empire Intelligence Systems Registry
=====================================
10 interconnected systems that turn data into action.

Wave 1 (Foundation):     Self-Healing, Opportunity Finder
Wave 2 (Content Intel):  Intelligence Amplifier, Cross-Pollination, Cascade Engine
Wave 3 (Analytics):      Economics Engine, Predictive Layer, Enhancement Enhancer
Wave 4 (Automation):     Autonomous Project Launcher
Wave 5 (Meta):           Infinite Feedback Loop
"""

SYSTEMS = {
    "self_healing": {
        "name": "Self-Healing Infrastructure",
        "module": "systems.self_healing",
        "wave": 1,
        "description": "Auto-restart services, diagnose WordPress, investigate traffic drops",
    },
    "opportunity_finder": {
        "name": "Opportunity Finder",
        "module": "systems.opportunity_finder",
        "wave": 1,
        "description": "Multi-dimensional keyword/content scoring from GSC/GA4/Bing data",
    },
    "intelligence_amplifier": {
        "name": "Intelligence Amplifier",
        "module": "systems.intelligence_amplifier",
        "wave": 2,
        "description": "Learns what content wins, builds niche playbooks",
    },
    "cross_pollination": {
        "name": "Cross-Pollination Engine",
        "module": "systems.cross_pollination",
        "wave": 2,
        "description": "Exploits 16-site network: overlap detection, cross-linking",
    },
    "cascade_engine": {
        "name": "Compound Cascade Engine",
        "module": "systems.cascade_engine",
        "wave": 2,
        "description": "One article triggers 10+ assets across 8+ platforms",
    },
    "economics_engine": {
        "name": "Empire Economics Engine",
        "module": "systems.economics_engine",
        "wave": 3,
        "description": "Full P&L per article/site/niche, ROI, investment allocation",
    },
    "predictive_layer": {
        "name": "Predictive Intelligence Layer",
        "module": "systems.predictive_layer",
        "wave": 3,
        "description": "Algo update detection, decay prediction, revenue forecasting",
    },
    "enhancement_enhancer": {
        "name": "Enhancement Enhancer",
        "module": "systems.enhancement_enhancer",
        "wave": 3,
        "description": "Quality monitoring, A/B experiments, config propagation",
    },
    "project_launcher": {
        "name": "Autonomous Project Launcher",
        "module": "systems.project_launcher",
        "wave": 4,
        "description": "Niche research -> ROI projection -> full automated site launch",
    },
    "feedback_loop": {
        "name": "Infinite Feedback Loop",
        "module": "systems.feedback_loop",
        "wave": 5,
        "description": "Master orchestrator connecting all 9 into compounding improvement cycles",
    },
}


def get_system(name: str):
    """Get system info by name."""
    return SYSTEMS.get(name)


def list_systems(wave: int = None):
    """List all systems, optionally filtered by wave."""
    if wave:
        return {k: v for k, v in SYSTEMS.items() if v["wave"] == wave}
    return SYSTEMS
