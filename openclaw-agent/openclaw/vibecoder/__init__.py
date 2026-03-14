"""vibecoder — autonomous coding agent for the OpenClaw empire.

Submits natural language missions, decomposes into steps, executes via
hybrid engine (Anthropic API + Claude Code CLI + VPS), reviews, commits,
and deploys.
"""

from openclaw.vibecoder.vibecoder_engine import VibeCoderEngine

__all__ = ["VibeCoderEngine"]
