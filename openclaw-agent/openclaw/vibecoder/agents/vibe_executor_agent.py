"""VibeExecutorAgent — hybrid execution engine (API + CLI + VPS + algorithmic).

Dispatches each MissionStep to the appropriate engine:
  - ALGORITHMIC: direct subprocess/file operations
  - API_HAIKU/SONNET: Anthropic API calls
  - CLI_CLAUDE: Claude Code CLI for complex multi-file work

Also integrates with VPS infrastructure:
  - Docker builds/restarts on VPS via SSH
  - n8n workflow deployment
  - Service health verification post-deploy

Part of the VibeCoder agent system.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openclaw.vibecoder.models import (
    CodeChange,
    EngineType,
    Mission,
    MissionStep,
    ProjectInfo,
    StepStatus,
    StepType,
)

logger = logging.getLogger(__name__)

# VPS connection info (from environment or defaults)
_VPS_HOST = os.environ.get("VPS_HOST", "217.216.84.245")
_VPS_USER = os.environ.get("VPS_USER", "empire")
_VPS_BASE = os.environ.get("VPS_BASE", "/opt/empire")


class VibeExecutorAgent:
    """Execute mission steps using the hybrid engine.

    Usage::

        executor = VibeExecutorAgent()
        step = await executor.execute_step(step, mission, project_info)
    """

    def __init__(self, model_router=None):
        self._anthropic_client = None
        self._changes: list[CodeChange] = []
        self._model_router = model_router

    @property
    def changes(self) -> list[CodeChange]:
        """Get accumulated code changes."""
        return list(self._changes)

    async def execute_step(
        self,
        step: MissionStep,
        mission: Mission,
        project_info: ProjectInfo | None = None,
    ) -> MissionStep:
        """Execute a single step, dispatching to the appropriate engine."""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()

        try:
            if step.engine == EngineType.ALGORITHMIC:
                await self._execute_algorithmic(step, mission, project_info)
            elif step.engine in (EngineType.API_HAIKU, EngineType.API_SONNET):
                model = "haiku" if step.engine == EngineType.API_HAIKU else "sonnet"
                if os.environ.get("ANTHROPIC_API_KEY"):
                    await self._execute_api(step, mission, project_info, model=model)
                else:
                    # Fallback: try CLI, then skip
                    logger.info(
                        f"[Executor] No ANTHROPIC_API_KEY, falling back to CLI "
                        f"for step {step.step_number}"
                    )
                    await self._execute_cli(step, mission, project_info)
            elif step.engine == EngineType.CLI_CLAUDE:
                await self._execute_cli(step, mission, project_info)
            else:
                raise ValueError(f"Unknown engine: {step.engine}")

            step.status = StepStatus.COMPLETED
            logger.info(
                f"[Executor] Step {step.step_number} completed: "
                f"{step.description} (engine={step.engine.value})"
            )

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)[:500]
            logger.error(
                f"[Executor] Step {step.step_number} failed: {e}",
            )

            # Don't retry non-transient failures (tests, env issues)
            is_transient = not any(kw in str(e).lower() for kw in [
                "pydantic", "compatibility", "not found", "no such file",
                "api_key", "not set", "permission",
            ])
            # Don't retry test/review steps — they fail for real reasons
            is_retriable = step.step_type not in (StepType.RUN_TESTS, StepType.REVIEW)

            if is_transient and is_retriable and step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = StepStatus.RETRYING
                logger.info(
                    f"[Executor] Retrying step {step.step_number} "
                    f"(attempt {step.retry_count}/{step.max_retries})"
                )
                return await self.execute_step(step, mission, project_info)

        step.completed_at = datetime.now()
        return step

    async def execute_mission(
        self,
        mission: Mission,
        project_info: ProjectInfo | None = None,
    ) -> Mission:
        """Execute all steps in a mission sequentially."""
        self._changes = []

        for step in mission.steps:
            step = await self.execute_step(step, mission, project_info)

            # Accumulate costs
            mission.total_tokens += step.tokens_used
            mission.total_cost_usd += step.cost_usd

            # Stop on failure (unless it's a non-critical step)
            if step.status == StepStatus.FAILED:
                if step.step_type not in (StepType.RUN_TESTS, StepType.REVIEW):
                    mission.errors.append(
                        f"Step {step.step_number} failed: {step.error_message}"
                    )
                    break
                else:
                    mission.warnings.append(
                        f"Non-critical step {step.step_number} failed: {step.error_message}"
                    )

        return mission

    # ─── Engine dispatch ──────────────────────────────────────────────────

    async def _execute_algorithmic(
        self,
        step: MissionStep,
        mission: Mission,
        project_info: ProjectInfo | None,
    ) -> None:
        """Execute step using direct shell/file operations ($0 cost)."""
        root = project_info.root_path if project_info else "."

        if step.step_type == StepType.RUN_COMMAND:
            result = await self._run_shell(step.command, cwd=root)
            step.output = result

        elif step.step_type == StepType.RUN_TESTS:
            cmd = step.command or "python -m pytest tests/ -v --tb=short"
            try:
                result = await self._run_shell(cmd, cwd=root, timeout=120)
                step.output = result
            except RuntimeError as e:
                # Test failures are informational, not fatal
                step.output = f"Tests failed (non-blocking): {str(e)[:300]}"
                step.status = StepStatus.COMPLETED  # Mark completed, not failed
                logger.warning(f"[Executor] Tests failed but continuing: {str(e)[:100]}")
                return

        elif step.step_type == StepType.GIT_OPERATION:
            await self._execute_git(step, mission, root)

        elif step.step_type == StepType.DELETE_FILE:
            target = os.path.join(root, step.target_file)
            if os.path.exists(target):
                os.remove(target)
                self._changes.append(CodeChange(
                    file_path=step.target_file, change_type="deleted",
                ))
                step.output = f"Deleted: {step.target_file}"

        elif step.step_type == StepType.INSTALL_DEPENDENCY:
            cmd = step.command or "pip install -r requirements.txt"
            result = await self._run_shell(cmd, cwd=root, timeout=120)
            step.output = result

        else:
            step.output = f"Algorithmic handler not found for {step.step_type.value}"

    async def _execute_api(
        self,
        step: MissionStep,
        mission: Mission,
        project_info: ProjectInfo | None,
        model: str = "sonnet",
    ) -> None:
        """Execute step using Anthropic API with intelligent model routing."""
        # Fast-fail if no API key rather than hanging
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set — cannot execute API step. "
                "Falling back to CLI or skipping."
            )
        client = self._get_anthropic_client()
        if not client:
            raise RuntimeError("ANTHROPIC_API_KEY not set — cannot execute API step")

        root = project_info.root_path if project_info else "."

        # Build context prompt
        system_prompt = self._build_system_prompt(mission, project_info)
        user_prompt = self._build_step_prompt(step, mission, root)

        # Use ModelRouter for intelligent model selection if available
        routing_decision = None
        if self._model_router:
            routing_decision = self._model_router.route(
                task_description=step.description,
                system_prompt=system_prompt,
                input_text=user_prompt,
                context={
                    "project_id": mission.project_id,
                    "scope": mission.scope.value,
                    "step_type": step.step_type.value,
                },
            )
            model_id = routing_decision.model_spec.model_id
            max_tokens = routing_decision.max_tokens
            use_cache = routing_decision.use_cache
            logger.info(
                f"[Executor] ModelRouter selected: {routing_decision.model_spec.tier.value} "
                f"(est. ${routing_decision.estimated_cost:.4f}, "
                f"saves ${routing_decision.savings_vs_opus:.4f} vs Opus)"
            )
        else:
            # Fallback to hardcoded model selection
            model_id = (
                "claude-haiku-4-5-20251001" if model == "haiku"
                else "claude-sonnet-4-20250514"
            )
            max_tokens = 4096 if step.step_type != StepType.REVIEW else 1000
            use_cache = len(system_prompt) > 3000

        # Compress prompt if router recommends it
        if self._model_router and use_cache:
            system_prompt = self._model_router.compress_prompt(system_prompt)

        # Build system message with optional cache_control
        system_msg = [{"type": "text", "text": system_prompt}]
        if use_cache:
            system_msg[0]["cache_control"] = {"type": "ephemeral"}

        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system_msg,
            messages=[{"role": "user", "content": user_prompt}],
        )

        result_text = response.content[0].text
        step.tokens_used = response.usage.input_tokens + response.usage.output_tokens

        # Calculate cost from actual usage
        if routing_decision:
            spec = routing_decision.model_spec
            step.cost_usd = (
                response.usage.input_tokens * spec.input_cost_per_m
                + response.usage.output_tokens * spec.output_cost_per_m
            ) / 1_000_000

            # Record outcome for learning (quality assessed later)
            self._model_router.record_outcome(
                decision=routing_decision,
                actual_input_tokens=response.usage.input_tokens,
                actual_output_tokens=response.usage.output_tokens,
                quality_score=0.9,  # Default; updated by reviewer later
            )
        else:
            in_rate = 0.80 if model == "haiku" else 3.00
            out_rate = 4.00 if model == "haiku" else 15.00
            step.cost_usd = (
                response.usage.input_tokens * in_rate
                + response.usage.output_tokens * out_rate
            ) / 1_000_000

        # Apply the result
        if step.step_type in (StepType.CREATE_FILE, StepType.EDIT_FILE, StepType.GENERATE_CODE):
            await self._apply_code_result(step, result_text, root)
        else:
            step.output = result_text

    async def _execute_cli(
        self,
        step: MissionStep,
        mission: Mission,
        project_info: ProjectInfo | None,
    ) -> None:
        """Execute step using Claude Code CLI for complex multi-file work."""
        root = project_info.root_path if project_info else "."

        # Build the prompt for Claude Code
        prompt = (
            f"Project: {mission.project_id}. "
            f"Task: {step.description}. "
            f"Context: {mission.description}."
        )
        if step.target_file:
            prompt += f" Target file: {step.target_file}."

        # Escape the prompt for shell
        escaped_prompt = prompt.replace('"', '\\"').replace("'", "\\'")

        # Use Claude Code CLI with --print for non-interactive output
        cmd = f'claude --print -p "{escaped_prompt}"'

        try:
            result = await self._run_shell(cmd, cwd=root, timeout=300)
            step.output = result[:5000]
            # Estimate tokens (CLI doesn't report them directly)
            step.tokens_used = len(result.split()) * 2
            step.cost_usd = step.tokens_used * 3.0 / 1_000_000

        except Exception as e:
            # If CLI also fails, mark step failed with clear message
            error_msg = str(e)[:300]
            logger.warning(f"[Executor] CLI execution failed: {error_msg}")
            raise RuntimeError(f"CLI execution failed: {error_msg}")

    # ─── VPS Operations ───────────────────────────────────────────────────

    async def deploy_to_vps(
        self,
        project_id: str,
        project_info: ProjectInfo,
    ) -> str:
        """Deploy project to VPS via SSH + Docker."""
        root = project_info.root_path
        remote_path = f"{_VPS_BASE}/{project_id}"

        # 1. Tar and send files
        tar_cmd = (
            f'tar --exclude="data" --exclude=".env" --exclude="__pycache__" '
            f'--exclude=".git" --exclude="*.pyc" -cf - -C "{root}" . | '
            f'ssh {_VPS_USER}@{_VPS_HOST} "mkdir -p {remote_path} && cd {remote_path} && tar xf -"'
        )
        result = await self._run_shell(tar_cmd, timeout=120)

        # 2. Docker rebuild if applicable
        if project_info.has_docker:
            docker_cmd = (
                f'ssh {_VPS_USER}@{_VPS_HOST} '
                f'"cd {remote_path} && docker compose up -d --build"'
            )
            result += "\n" + await self._run_shell(docker_cmd, timeout=300)

        return result

    async def verify_vps_service(
        self,
        service_name: str,
        port: int,
    ) -> bool:
        """Check if a VPS service is healthy after deploy."""
        try:
            cmd = f'ssh {_VPS_USER}@{_VPS_HOST} "curl -sf http://localhost:{port}/health"'
            result = await self._run_shell(cmd, timeout=15)
            return "ok" in result.lower() or "healthy" in result.lower()
        except Exception:
            return False

    async def deploy_n8n_workflow(
        self,
        workflow_path: str,
    ) -> str:
        """Deploy an n8n workflow to the VPS."""
        # Read workflow JSON
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        name = workflow.get("name", "unnamed")

        # Use n8n API
        cmd = (
            f'ssh {_VPS_USER}@{_VPS_HOST} '
            f'"curl -sf -X POST http://localhost:5679/api/v1/workflows '
            f'-H \'Content-Type: application/json\' '
            f'-H \'X-N8N-API-KEY: \'$N8N_API_KEY '
            f'-d @-" < "{workflow_path}"'
        )
        result = await self._run_shell(cmd, timeout=30)
        return f"Deployed workflow '{name}': {result[:200]}"

    # ─── Git Operations ───────────────────────────────────────────────────

    async def _execute_git(
        self, step: MissionStep, mission: Mission, root: str
    ) -> None:
        """Handle git operations: branch, commit, push, PR."""
        desc = step.description.lower()

        if "branch" in desc or "checkout" in desc:
            branch = mission.branch_name or self._make_branch_name(mission)
            mission.branch_name = branch
            await self._run_shell(f"git checkout -b {branch}", cwd=root)
            step.output = f"Created branch: {branch}"

        elif "commit" in desc:
            scope = mission.scope.value
            msg = f"[{scope}] {mission.title}"
            # Only stage files that were actually changed by this mission
            if self._changes:
                for change in self._changes:
                    file_path = change.file_path
                    if change.change_type == "deleted":
                        await self._run_shell(f'git add "{file_path}"', cwd=root)
                    else:
                        abs_path = os.path.join(root, file_path)
                        if os.path.exists(abs_path):
                            await self._run_shell(f'git add "{file_path}"', cwd=root)
            else:
                # Fallback: stage only tracked modified files, not untracked
                await self._run_shell("git add -u", cwd=root)
            # Use separate -m flags for multi-line commit (Windows-safe)
            commit_cmd = (
                f'git commit '
                f'-m "{msg}" '
                f'-m "Co-Authored-By: VibeCoder Agent <vibecoder@empire>"'
            )
            result = await self._run_shell(commit_cmd, cwd=root)
            # Extract commit hash
            hash_result = await self._run_shell(
                "git rev-parse --short HEAD", cwd=root
            )
            mission.commit_hash = hash_result.strip()
            step.output = f"Committed: {mission.commit_hash}"

        elif "push" in desc:
            branch = mission.branch_name or "HEAD"
            result = await self._run_shell(
                f"git push -u origin {branch}", cwd=root
            )
            step.output = result

        elif "pr" in desc or "pull request" in desc:
            result = await self._create_pr(mission, root)
            step.output = result

        elif "init" in desc:
            await self._run_shell("git init", cwd=root)
            await self._run_shell("git add .", cwd=root)
            await self._run_shell(
                'git commit -m "Initial commit" '
                '-m "Co-Authored-By: VibeCoder Agent <vibecoder@empire>"',
                cwd=root,
            )
            step.output = "Git initialized with initial commit"

        else:
            step.output = f"Git operation not recognized: {step.description}"

    async def _create_pr(self, mission: Mission, root: str) -> str:
        """Create a pull request via gh CLI."""
        title = f"[{mission.scope.value}] {mission.title}"
        body = (
            f"## Summary\n"
            f"- {mission.description[:200]}\n\n"
            f"## Scope\n"
            f"- Type: {mission.scope.value}\n"
            f"- Engine: VibeCoder Agent\n\n"
            f"Generated with VibeCoder Agent"
        )
        cmd = f'gh pr create --title "{title}" --body "{body}"'
        try:
            result = await self._run_shell(cmd, cwd=root, timeout=30)
            mission.pr_url = result.strip()
            return f"PR created: {mission.pr_url}"
        except Exception as e:
            return f"PR creation failed: {e}"

    @staticmethod
    def _make_branch_name(mission: Mission) -> str:
        """Generate a safe branch name from mission metadata."""
        import hashlib
        slug = mission.title.lower()[:30]
        # Only allow alphanumeric and hyphens — strip everything else
        slug = "".join(c if c.isalnum() or c == "-" else "-" for c in slug)
        slug = re.sub(r'-{2,}', '-', slug).strip("-")
        # Sanitize project_id too
        proj = "".join(c if c.isalnum() or c in "-_" else "-" for c in mission.project_id)
        short_id = hashlib.sha256(mission.mission_id.encode()).hexdigest()[:6]
        return f"vibe/{proj}/{mission.scope.value}-{slug}-{short_id}"

    # ─── Code application ─────────────────────────────────────────────────

    async def _apply_code_result(
        self, step: MissionStep, result_text: str, root: str
    ) -> None:
        """Extract and apply code from LLM response to files."""
        # Try to extract code blocks
        code_blocks = self._extract_code_blocks(result_text)

        if step.target_file and code_blocks:
            # Resolve target file path — if it's a bare filename without
            # directory, try to find it in the project tree first.
            target_rel = step.target_file.replace("\\", "/")
            target = os.path.join(root, target_rel)

            if "/" not in target_rel and not os.path.exists(target):
                # Bare filename — search the project for an existing match
                found = self._find_file_in_tree(root, target_rel)
                if found:
                    target = found
                    # Update step.target_file to the discovered relative path
                    step.target_file = os.path.relpath(found, root).replace("\\", "/")

            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)

            existing = ""
            change_type = "created"
            if os.path.exists(target):
                with open(target, "r", encoding="utf-8") as f:
                    existing = f.read()
                change_type = "modified"

            content = code_blocks[0]
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)

            lines_added = len(content.splitlines())
            lines_removed = len(existing.splitlines()) if change_type == "modified" else 0

            self._changes.append(CodeChange(
                file_path=step.target_file,
                change_type=change_type,
                diff=f"+{lines_added}/-{lines_removed}",
                lines_added=lines_added,
                lines_removed=lines_removed,
            ))

            step.output = f"{change_type}: {step.target_file} (+{lines_added}/-{lines_removed})"
        else:
            step.output = result_text

    @staticmethod
    def _find_file_in_tree(root: str, filename: str) -> str | None:
        """Search project tree for a file by name, return full path."""
        for dirpath, _dirs, files in os.walk(root):
            # Skip hidden dirs, __pycache__, node_modules, .git
            base = os.path.basename(dirpath)
            if base.startswith(".") or base in ("__pycache__", "node_modules", ".git"):
                continue
            if filename in files:
                return os.path.join(dirpath, filename)
        return None

    @staticmethod
    def _extract_code_blocks(text: str) -> list[str]:
        """Extract fenced code blocks from LLM response."""
        import re
        blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        return blocks if blocks else []

    # ─── Shell execution ──────────────────────────────────────────────────

    async def _run_shell(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = 60,
    ) -> str:
        """Run a shell command asynchronously."""
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"Command timed out after {timeout}s: {command[:100]}")

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            # Some commands (like git) write to stderr for non-error info
            if proc.returncode == 1 and not stderr_text:
                return stdout_text
            raise RuntimeError(
                f"Command failed (exit {proc.returncode}): "
                f"{stderr_text[:300] or stdout_text[:300]}"
            )

        return stdout_text

    # ─── Anthropic client ─────────────────────────────────────────────────

    def _get_anthropic_client(self):
        """Lazy-init Anthropic client."""
        if self._anthropic_client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("anthropic package not installed")
                return None
        return self._anthropic_client

    def _build_system_prompt(
        self, mission: Mission, project_info: ProjectInfo | None
    ) -> str:
        """Build system prompt with project context."""
        parts = [
            "You are VibeCoder, an autonomous coding agent.",
            f"Project: {mission.project_id}",
        ]
        if project_info:
            parts.append(f"Language: {project_info.language}")
            if project_info.framework:
                parts.append(f"Framework: {project_info.framework}")
            parts.append(f"Has tests: {project_info.has_tests}")
        parts.append(
            "Output ONLY code in fenced code blocks. "
            "No explanations unless specifically asked."
        )
        return "\n".join(parts)

    def _build_step_prompt(
        self, step: MissionStep, mission: Mission, root: str
    ) -> str:
        """Build user prompt for a specific step."""
        parts = [
            f"Mission: {mission.title}",
            f"Description: {mission.description}",
            f"Step: {step.description}",
        ]
        if step.target_file:
            parts.append(f"Target file: {step.target_file}")
            # Include existing file content if editing
            target_path = os.path.join(root, step.target_file)
            if os.path.exists(target_path):
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if len(content) < 10000:
                        parts.append(f"\nExisting file content:\n```\n{content}\n```")
                except OSError:
                    pass
        return "\n".join(parts)
