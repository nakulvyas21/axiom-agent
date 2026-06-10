"""
Axiom Agent - orchestration core.

An autonomous physics design engine, built on the Google Agent Development Kit
(ADK). Given a goal, the agent proposes a design, validates it against physics,
reads the violations, and iterates until the design is certified. It ties
together:
  1. A Gemini model (via Vertex AI) driving an ADK `LlmAgent`.
  2. Validation tools: closed-form physics checks, plus `run_simulation` for
     designs that need simulation (e.g. molecular dynamics), which runs on
     Google Cloud GPU compute (Vertex AI) for heavy jobs.
  3. The Axiom Reasoning Framework (ARF) for grounding conclusions in axioms.

The agent depends only on the ARF `AxiomEngine` protocol and the
`ComputeBackend` protocol, so the reasoning backend and the compute backend can
each be swapped without touching this module.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from .arf import get_engine
from .tools import TOOL_FUNCTIONS, call_tool

SYSTEM_PROMPT = """You are the Axiom Agent, an autonomous physics design engine.
Given a goal, you design a solution that is certified by physics:
  1. Identify the governing physical principles and propose a candidate design.
  2. Validate it with the provided tools - never guess a number you can compute
     with a tool. For designs that need simulation (equilibrium energy,
     stability), call run_simulation with the appropriate domain; heavy jobs
     run on Google Cloud GPU compute (Vertex AI) automatically.
  3. Read the violations, revise the design, and re-validate until it passes.
  4. Ground the certified design against the stated axioms.
Always show each iteration: what you proposed, what failed, and how you fixed it.
"""


@dataclass
class Step:
    kind: str  # "tool" | "arf" | "answer"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)


class AxiomAgent:
    """Plans with an ADK LlmAgent, executes tools, grounds results with ARF."""

    def __init__(self, model: str | None = None, max_turns: int = 6) -> None:
        self.model_name = model or os.getenv("AXIOM_MODEL", "gemini-2.5-pro")
        self.max_turns = max_turns
        self._engine = get_engine()
        self._adk = _maybe_build_adk()

    # ── Public API ─────────────────────────────────────────────────────
    def solve(self, problem: str, axioms: list[str] | None = None) -> AgentResult:
        """Solve a problem end-to-end.

        Falls back to a deterministic offline plan when ADK/Gemini is not
        configured, so the agent always runs (great for CI and demos).
        """
        if self._adk is None:
            return self._solve_offline(problem, axioms or [])
        return asyncio.run(self._solve_with_adk(problem, axioms or []))

    # ── ARF grounding ──────────────────────────────────────────────────
    def _ground(self, premises: list[str], claim: str) -> Step:
        # Delegate to the active ARF backend via the AxiomEngine protocol.
        result = self._engine.derive(premises, claim)
        return Step(
            kind="arf",
            detail={
                "claim": result.claim,
                "supported": result.supported,
                "confidence": result.confidence,
                "rationale": result.rationale,
            },
        )

    # ── ADK-backed loop ────────────────────────────────────────────────
    async def _solve_with_adk(self, problem: str, axioms: list[str]) -> AgentResult:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types as genai_types

        agent = self._adk
        session_svc = InMemorySessionService()
        session_id = str(uuid.uuid4())
        await session_svc.create_session(
            app_name="axiom", user_id="agent", session_id=session_id
        )
        runner = Runner(agent=agent, app_name="axiom", session_service=session_svc)

        message = genai_types.Content(
            role="user", parts=[genai_types.Part(text=problem)]
        )

        steps: list[Step] = []
        answer = ""
        pending_call: dict[str, Any] = {}

        async for event in runner.run_async(
            user_id="agent", session_id=session_id, new_message=message
        ):
            if not event.content or not event.content.parts:
                continue
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    pending_call = dict(part.function_call.args or {})
                    pending_call["__name__"] = part.function_call.name
                elif getattr(part, "function_response", None):
                    fr = part.function_response
                    steps.append(
                        Step(
                            kind="tool",
                            detail={
                                "name": fr.name,
                                "args": {k: v for k, v in pending_call.items() if k != "__name__"},
                                "result": dict(fr.response) if fr.response else {},
                            },
                        )
                    )
                elif getattr(part, "text", None):
                    answer = part.text

        steps.append(self._ground(axioms, answer))
        steps.append(Step(kind="answer", detail={"text": answer}))
        return AgentResult(answer=answer, steps=steps)

    # ── Offline deterministic path (no ADK/Gemini configured) ──────────
    def _solve_offline(self, problem: str, axioms: list[str]) -> AgentResult:
        """Run a fixed tool plan when ADK/Gemini is not configured.

        This lets the agent run with zero credentials. When ADK is configured,
        the LlmAgent plans the tool calls instead.
        """
        steps: list[Step] = []
        # Compute kinetic energy of a 2 kg mass at 3 m/s as a worked example.
        value = call_tool("kinetic_energy", {"mass": 2.0, "velocity": 3.0})
        steps.append(
            Step(
                kind="tool",
                detail={"name": "kinetic_energy", "args": {"mass": 2.0, "velocity": 3.0}, "result": str(value)},
            )
        )
        answer = f"[review mode] Kinetic energy = {value}. Configure Vertex AI for the full ADK + Gemini loop over: {problem!r}"
        steps.append(self._ground(axioms, answer))
        steps.append(Step(kind="answer", detail={"text": answer}))
        return AgentResult(answer=answer, steps=steps)


def _maybe_build_adk() -> Any | None:
    """Build the ADK LlmAgent if ADK is installed and Vertex AI is configured."""
    try:
        from google.adk.agents import LlmAgent  # type: ignore
    except Exception:
        return None

    # ADK uses Vertex AI when a project is configured (no API key needed).
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not (project or os.getenv("GOOGLE_API_KEY")):
        return None
    if project:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")

    try:
        return LlmAgent(
            name="axiom_agent",
            model=os.getenv("AXIOM_MODEL", "gemini-2.5-pro"),
            instruction=SYSTEM_PROMPT,
            tools=list(TOOL_FUNCTIONS.values()),
        )
    except Exception:
        return None


def _pretty(result: AgentResult) -> str:  # pragma: no cover - cosmetic
    return json.dumps(
        {"answer": result.answer, "steps": [s.__dict__ for s in result.steps]},
        indent=2,
    )
