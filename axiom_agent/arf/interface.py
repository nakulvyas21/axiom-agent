"""
ARF interface - the contract between the agent and the reasoning backend.

The agent talks only to the `AxiomEngine` protocol defined here. At runtime,
`get_engine()` returns the active backend: `ReferenceAxiomEngine` by default,
or an alternative registered through `register_engine()` / `AXIOM_ARF_ENGINE`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AxiomResult:
    """Structured output of an axiom-reasoning step.

    `confidence` scores how strongly the premises support the claim, and
    `reasoning_graph` records the nodes and edges relating them. This schema is
    the contract that every `AxiomEngine` backend returns.
    """

    claim: str
    supported: bool
    confidence: float
    # A minimal, serialisable reasoning graph: nodes + directed edges.
    reasoning_graph: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@runtime_checkable
class AxiomEngine(Protocol):
    """Stable interface every ARF implementation must satisfy."""

    def derive(self, premises: list[str], query: str) -> AxiomResult:
        """Derive whether `query` follows from `premises`.

        Returns an `AxiomResult` with a support decision, a confidence score,
        and the reasoning graph relating the premises to the query.
        """
        ...

    def rank_interventions(
        self, state: dict[str, Any], goal: str
    ) -> list[dict[str, Any]]:
        """Rank candidate interventions by predicted effect on `goal`.

        Given the current `state`, returns the available interventions ordered
        by how strongly each is expected to move the system toward `goal`.
        """
        ...


_ENGINE: AxiomEngine | None = None


def register_engine(engine: AxiomEngine) -> None:
    """Register the backend the agent should use."""
    global _ENGINE
    if not isinstance(engine, AxiomEngine):
        raise TypeError("engine must satisfy the AxiomEngine protocol")
    _ENGINE = engine


def get_engine() -> AxiomEngine:
    """Return the active backend.

    Resolution order:
      1. A backend registered via `register_engine()`.
      2. A backend named by `AXIOM_ARF_ENGINE` as "module:Class".
      3. `ReferenceAxiomEngine`, the default shipped with this package.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    spec = os.getenv("AXIOM_ARF_ENGINE")
    if spec and ":" in spec:
        module_name, class_name = spec.split(":", 1)
        try:
            import importlib

            module = importlib.import_module(module_name)
            _ENGINE = getattr(module, class_name)()
            return _ENGINE
        except Exception:
            pass  # Fall back to the default backend below.

    from .fallback import ReferenceAxiomEngine

    _ENGINE = ReferenceAxiomEngine()
    return _ENGINE
