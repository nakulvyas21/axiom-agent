"""
Compute backend interface and resolution.

`get_backend()` returns the active backend:

  1. A backend registered via `register_backend()`.
  2. The backend named by `AXIOM_COMPUTE_BACKEND`:
       - "vertex"  -> VertexComputeBackend (Vertex AI GPU compute, production)
       - "local"   -> LocalBackend (review mode, no credentials)
  3. `LocalBackend` if nothing is set, so the agent can be reviewed offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SimulationResult:
    """Result of a heavy simulation, regardless of where it ran."""

    converged: bool
    # Where the work executed: "local" or "vertex-custom-job".
    backend: str
    # Scalar outputs of the simulation (e.g. potential energy, RMSD).
    metrics: dict[str, float] = field(default_factory=dict)
    # Free-form notes (job id, wall-clock, warnings).
    detail: str = ""


@runtime_checkable
class ComputeBackend(Protocol):
    """Where a heavy simulation runs."""

    name: str

    def run_simulation(
        self,
        domain: str,
        spec: str,
        steps: int,
        temperature_k: float,
    ) -> SimulationResult:
        """Run a simulation for `domain` and return its converged metrics."""
        ...


_BACKEND: ComputeBackend | None = None


def register_backend(backend: ComputeBackend) -> None:
    """Register the compute backend the agent should use."""
    global _BACKEND
    if not isinstance(backend, ComputeBackend):
        raise TypeError("backend must satisfy the ComputeBackend protocol")
    _BACKEND = backend


def get_backend() -> ComputeBackend:
    """Return the active compute backend (see module docstring for order)."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    choice = os.getenv("AXIOM_COMPUTE_BACKEND", "local").lower()
    if choice == "vertex":
        from .vertex import VertexComputeBackend

        _BACKEND = VertexComputeBackend()
        return _BACKEND

    from .local import LocalBackend

    _BACKEND = LocalBackend()
    return _BACKEND


# Heuristic the agent uses to decide local vs Vertex AI GPU. Exposed so the tool layer
# and tests share one definition of "heavy".
HEAVY_STEP_THRESHOLD = 100_000


def is_heavy(steps: int) -> bool:
    """True when a job is large enough to be worth offloading to Vertex AI GPU."""
    return steps >= HEAVY_STEP_THRESHOLD
