"""
Heavy-simulation tool the agent calls to validate a candidate design.

`run_simulation` is exposed to the LLM as a function tool. The agent decides
when a design needs simulation-based validation (vs a closed-form check); this
layer dispatches the job to the active compute backend. In production that is
Google Cloud (Vertex AI Custom Job, GPU workers); in review mode it runs
in-process so the flow can be inspected without credentials.

Molecular dynamics is the reference domain shipped here; any physics domain
that needs heavy simulation is dispatched through the same path.
"""

from __future__ import annotations

from typing import Any

from ..compute import get_backend
from ..compute.backend import HEAVY_STEP_THRESHOLD, is_heavy


def run_simulation(
    domain: str,
    spec: str,
    steps: int,
    temperature_k: float = 300.0,
) -> dict[str, Any]:
    """Run a numerical physics simulation to validate a candidate design.

    Use this only when validation requires numerical simulation - quantities
    that come from integrating a system forward (equilibrium energy, stability,
    dynamic response). For closed-form checks, call the direct domain tools
    (e.g. force, kinetic_energy) instead; they are cheaper and exact. Small jobs
    run in-process; large jobs run on Google Cloud GPU workers (Vertex AI)
    automatically.

    Args:
        domain: Physics domain to simulate, e.g. "molecular_dynamics".
        spec: The system to simulate (a structure, sequence, or identifier the
            domain's engine understands).
        steps: Number of integration steps. Jobs at or above 100000 steps run
            on Vertex AI GPU.
        temperature_k: Simulation temperature in Kelvin (default 300).

    Returns:
        A dict with the converged metrics, which backend ran the job, and a
        recommendation to use GPU compute when the job is heavy.
    """
    backend = get_backend()
    result = backend.run_simulation(
        domain=domain, spec=spec, steps=steps, temperature_k=temperature_k
    )

    payload: dict[str, Any] = {
        "domain": domain,
        "converged": result.converged,
        "backend": result.backend,
        "metrics": result.metrics,
        "detail": result.detail,
    }

    # Recommend GPU compute when a job is heavy but ran in review mode.
    if is_heavy(steps) and result.backend == "local":
        payload["recommendation"] = (
            f"This job ({steps} steps) exceeds the heavy threshold of "
            f"{HEAVY_STEP_THRESHOLD} steps. For production-accuracy results, "
            "run it on Google Cloud GPU compute (Vertex AI) by setting AXIOM_COMPUTE_BACKEND=vertex."
        )

    return payload
