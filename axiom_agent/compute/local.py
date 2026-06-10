"""
Local compute backend.

Runs a fast, in-process approximation of a simulation. This is the review-mode
backend: it lets the agent be inspected end to end without cloud credentials.
Production runs go to `VertexComputeBackend` (Vertex AI GPU).
"""

from __future__ import annotations

import math

from .backend import ComputeBackend, SimulationResult


class LocalBackend(ComputeBackend):
    """In-process approximation. Deterministic and dependency-free."""

    name = "local"

    def run_simulation(
        self,
        domain: str,
        spec: str,
        steps: int,
        temperature_k: float,
    ) -> SimulationResult:
        # A lightweight surrogate for a full integration: estimate equilibrium
        # potential energy from system size and temperature. Good enough to
        # drive the agent loop; a real run goes to the Vertex AI GPU backend.
        n_units = max(len(spec), 1)
        # Equipartition-style estimate: thermal energy scales with size and
        # temperature. Placeholder only; production runs go to the GPU engine.
        kt = 0.00831446 * temperature_k  # gas constant, kJ/mol/K
        potential_energy = -round(n_units * kt * math.log1p(steps), 3)
        rmsd = round(0.1 * math.sqrt(steps) / (1 + n_units), 4)

        return SimulationResult(
            converged=steps > 0,
            backend=self.name,
            metrics={
                "potential_energy_kj_mol": potential_energy,
                "rmsd_nm": rmsd,
                "size": float(n_units),
            },
            detail=(
                f"Local approximation of '{domain}' over {steps} steps at "
                f"{temperature_k} K. Use the Vertex backend for production runs."
            ),
        )
