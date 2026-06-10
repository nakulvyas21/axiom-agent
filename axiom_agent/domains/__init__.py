"""
Physics domain registry.

Axiom Agent is a domain-agnostic design engine: any physics domain plugs in by
registering its validation tools here. The agent then proposes a design,
validates it with the domain's tools, reads the violations, and iterates until
the design is certified.

    from axiom_agent.domains import register_domain, get_tools_for_domain, list_domains

    register_domain("classical_mechanics", "...", tools=[...])
    tools = get_tools_for_domain("classical_mechanics")

Tool functions are plain Python with rich docstrings; each becomes an ADK
function tool whose docstring is passed to Gemini as the tool spec.
"""

from __future__ import annotations

from typing import Callable

DOMAIN_REGISTRY: dict[str, dict] = {}


def register_domain(
    name: str,
    description: str,
    tools: list[Callable],
    requires: list[str] | None = None,
) -> None:
    """Register a physics domain and its validation tools.

    Args:
        name: Domain identifier (e.g. "classical_mechanics").
        description: One-line description shown to the agent.
        tools: Validation tool functions; each becomes an ADK function tool.
        requires: Optional pip packages the domain needs (for user guidance).
    """
    DOMAIN_REGISTRY[name] = {
        "name": name,
        "description": description,
        "tools": tools,
        "requires": requires or [],
    }


def get_tools_for_domain(domain: str) -> list[Callable]:
    """Return the validation tool functions for a domain."""
    if domain not in DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain: {domain!r}. Available: {list(DOMAIN_REGISTRY)}"
        )
    return DOMAIN_REGISTRY[domain]["tools"]


def list_domains() -> list[dict]:
    """Return metadata for all registered domains."""
    return [
        {
            "domain": d["name"],
            "description": d["description"],
            "tools": [f.__name__ for f in d["tools"]],
            "requires": d["requires"],
        }
        for d in DOMAIN_REGISTRY.values()
    ]


# ── Built-in example domains ─────────────────────────────────────────────────
# These illustrate the registry. Real deployments register their own domains.

from .. import tools as _tools  # noqa: E402  (registration import)

register_domain(
    name="classical_mechanics",
    description=(
        "Closed-form classical mechanics: kinematics, dynamics, energy, and "
        "momentum. Validates designs against exact textbook physics."
    ),
    tools=[
        _tools.physics.kinematic_final_velocity,
        _tools.physics.kinematic_displacement,
        _tools.physics.projectile_range,
        _tools.physics.projectile_time_of_flight,
        _tools.physics.kinetic_energy,
        _tools.physics.gravitational_potential_energy,
        _tools.physics.force,
        _tools.physics.momentum,
    ],
)

register_domain(
    name="molecular_dynamics",
    description=(
        "Molecular-dynamics validation for equilibrium energy and stability. "
        "Heavy jobs run on Google Cloud GPU workers (Vertex AI Custom Jobs)."
    ),
    tools=[_tools.run_simulation],
    requires=["google-cloud-aiplatform"],
)


__all__ = ["register_domain", "get_tools_for_domain", "list_domains", "DOMAIN_REGISTRY"]
