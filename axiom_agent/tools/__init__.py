"""
Tool registry for the Axiom Agent.

Each tool is a plain Python function plus a JSON-schema declaration that the
LLM uses for function-calling. This module assembles the registry the agent
exposes to the model.
"""

from __future__ import annotations

from typing import Any, Callable

from . import physics
from .simulation import run_simulation

# A tool is (callable, schema). Schemas follow the Gemini / OpenAI
# function-declaration shape so they map cleanly onto any LLM provider.
ToolFn = Callable[..., Any]


def _num(desc: str) -> dict[str, Any]:
    return {"type": "number", "description": desc}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "kinematic_final_velocity",
        "description": "Final velocity under constant acceleration: v = v0 + a*t.",
        "parameters": {
            "type": "object",
            "properties": {
                "v0": _num("initial velocity (m/s)"),
                "a": _num("acceleration (m/s^2)"),
                "t": _num("time (s)"),
            },
            "required": ["v0", "a", "t"],
        },
    },
    {
        "name": "kinematic_displacement",
        "description": "Displacement under constant acceleration: s = v0*t + 0.5*a*t^2.",
        "parameters": {
            "type": "object",
            "properties": {
                "v0": _num("initial velocity (m/s)"),
                "a": _num("acceleration (m/s^2)"),
                "t": _num("time (s)"),
            },
            "required": ["v0", "a", "t"],
        },
    },
    {
        "name": "projectile_range",
        "description": "Horizontal range of a projectile on flat ground.",
        "parameters": {
            "type": "object",
            "properties": {
                "v0": _num("launch speed (m/s)"),
                "angle_deg": _num("launch angle in degrees"),
            },
            "required": ["v0", "angle_deg"],
        },
    },
    {
        "name": "projectile_time_of_flight",
        "description": "Time of flight of a projectile on flat ground.",
        "parameters": {
            "type": "object",
            "properties": {
                "v0": _num("launch speed (m/s)"),
                "angle_deg": _num("launch angle in degrees"),
            },
            "required": ["v0", "angle_deg"],
        },
    },
    {
        "name": "kinetic_energy",
        "description": "Kinetic energy: KE = 0.5*m*v^2.",
        "parameters": {
            "type": "object",
            "properties": {
                "mass": _num("mass (kg)"),
                "velocity": _num("velocity (m/s)"),
            },
            "required": ["mass", "velocity"],
        },
    },
    {
        "name": "gravitational_potential_energy",
        "description": "Gravitational potential energy: PE = m*g*h.",
        "parameters": {
            "type": "object",
            "properties": {
                "mass": _num("mass (kg)"),
                "height": _num("height (m)"),
            },
            "required": ["mass", "height"],
        },
    },
    {
        "name": "force",
        "description": "Newton's second law: F = m*a.",
        "parameters": {
            "type": "object",
            "properties": {
                "mass": _num("mass (kg)"),
                "acceleration": _num("acceleration (m/s^2)"),
            },
            "required": ["mass", "acceleration"],
        },
    },
    {
        "name": "momentum",
        "description": "Linear momentum: p = m*v.",
        "parameters": {
            "type": "object",
            "properties": {
                "mass": _num("mass (kg)"),
                "velocity": _num("velocity (m/s)"),
            },
            "required": ["mass", "velocity"],
        },
    },
    {
        "name": "run_simulation",
        "description": (
            "Run a physics simulation to validate a design (e.g. domain "
            "'molecular_dynamics' for equilibrium energy/stability). Heavy jobs "
            "run on Google Cloud GPU compute (Vertex AI) automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "physics domain, e.g. molecular_dynamics"},
                "spec": {"type": "string", "description": "the system to simulate (structure/sequence/identifier)"},
                "steps": {"type": "integer", "description": "integration steps; >=100000 runs on Vertex AI GPU"},
                "temperature_k": _num("temperature in Kelvin (default 300)"),
            },
            "required": ["domain", "spec", "steps"],
        },
    },
]


TOOL_FUNCTIONS: dict[str, ToolFn] = {
    "kinematic_final_velocity": physics.kinematic_final_velocity,
    "kinematic_displacement": physics.kinematic_displacement,
    "projectile_range": physics.projectile_range,
    "projectile_time_of_flight": physics.projectile_time_of_flight,
    "kinetic_energy": physics.kinetic_energy,
    "gravitational_potential_energy": physics.gravitational_potential_energy,
    "force": physics.force,
    "momentum": physics.momentum,
    "run_simulation": run_simulation,
}


def call_tool(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by name with keyword args."""
    if name not in TOOL_FUNCTIONS:
        raise KeyError(f"unknown tool: {name}")
    return TOOL_FUNCTIONS[name](**args)
