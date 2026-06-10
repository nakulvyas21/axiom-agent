"""
Physics tools the agent can call.

Classical-mechanics, energy, and units calculations. Each function is pure,
typed, and unit-test covered, giving the agent exact quantities to reason with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Standard gravity on Earth's surface (m/s^2).
G_EARTH = 9.80665


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.value:g} {self.unit}"


def kinematic_final_velocity(v0: float, a: float, t: float) -> Quantity:
    """v = v0 + a·t  (constant acceleration)."""
    return Quantity(v0 + a * t, "m/s")


def kinematic_displacement(v0: float, a: float, t: float) -> Quantity:
    """s = v0·t + ½·a·t²  (constant acceleration)."""
    return Quantity(v0 * t + 0.5 * a * t * t, "m")


def projectile_range(v0: float, angle_deg: float, g: float = G_EARTH) -> Quantity:
    """Horizontal range of a projectile launched on flat ground.

    R = v0²·sin(2θ) / g
    """
    if v0 < 0:
        raise ValueError("initial speed must be non-negative")
    theta = math.radians(angle_deg)
    return Quantity((v0 * v0) * math.sin(2 * theta) / g, "m")


def projectile_time_of_flight(v0: float, angle_deg: float, g: float = G_EARTH) -> Quantity:
    """t = 2·v0·sin(θ) / g  (flat ground)."""
    theta = math.radians(angle_deg)
    return Quantity(2 * v0 * math.sin(theta) / g, "s")


def kinetic_energy(mass: float, velocity: float) -> Quantity:
    """KE = ½·m·v²."""
    if mass < 0:
        raise ValueError("mass must be non-negative")
    return Quantity(0.5 * mass * velocity * velocity, "J")


def gravitational_potential_energy(
    mass: float, height: float, g: float = G_EARTH
) -> Quantity:
    """PE = m·g·h."""
    if mass < 0:
        raise ValueError("mass must be non-negative")
    return Quantity(mass * g * height, "J")


def force(mass: float, acceleration: float) -> Quantity:
    """Newton's second law: F = m·a."""
    return Quantity(mass * acceleration, "N")


def momentum(mass: float, velocity: float) -> Quantity:
    """p = m·v."""
    return Quantity(mass * velocity, "kg·m/s")
