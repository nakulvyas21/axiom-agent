import math

import pytest

from axiom_agent.tools import physics


def test_final_velocity():
    assert physics.kinematic_final_velocity(0, 9.8, 2).value == pytest.approx(19.6)


def test_displacement():
    assert physics.kinematic_displacement(0, 2, 3).value == pytest.approx(9.0)


def test_projectile_range_45deg():
    # R = v0^2 * sin(90) / g = 400 / 9.80665
    r = physics.projectile_range(20, 45).value
    assert r == pytest.approx(400 / physics.G_EARTH, rel=1e-6)


def test_projectile_time_of_flight():
    t = physics.projectile_time_of_flight(20, 90).value
    assert t == pytest.approx(2 * 20 / physics.G_EARTH, rel=1e-6)


def test_kinetic_energy():
    assert physics.kinetic_energy(2, 3).value == pytest.approx(9.0)


def test_pe():
    assert physics.gravitational_potential_energy(2, 10).value == pytest.approx(2 * 10 * physics.G_EARTH)


def test_force_and_momentum():
    assert physics.force(5, 2).value == pytest.approx(10.0)
    assert physics.momentum(5, 2).value == pytest.approx(10.0)


def test_negative_mass_rejected():
    with pytest.raises(ValueError):
        physics.kinetic_energy(-1, 3)
