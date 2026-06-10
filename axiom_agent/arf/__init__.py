"""
Axiom Reasoning Framework (ARF).

ARF is the reasoning layer of the Axiom Agent. It defines the `AxiomEngine`
protocol and resolves the active backend through `get_engine()`. The package
ships `ReferenceAxiomEngine` as the default; deployments may register their own
backend through the same interface.
"""

from .interface import AxiomEngine, AxiomResult, get_engine, register_engine

__all__ = ["AxiomEngine", "AxiomResult", "get_engine", "register_engine"]
