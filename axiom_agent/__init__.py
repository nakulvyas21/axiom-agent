"""Axiom Agent - an autonomous physics design engine on the Axiom Reasoning Framework."""

from .agent import AgentResult, AxiomAgent, Step
from . import domains  # noqa: F401  (registers built-in physics domains)

__version__ = "0.1.0"
__all__ = ["AxiomAgent", "AgentResult", "Step", "domains", "__version__"]
