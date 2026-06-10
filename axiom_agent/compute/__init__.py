"""
Compute backends for the Axiom Agent.

Heavy simulations (e.g. molecular dynamics) are delegated to a
`ComputeBackend`, which the agent selects at runtime:

  • `VertexComputeBackend` - the production path. Submits the job to Google
    Cloud Vertex AI as a Custom Job on GPU workers, polls to completion, and
    returns the result.
  • `LocalBackend` - a self-contained review mode that runs an in-process
    approximation, so the agent can be inspected end to end without cloud
    credentials.

The agent depends only on the `ComputeBackend` protocol, so the compute backend
evolves without changing any agent or tool code.
"""

from .backend import ComputeBackend, SimulationResult, get_backend, register_backend

__all__ = [
    "ComputeBackend",
    "SimulationResult",
    "get_backend",
    "register_backend",
]
