# Architecture

Axiom Agent is an autonomous physics design engine: given a goal, it proposes a
design, validates it against physics, and iterates until the design is
certified. It runs on **Google Cloud / Vertex AI** - Gemini serves the ADK
agent, and heavy validations run as Vertex AI Custom Jobs on GPU workers. It is
built in five layers, each independently testable.

## Layers

1. **Orchestration (`agent.py`)** - a Google ADK `LlmAgent` driven by a `Runner`
   over an in-memory session. The agent proposes a design, the runner executes
   the domain's validation tools and streams results back, the agent revises and
   re-validates, and the certified design is grounded against the supplied
   axioms. When ADK/Gemini is not configured, a deterministic review path keeps
   the agent runnable.

2. **Domains (`domains/`)** - a registry where any physics domain plugs in its
   validation tools (`register_domain` / `get_tools_for_domain` / `list_domains`).
   The repo ships `classical_mechanics` and `molecular_dynamics` as examples.

3. **Tools (`tools/`)** - the validation tools: typed, unit-tested closed-form
   physics functions, plus `run_simulation` for designs that need simulation.
   Each is a plain function with a rich docstring, registered as an ADK tool.

4. **Compute (`compute/`)** - where heavy simulations run, defined as a
   `ComputeBackend` interface. See *The compute boundary* below.

5. **Axiom Reasoning Framework (`arf/`)** - the reasoning layer, defined as an
   interface with a pluggable backend.

## The reasoning interface

The agent depends only on the `AxiomEngine` protocol, never on a concrete
implementation:

```python
class AxiomEngine(Protocol):
    def derive(self, premises: list[str], query: str) -> AxiomResult: ...
    def rank_interventions(self, state, goal) -> list[dict]: ...
```

`arf.get_engine()` resolves the active backend:

- By default it returns `ReferenceAxiomEngine`, the open implementation shipped
  in this repository. The agent runs with no external services or credentials.
- A deployment can register a different backend with `register_engine()`, or
  have one auto-loaded via `AXIOM_ARF_ENGINE` and an installed engine package.

Because every caller targets the protocol, swapping the backend requires no
changes to agent or tool code. This is a standard dependency-inversion
boundary: the reasoning engine is a plugin, and the rest of the system is
agnostic to which one is loaded.

## The compute boundary

Heavy simulations follow the same pattern through a separate interface:

```python
class ComputeBackend(Protocol):
    def run_simulation(self, domain, spec, steps, temperature_k) -> SimulationResult: ...
```

`compute.get_backend()` resolves the active backend from `AXIOM_COMPUTE_BACKEND`:

- `vertex` - `VertexComputeBackend` submits the job to **Google Cloud Vertex AI
  as a Custom Job** on GPU workers, polls to completion, and returns the result.
  This is the production path.
- `local` - `LocalBackend` runs an in-process approximation for review, so the
  agent can be inspected without cloud credentials.

The `run_simulation` tool routes by job size: when a job exceeds the heavy
threshold it runs on Google Cloud GPU compute (Vertex AI). The simulation engine
is a container referenced by image URI - a swappable artifact, dispatched by
domain, that the agent orchestrates but does not embed.
