# Axiom Agent

> **Democratizing scientific computing.** Serious research shouldn't require a
> university supercomputer. Axiom Agent is an **autonomous physics design
> engine**: give it a goal, and it proposes a design, validates it against
> physics, and iterates until the design is certified - dispatching heavy
> validations to elastic cloud GPUs, no owned HPC cluster required.
>
> Any physics domain plugs in. Built on the **Google Agent Development Kit
> (ADK)**, **Gemini** (Vertex AI), and **Vertex AI Custom Jobs** for on-demand
> GPU compute.

---

## Mission

High-end scientific computing is gated by infrastructure: the labs that can run
large simulations are the ones that already own HPC clusters. That concentrates
research capability in a handful of well-funded institutions.

Axiom Agent removes the cluster from the critical path. The reasoning runs in an
agent; the heavy compute runs on **elastic cloud GPUs you rent by the job**, not
hardware you own. A graduate student, an independent researcher, or a startup
gets the same simulation capability as a national lab - on demand, per job.

---

## What it does

Give the agent a goal and (optionally) a set of axioms. It runs a design loop:

1. **Propose** - the ADK `LlmAgent` (Gemini) identifies the governing principles
   and proposes a candidate design.
2. **Validate** - it checks the design with the domain's tools: closed-form
   physics for exact quantities, and `run_simulation` for designs that need
   simulation. Heavy jobs run as **GPU jobs on Google Cloud** (Vertex AI Custom
   Jobs).
3. **Iterate** - it reads the violations, revises the design, and re-validates
   until the design passes.
4. **Certify** - it grounds the certified design against the stated axioms
   through the **ARF interface**.

Physics domains are pluggable (`axiom_agent/domains/`): the architecture defines
how a domain registers its validation tools, with `classical_mechanics` and
`molecular_dynamics` shown as illustrative interfaces.

The agent runs on **Google Cloud (Vertex AI)**: Gemini drives the ADK agent, and
Vertex AI Custom Jobs run heavy simulations on GPU workers. The certified design
loop is powered by the proprietary physics framework (see **Availability**).

---

## Architecture

```
  AGENT LAYER          design loop: propose → validate → iterate → certify
  ┌─────────────────────────────────────────────────────────────┐
  │            AxiomAgent   (ADK LlmAgent + Runner)              │
  └───────┬──────────────────┬───────────────────┬──────────────┘
          │                  │                   │
  ┌───────▼───────┐  ┌───────▼────────┐  ┌───────▼───────────────┐
  │ reasoning →   │  │ physics domains│  │ run_simulation        │
  │ ARF (plugin)  │  │ (registry)     │  │  → ComputeBackend     │
  └───────┬───────┘  └────────────────┘  └───────┬───────────────┘
          │                                       │
══════════╪═══════════ GOOGLE CLOUD · VERTEX AI ══╪═══════════════════
          │                                       │
  ┌───────▼───────────┐               ┌───────────▼────────────────┐
  │  Gemini           │               │  Vertex AI Custom Job       │
  │  (model serving)  │               │  GPU workers (A100)    ←┐   │
  │  drives the ADK   │               │  run the simulation     │   │
  │  LlmAgent         │               │  engine                 │   │
  └───────────────────┘               └─────────────────────────┼──┘
                                       Artifact Registry ───────┘
                                       (simulation-engine container)
```

The agent runs on **Google Cloud / Vertex AI**: Gemini serves the ADK
`LlmAgent`, and heavy simulations run as **Vertex AI Custom Jobs on GPU workers
(A100)**, from a container in Artifact Registry. Three pluggable boundaries keep
the agent code stable while the platform scales:

- **Domains** (`domains/`) - any physics domain registers its validation tools;
  the repo ships `classical_mechanics` and `molecular_dynamics` as examples.
- **`AxiomEngine`** (`arf/`) - the reasoning layer, behind a stable interface so
  the reasoning core evolves independently of the agent.
- **`ComputeBackend`** (`compute/`) - heavy simulations run as Vertex AI Custom
  Jobs on GPU workers. The agent dispatches automatically once a job exceeds the
  heavy threshold.

The agent depends only on these boundaries, so domains, the reasoning engine,
and the compute backend each evolve without changing any agent code.

---

## Availability

This repository is the **public architecture** of Axiom Agent: the agent
orchestration (ADK), the domain-registry and tool interfaces, the compute
boundary (Vertex AI GPU jobs), and the ARF reasoning interface.

Running the engine end to end requires the **proprietary physics framework** -
the validated, patent-pending physics and reasoning core that powers the design
loop. That component is **not included here** and is licensed separately.

> **To request access to the proprietary physics framework, contact the author
> at [nvyas@heysuvi.com](mailto:nvyas@heysuvi.com).**

What you can see in this repo is the full system architecture and every
integration boundary; what you cannot run without the licensed core is the
certified physics design itself.

---

## Project layout

```
axiom_agent/
├── agent.py            # ADK design loop: propose → validate → iterate → certify
├── __main__.py         # CLI entry point
├── domains/            # physics domain registry (register_domain / list_domains)
├── tools/
│   ├── physics.py      # unit-tested physics calculations
│   ├── simulation.py   # run_simulation (local or GPU, any domain)
│   └── __init__.py     # tool registry + function-call schemas
├── compute/
│   ├── backend.py      # ComputeBackend protocol + backend resolution
│   ├── local.py        # LocalBackend - in-process review mode
│   └── vertex.py       # VertexComputeBackend - Vertex AI Custom Job (GPU)
└── arf/
    ├── interface.py    # AxiomEngine protocol + engine resolution
    └── fallback.py     # ReferenceAxiomEngine - default backend
compute_jobs/           # simulation-engine container (run_simulation.py, Dockerfile, deploy.sh)
examples/run_demo.py    # runnable demo (domains + design loop)
tests/                  # pytest suite (tools + agent + ARF + compute + domains)
```

## Configuration

| Variable                  | Purpose                                                       |
|---------------------------|---------------------------------------------------------------|
| `GOOGLE_CLOUD_PROJECT`    | Use Vertex AI (no API key). Preferred.                        |
| `GOOGLE_API_KEY`          | Use Gemini API-key mode instead.                             |
| `AXIOM_MODEL`             | Gemini model id (default `gemini-2.5-pro`).                  |
| `AXIOM_ARF_ENGINE`        | Name of an alternative ARF backend to load, if registered.   |
| `AXIOM_COMPUTE_BACKEND`   | `local` (review mode) or `vertex` (Vertex AI GPU compute).   |
| `AXIOM_SIM_IMAGE_URI`     | Simulation-engine container image on Vertex AI.             |

## License

Apache-2.0 for the architecture in this repository. See [LICENSE](LICENSE). The
proprietary physics framework is licensed separately (see **Availability**).

## Team

- **Nakul Vyas**, Heysuvi Labs, LLC - [heysuvi.com](https://heysuvi.com) - [nvyas@heysuvi.com](mailto:nvyas@heysuvi.com)
- **Dr. Iliya Stoev** - Scientific Advisor - [istoev@heysuvi.com](mailto:istoev@heysuvi.com)
