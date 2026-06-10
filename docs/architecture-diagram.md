# Axiom Agent - Architecture Diagrams

GitHub renders these Mermaid diagrams inline. Link to this page from anywhere:
`docs/architecture-diagram.md`.

---

## 1. The design loop

How the agent turns a goal into a physics-certified design.

```mermaid
flowchart LR
    Goal([Goal]) --> Propose[Propose<br/>candidate design]
    Propose --> Validate{Validate<br/>against physics}
    Validate -- violations --> Revise[Revise design]
    Revise --> Validate
    Validate -- passes --> Certify[Certify &<br/>ground in axioms]
    Certify --> Out([Certified design])

    classDef loop fill:#e8f0fe,stroke:#4285f4,color:#1a1a1a;
    classDef done fill:#e6f4ea,stroke:#34a853,color:#1a1a1a;
    class Propose,Validate,Revise loop;
    class Certify,Out done;
```

---

## 2. System architecture

The agent runs on Google Cloud / Vertex AI. Three pluggable boundaries
(domains, reasoning, compute) keep the agent code stable while the platform
scales.

```mermaid
flowchart TB
    subgraph AGENT["Agent layer"]
        A[AxiomAgent<br/>ADK LlmAgent + Runner]
        D["Domains registry<br/>classical_mechanics · molecular_dynamics · …"]
        T["Validation tools<br/>closed-form physics + run_simulation"]
        R["ARF interface<br/>(reasoning, pluggable)"]
        A --> D
        A --> T
        A --> R
    end

    subgraph GCP["Google Cloud · Vertex AI"]
        G[Gemini<br/>model serving]
        V["Vertex AI Custom Job<br/>GPU workers (A100)"]
        AR[(Artifact Registry<br/>simulation-engine container)]
        V --- AR
    end

    subgraph IP["Proprietary physics framework (licensed separately)"]
        P["Validated physics + reasoning core<br/>powers the certified design loop"]
    end

    A -. drives .-> G
    T -- heavy jobs --> V
    R -. backed by .-> P

    classDef agent fill:#e8f0fe,stroke:#4285f4,color:#1a1a1a;
    classDef gcp fill:#fef7e0,stroke:#fbbc04,color:#1a1a1a;
    classDef ip fill:#fce8e6,stroke:#ea4335,color:#1a1a1a;
    class A,D,T,R agent;
    class G,V,AR gcp;
    class P ip;
```

---

## 3. Pluggable boundaries

Each boundary is an interface; the implementation behind it can be swapped
without touching agent code.

| Boundary | Interface | Ships in this repo | Production |
|---|---|---|---|
| Reasoning | `AxiomEngine` (`arf/`) | `ReferenceAxiomEngine` | Proprietary physics framework |
| Compute | `ComputeBackend` (`compute/`) | `LocalBackend` (review) | `VertexComputeBackend` (Vertex AI GPU) |
| Domains | `register_domain` (`domains/`) | `classical_mechanics`, `molecular_dynamics` | Any physics domain |
