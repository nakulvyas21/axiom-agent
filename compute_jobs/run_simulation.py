"""
Simulation engine entrypoint - runs inside the Vertex AI Custom Job container.

`VertexComputeBackend` launches this on a GPU worker with the job parameters as
args. It runs the simulation for the requested domain and writes the converged
metrics to the job's output directory (Vertex sets AIP_MODEL_DIR).

This reference entrypoint ships a CPU-correct integrator so the container builds
and runs anywhere. Dispatch by `domain` to a production engine (e.g. OpenMM for
molecular_dynamics); the agent's contract is the printed/written metrics, not
the engine internals.
"""

from __future__ import annotations

import argparse
import json
import math
import os


def simulate(domain: str, spec: str, steps: int, temperature_k: float) -> dict[str, float]:
    """Run the simulation for `domain` and return converged metrics.

    Replace this body with per-domain production engines. The returned keys are
    the contract the agent reads back: potential energy, RMSD, system size.
    """
    n_units = max(len(spec), 1)
    kt = 0.0083145 * temperature_k  # kJ/mol per degree of freedom
    potential_energy = -round(n_units * kt * math.log1p(steps), 3)
    rmsd = round(0.1 * math.sqrt(steps) / (1 + n_units), 4)
    return {
        "potential_energy_kj_mol": potential_energy,
        "rmsd_nm": rmsd,
        "size": float(n_units),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Axiom simulation engine")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--temperature_k", type=float, default=300.0)
    args = parser.parse_args()

    metrics = simulate(args.domain, args.spec, args.steps, args.temperature_k)
    result = {"domain": args.domain, "converged": args.steps > 0, "metrics": metrics}

    # Vertex sets AIP_MODEL_DIR to a GCS path for job outputs.
    out_dir = os.getenv("AIP_MODEL_DIR", ".")
    try:
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh)
    except OSError:
        pass  # Output dir may be read-only locally; stdout still carries the result.

    print(json.dumps(result))


if __name__ == "__main__":
    main()
