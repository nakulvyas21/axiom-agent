"""CLI entry point:  python -m axiom_agent "a physics question" """

from __future__ import annotations

import sys

from .agent import AxiomAgent, _pretty


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    problem = " ".join(argv) or "A 2 kg ball moves at 3 m/s. What is its kinetic energy?"
    agent = AxiomAgent()
    result = agent.solve(problem, axioms=["energy is conserved", "mass is positive"])
    print(_pretty(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
