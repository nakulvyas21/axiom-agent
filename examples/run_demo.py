"""
Illustrates the Axiom Agent API: registered domains and the design loop.

Shows how the pieces wire together. The certified design loop is powered by the
proprietary physics framework (see the README -> Availability); this example
exercises the public interfaces with the shipped review-mode backend.
"""

from axiom_agent import AxiomAgent
from axiom_agent.agent import _pretty
from axiom_agent.domains import list_domains
from axiom_agent.tools.simulation import run_simulation

PROBLEMS = [
    "A projectile is launched at 20 m/s at 45 degrees. How far does it travel?",
    "A 5 kg object accelerates at 2 m/s^2. What force acts on it?",
    "A 2 kg ball moves at 3 m/s. What is its kinetic energy?",
]


def main() -> None:
    print("Registered physics domains:")
    for d in list_domains():
        print(f"  - {d['domain']}: {d['description']}")

    agent = AxiomAgent()
    for problem in PROBLEMS:
        print("=" * 72)
        print("GOAL:", problem)
        result = agent.solve(problem, axioms=["energy is conserved", "mass is positive"])
        print(_pretty(result))

    # Heavy-compute path: a large simulation runs on Google Cloud GPU.
    print("=" * 72)
    print("SIMULATION: molecular_dynamics domain (heavy job)")
    sim = run_simulation(domain="molecular_dynamics", spec="ATCGGCTAGCTA", steps=200_000, temperature_k=310.0)
    print(f"  backend: {sim['backend']}")
    print(f"  recommendation: {sim.get('recommendation', '(none)')}")


if __name__ == "__main__":
    main()
