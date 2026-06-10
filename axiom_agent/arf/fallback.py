"""
Reference implementation of the ARF `AxiomEngine`.

`ReferenceAxiomEngine` is the default backend shipped with the package. It is
deterministic and dependency-free, so the agent runs end-to-end without any
external services. It grades support for a claim by measuring how much of the
query is covered by the premises.
"""

from __future__ import annotations

from typing import Any

from .interface import AxiomEngine, AxiomResult


class ReferenceAxiomEngine(AxiomEngine):
    """Deterministic axiom engine based on premise/query term coverage."""

    def derive(self, premises: list[str], query: str) -> AxiomResult:
        # Score support as the fraction of query terms covered by the premises.
        q_tokens = _tokens(query)
        premise_tokens = set().union(*(_tokens(p) for p in premises)) if premises else set()
        overlap = q_tokens & premise_tokens
        coverage = len(overlap) / len(q_tokens) if q_tokens else 0.0

        supported = coverage >= 0.5
        graph = {
            "nodes": sorted(premise_tokens | q_tokens),
            "edges": [{"from": "premises", "to": "query", "weight": round(coverage, 3)}],
        }
        return AxiomResult(
            claim=query,
            supported=supported,
            confidence=round(coverage, 3),
            reasoning_graph=graph,
            rationale=f"Premise/query term coverage = {coverage:.2f}.",
        )

    def rank_interventions(
        self, state: dict[str, Any], goal: str
    ) -> list[dict[str, Any]]:
        # Enumerate one candidate intervention per state variable, in a
        # stable order. Backends with a model of the system can score these.
        ranked = [
            {
                "intervention": f"adjust:{key}",
                "predicted_effect": 0.0,
                "explanation": f"Candidate adjustment to '{key}' toward goal '{goal}'.",
            }
            for key in sorted(state.keys())
        ]
        return ranked


def _tokens(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2}
