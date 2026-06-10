from axiom_agent import AxiomAgent
from axiom_agent.arf import get_engine
from axiom_agent.arf.fallback import ReferenceAxiomEngine


def test_offline_agent_runs_without_credentials(monkeypatch):
    # Force the offline path by ensuring no client is built.
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    agent = AxiomAgent()
    agent._client = None  # belt-and-suspenders for environments with libs present
    result = agent.solve("compute kinetic energy", axioms=["mass is positive"])
    assert result.answer
    assert any(s.kind == "tool" for s in result.steps)
    assert any(s.kind == "arf" for s in result.steps)
    assert result.steps[-1].kind == "answer"


def test_reference_engine_is_default_in_public_repo():
    engine = get_engine()
    assert isinstance(engine, ReferenceAxiomEngine)


def test_arf_derive_contract():
    engine = ReferenceAxiomEngine()
    r = engine.derive(["energy is conserved", "the ball has mass"], "energy of the ball")
    assert 0.0 <= r.confidence <= 1.0
    assert r.claim == "energy of the ball"
    assert "nodes" in r.reasoning_graph
