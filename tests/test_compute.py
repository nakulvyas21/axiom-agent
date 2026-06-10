from axiom_agent.compute import get_backend, register_backend
from axiom_agent.compute.local import LocalBackend
from axiom_agent.compute.backend import is_heavy, HEAVY_STEP_THRESHOLD
from axiom_agent.tools.simulation import run_simulation


def test_local_is_default_backend(monkeypatch):
    monkeypatch.delenv("AXIOM_COMPUTE_BACKEND", raising=False)
    # Reset the module-level singleton so the env change takes effect.
    import axiom_agent.compute.backend as b
    b._BACKEND = None
    assert isinstance(get_backend(), LocalBackend)


def test_local_simulation_runs_in_process():
    backend = LocalBackend()
    result = backend.run_simulation(
        domain="molecular_dynamics", spec="ATCG", steps=1000, temperature_k=300.0
    )
    assert result.converged
    assert result.backend == "local"
    assert "potential_energy_kj_mol" in result.metrics


def test_heavy_threshold():
    assert not is_heavy(HEAVY_STEP_THRESHOLD - 1)
    assert is_heavy(HEAVY_STEP_THRESHOLD)


def test_tool_recommends_gpu_for_heavy_local_jobs(monkeypatch):
    monkeypatch.delenv("AXIOM_COMPUTE_BACKEND", raising=False)
    register_backend(LocalBackend())
    payload = run_simulation(
        domain="molecular_dynamics", spec="ATCG", steps=HEAVY_STEP_THRESHOLD, temperature_k=310.0
    )
    assert payload["domain"] == "molecular_dynamics"
    assert payload["backend"] == "local"
    assert "recommendation" in payload
    assert "vertex" in payload["recommendation"].lower()


def test_tool_no_recommendation_for_light_jobs():
    register_backend(LocalBackend())
    payload = run_simulation(domain="molecular_dynamics", spec="ATCG", steps=500)
    assert "recommendation" not in payload
