import axiom_agent  # noqa: F401  (triggers domain registration)
from axiom_agent.domains import (
    DOMAIN_REGISTRY,
    get_tools_for_domain,
    list_domains,
    register_domain,
)


def test_builtin_domains_registered():
    names = {d["domain"] for d in list_domains()}
    assert "classical_mechanics" in names
    assert "molecular_dynamics" in names


def test_get_tools_for_domain_returns_callables():
    tools = get_tools_for_domain("classical_mechanics")
    assert tools and all(callable(t) for t in tools)


def test_unknown_domain_raises():
    try:
        get_tools_for_domain("nonexistent_domain")
    except ValueError as e:
        assert "nonexistent_domain" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown domain")


def test_register_custom_domain():
    def my_check(x: float) -> float:
        """A custom validation tool."""
        return x

    register_domain("custom_test", "A custom domain.", tools=[my_check])
    assert "custom_test" in DOMAIN_REGISTRY
    assert get_tools_for_domain("custom_test") == [my_check]
