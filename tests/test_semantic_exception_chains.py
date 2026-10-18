"""Provider exception chains may cycle; semantic error handling must terminate."""

from __future__ import annotations

import httpx2
import pytest
from systemoneprompts.cache import CacheMissError

from agent_hotwash.semantic.client import StrictSemanticResponseError, SystemOneAsker


def run_raised(tmp_path, monkeypatch, error):
    """Inject the SDK failure before transport, without any provider requests."""

    def no_provider(request):
        pytest.fail("Exception-chain test must not reach provider transport")

    asker = SystemOneAsker("jev-1.13.0", tmp_path, mode="cached", transport_override=httpx2.MockTransport(no_provider))

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(asker._client, "system_one_sync", fail)
    try:
        asker.ask(
            {},
            {
                "task.test.evidence": {
                    "type": "noul",
                    "instructions": {"question": "Is evidence present?"},
                    "criteria": {},
                }
            },
        )
    finally:
        assert asker.stats()["requests"] == 0
        asker.close()


@pytest.mark.parametrize("edge", ["__cause__", "__context__"])
def test_cyclic_sdk_exception_chain_preserves_original_error(tmp_path, monkeypatch, edge):
    outer = ValueError("SDK wrapper")
    inner = RuntimeError("SDK underlying failure")
    setattr(outer, edge, inner)
    setattr(inner, edge, outer)
    with pytest.raises(ValueError) as raised:
        run_raised(tmp_path, monkeypatch, outer)
    assert raised.value is outer


@pytest.mark.parametrize("edge", ["__cause__", "__context__"])
def test_cyclic_cache_miss_chain_still_classifies_as_cache_miss(tmp_path, monkeypatch, edge):
    outer = ValueError("SDK wrapper")
    miss = CacheMissError(["task.test.evidence"])
    setattr(outer, edge, miss)
    setattr(miss, edge, outer)
    with pytest.raises(CacheMissError) as raised:
        run_raised(tmp_path, monkeypatch, outer)
    assert raised.value is miss


@pytest.mark.parametrize("cyclic", [False, True])
def test_strict_response_error_survives_wrapped_exception_chain(tmp_path, monkeypatch, cyclic):
    outer = RuntimeError("SDK wrapper")
    strict = StrictSemanticResponseError("Malformed response")
    outer.__cause__ = strict
    if cyclic:
        strict.__context__ = outer
    with pytest.raises(StrictSemanticResponseError) as raised:
        run_raised(tmp_path, monkeypatch, outer)
    assert raised.value is strict
