"""Request provenance must identify the model on live, replay, and failure paths."""

import pytest

from agent_hotwash.memory.extraction import ProviderCache


@pytest.mark.parametrize("effort", [None, "high"])
def test_model_provenance_survives_cached_replay(tmp_path, effort):
    request = {"model": "test-generation-model", "input": "synthetic evidence"}
    if effort is not None:
        request["reasoning"] = {"effort": effort}
    live = ProviderCache(tmp_path, allow_paid=True, transport=lambda *_: {})
    live.call("openai", request)
    replay = ProviderCache(tmp_path)
    replay.call("openai", request)
    assert not live.calls[0]["cache_hit"]
    assert replay.calls[0]["cache_hit"]
    for call in [*live.calls, *replay.calls]:
        assert call["model"] == request["model"]
        assert call["reasoning_effort"] == effort


def test_failed_request_preserves_model_provenance(tmp_path):
    def fail(*_):
        raise OSError("synthetic transport failure")

    cache = ProviderCache(tmp_path, allow_paid=True, transport=fail)
    with pytest.raises(RuntimeError, match="synthetic transport failure"):
        cache.call("openai", {"model": "test-generation-model", "reasoning": {"effort": "high"}})
    assert cache.calls[0]["model"] == "test-generation-model"
    assert cache.calls[0]["reasoning_effort"] == "high"
