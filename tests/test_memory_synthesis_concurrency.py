"""Concurrent synthesis preserves repair ownership, cache identity, and retry accounting."""

import json
import threading
import time
from datetime import UTC, datetime

import httpx
import pytest

from agent_hotwash.config import Config, SemanticConfig
from agent_hotwash.memory import extraction as ex
from agent_hotwash.memory import synthesis
from agent_hotwash.memory.pipeline import _synthesize


def records():
    return [
        {
            "id": f"record-{index}",
            "session_id": f"session-{index}",
            "role": "user",
            "kind": "message",
            "authorship": "human",
            "text": f"No, preserve schema {index}. " + "x" * 65000,
            "paths": [],
        }
        for index in range(6)
    ]


def proposal(row):
    return {
        "category": "user_corrections",
        "statement": "Preserve " + row["id"],
        "applicability": "The existing schema",
        "evidence_ids": [row["id"]],
        "paths": [],
        "intervention": "instruction",
        "draft": "Preserve the schema.",
        "reason": "Explicit correction",
    }


def test_parallel_synthesis_matches_serial_with_worker_local_repair_and_replay(tmp_path):
    source = records()
    features = [{"record_id": row["id"], "retained": True} for row in source]
    lock = threading.Lock()
    active, high_water = 0, 0

    def transport(provider, request):
        nonlocal active, high_water
        assert provider == "openai"
        with lock:
            active += 1
            high_water = max(high_water, active)
        try:
            time.sleep(0.01)
            data = json.loads(request["input"])
            repaired = "repair" in data
            if repaired:
                data = data["evidence"]
            candidates = data.get("candidates")
            if candidates is None:
                candidates = [proposal(row) for row in data["records"]]
                if candidates[0]["evidence_ids"] == ["record-0"] and not repaired:
                    candidates[0]["paths"] = ["unsupported.py"]
            return {"output": [{"content": [{"type": "output_text", "text": json.dumps({"candidates": candidates})}]}]}
        finally:
            with lock:
                active -= 1

    def cache(name, concurrency):
        return ex.ProviderCache(
            tmp_path / name,
            allow_paid=True,
            transport=transport,
            config=Config(semantic=SemanticConfig(max_concurrency=concurrency)),
        )

    serial = cache("serial", 1)
    expected = _synthesize({"records": source}, features, serial, serial.config.memory.model)
    assert high_water == 1
    parallel = cache("parallel", 3)
    assert _synthesize({"records": source}, features, parallel, parallel.config.memory.model) == expected
    assert 1 < high_water <= 3
    assert len(parallel.calls) == 8
    assert parallel.validation_events == serial.validation_events
    assert len(parallel.validation_events) == 1
    assert [call["request_sha256"] for call in parallel.calls] == [call["request_sha256"] for call in serial.calls]
    replay = ex.ProviderCache(tmp_path / "parallel", config=parallel.config)
    assert replay.config is not None
    assert _synthesize({"records": source}, features, replay, replay.config.memory.model) == expected
    assert all(call["cache_hit"] for call in replay.calls)


def install_client(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(synthesis.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler)))


def test_generation_retries_share_cooldown_and_capture_only_safe_headers(monkeypatch):
    calls, now, waits = [], [0.0], []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    control = synthesis.GenerationControl(clock=lambda: now[0], sleep=sleep)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "4", "authorization": "secret"}, json={"error": "limit"})
        return httpx.Response(200, headers={"x-ratelimit-remaining-tokens": "1234"}, json={"output": []})

    install_client(monkeypatch, handler)
    result = synthesis.generate({"model": "test"}, "synthetic", control=control)
    assert waits == [4]
    assert len(calls) == 2
    assert result["_transport"]["retries"] == 1
    assert result["_transport"]["attempts"][1]["headers"] == {"x-ratelimit-remaining-tokens": "1234"}
    assert "authorization" not in result["_transport"]["attempts"][0]["headers"]
    control.defer(3)
    synthesis.generate({"model": "test"}, "synthetic", control=control)
    assert waits == [4, 3]
    assert synthesis._retry_delay("Sun, 18 Oct 2026 12:00:05 GMT", 1, now=datetime(2026, 10, 18, 12, tzinfo=UTC)) == 5


def test_generation_read_timeout_does_not_preempt_wall_clock_deadline(monkeypatch):
    timeouts = []
    requests = []
    original = httpx.Client

    def client_factory(*, timeout, **kwargs):
        timeouts.append(timeout)

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"output": []})

        return original(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            **kwargs,
        )

    monkeypatch.setattr(synthesis.httpx, "Client", client_factory)
    synthesis.generate({"model": "test"}, "synthetic", timeout_s=180, max_retries=0)

    assert timeouts[0].read == 180
    assert timeouts[0].connect == 30
    assert requests[0].headers["accept-encoding"] == "identity"


@pytest.mark.parametrize("status,retries,attempts", [(400, 3, 1), (429, 2, 3), (503, 2, 3)])
def test_terminal_http_error_retains_status_attempts_without_cache(monkeypatch, tmp_path, status, retries, attempts):
    sent, now = [], [0.0]

    def handler(request):
        sent.append(request)
        return httpx.Response(status, headers={"x-request-id": "synthetic"}, json={"error": "synthetic"})

    install_client(monkeypatch, handler)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    control = synthesis.GenerationControl(
        clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    cache = ex.ProviderCache(
        tmp_path / "calls", allow_paid=True, config=Config(semantic=SemanticConfig(max_retries=retries))
    )
    cache.generation_control = control
    with pytest.raises(RuntimeError, match=r"HTTPError|HTTPStatusError"):
        cache.call("openai", {"model": "test"})
    assert len(sent) == attempts
    assert len(cache.calls) == 1
    assert cache.calls[0]["http_status"] == status
    assert len(cache.calls[0]["transport_metadata"]["attempts"]) == attempts
    assert not list((tmp_path / "calls").glob("*.json"))


def test_quoted_source_path_falls_back_to_string_schema_but_exact_validation_remains():
    source = records()[0]
    source.update(role="assistant", kind="reasoning", paths=['src/a"b.py'])
    request = ex.openai_request("test", {"records": [source]})
    items = request["text"]["format"]["schema"]["properties"]["candidates"]["items"]["properties"]["paths"]["items"]
    assert items == {"type": "string"}
    candidate = proposal(source)
    candidate.update(category="repeated_discovery", paths=source["paths"])

    def response():
        return {"output": [{"content": [{"type": "output_text", "text": json.dumps({"candidates": [candidate]})}]}]}

    assert ex.decode_candidates(response(), [source], require_recurrence=False)[0]["paths"] == source["paths"]
    candidate["paths"] = ["unknown.py"]
    with pytest.raises(ValueError, match="uncited path"):
        ex.decode_candidates(response(), [source], require_recurrence=False)
    source["paths"] = ["src/normal.py"]
    normal = ex.openai_request("test", {"records": [source]})
    items = normal["text"]["format"]["schema"]["properties"]["candidates"]["items"]["properties"]["paths"]["items"]
    assert items["enum"] == source["paths"]


def test_retry_attempt_accounting_and_successful_cache_replay(monkeypatch, tmp_path):
    sent, now = [], [0.0]

    def handler(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"error": "limit"})
        return httpx.Response(200, json={"output": []})

    install_client(monkeypatch, handler)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    cache = ex.ProviderCache(tmp_path / "calls", allow_paid=True, config=Config())
    cache.generation_control = synthesis.GenerationControl(
        clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    request = {"model": "test"}
    response = cache.call("openai", request)
    assert ex.openai_usage(cache.calls) == {"requests": 2, "retries": 1, "logical_requests": 1}
    replay = ex.ProviderCache(tmp_path / "calls", config=cache.config)
    assert replay.call("openai", request) == response
    assert len(sent) == 2
    assert ex.openai_usage(replay.calls) == {"requests": 0, "retries": 0, "logical_requests": 0}


def test_parallel_workers_share_one_generation_cooldown(tmp_path):
    cache = ex.ProviderCache(tmp_path, config=Config(semantic=SemanticConfig(max_concurrency=3)))
    barrier = threading.Barrier(3)

    def operation(worker, item):
        barrier.wait(timeout=3)
        return id(worker.generation_control)

    controls = list(ex.cached_operations(range(3), cache, operation))
    assert controls == [id(cache.generation_control)] * 3
