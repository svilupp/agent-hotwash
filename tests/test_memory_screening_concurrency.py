"""Bounded screening concurrency preserves deterministic evidence and accounting."""

import threading
import time

import pytest

from agent_hotwash.config import Config, MemoryConfig, SemanticConfig
from agent_hotwash.memory import extraction as ex


def records(count=8):
    return [
        {
            "id": f"record-{index}",
            "session_id": f"session-{index}",
            "role": "user",
            "kind": "message",
            "authorship": "human",
            "text": f"No, preserve schema {index}. " + "x" * 30000,
            "paths": [],
        }
        for index in range(count)
    ]


def config(concurrency):
    return Config(memory=MemoryConfig(screening_max_questions=4), semantic=SemanticConfig(max_concurrency=concurrency))


def answer(request):
    return {
        "answers": {key: {"noul": 0.9} for key in request["questions"]},
        "native_usage": {"requests": 1, "questions_asked": len(request["questions"]), "cache_hits": 0},
    }


def test_parallel_screening_matches_serial_and_replays_in_request_order(tmp_path):
    source = records()
    lock = threading.Lock()
    active, high_water = 0, 0

    def transport(provider, request):
        nonlocal active, high_water
        with lock:
            active += 1
            high_water = max(high_water, active)
        try:
            time.sleep(0.01)
            return answer(request)
        finally:
            with lock:
                active -= 1

    serial = ex.ProviderCache(tmp_path / "serial", allow_paid=True, transport=transport, config=config(1))
    expected = ex.screen_features({"records": source}, serial)
    assert high_water == 1
    parallel = ex.ProviderCache(
        tmp_path / "parallel",
        allow_paid=True,
        transport=transport,
        config=config(3),
    )
    assert ex.screen_features({"records": source}, parallel) == expected
    assert 1 < high_water <= 3
    assert [call["request_sha256"] for call in parallel.calls] == [call["request_sha256"] for call in serial.calls]
    assert sum(call["native_usage"]["requests"] for call in parallel.calls) == len(parallel.calls)
    replay = ex.ProviderCache(tmp_path / "parallel", config=parallel.config)
    assert ex.screen_features({"records": source}, replay) == expected
    assert all(call["cache_hit"] for call in replay.calls)
    assert all(call["native_usage"]["requests"] == 0 for call in replay.calls)


def test_failure_cancels_queued_work_and_collects_completed_worker_ledgers(tmp_path):
    source = records()
    requests = list(ex.feature_requests(source, max_questions=4))
    first_request = requests[0][0]
    started = []
    barrier = threading.Barrier(3)

    def transport(provider, request):
        started.append(request)
        barrier.wait(timeout=3)
        if request == first_request:
            raise RuntimeError("synthetic provider failure")
        return answer(request)

    cache = ex.ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport, config=config(3))
    with pytest.raises(RuntimeError, match="request failed"):
        ex.screen_features({"records": source}, cache)
    assert len(started) == 3
    assert len(cache.calls) == 3
    assert sum("error_type" in call for call in cache.calls) == 1
    assert len(list((tmp_path / "calls").glob("*.json"))) == 2
    assert [call["request_sha256"] for call in cache.calls] == [
        cache.request_identity("jev", request) for request, _ in requests[:3]
    ]
    replay = ex.ProviderCache(tmp_path / "calls", config=cache.config)
    for request, _ in requests[1:3]:
        assert replay.call("jev", request)["answers"] == answer(request)["answers"]
    assert all(call["cache_hit"] for call in replay.calls)


def test_native_workers_share_limiter_and_close_isolated_askers(tmp_path, monkeypatch):
    import agent_hotwash.semantic.client as client

    askers = []
    barrier = threading.Barrier(3)

    class Asker:
        def __init__(self, model, directory, **kwargs):
            self.model = model
            self.limiter = kwargs["limiter"]
            self.requests = 0
            self.closed = False
            askers.append(self)

        def stats(self):
            return {"requests": self.requests, "cache_hits": 0}

        def usage(self):
            return []

        def ask(self, state, questions):
            if self.requests == 0:
                barrier.wait(timeout=3)
            self.requests += 1
            return {key: {"noul": 0.9} for key in questions}

        def close(self):
            self.closed = True

    monkeypatch.setattr(client, "SystemOneAsker", Asker)
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-test-key")
    cache = ex.ProviderCache(tmp_path / "calls", allow_paid=True, config=config(3))
    ex.screen_features({"records": records()}, cache)
    assert len(askers) == 3
    assert len({id(asker.limiter) for asker in askers}) == 1
    assert all(asker.closed for asker in askers)
    assert sum(call["native_usage"]["requests"] for call in cache.calls) == sum(asker.requests for asker in askers)
    assert all(call["native_usage"]["requests"] == 1 for call in cache.calls)


def test_identical_concurrent_requests_use_one_provider_call_for_distinct_records(tmp_path):
    source = records(6)
    for row in source:
        row["text"] = source[0]["text"]
    sent = []

    def transport(provider, request):
        sent.append(request)
        time.sleep(0.02)
        return answer(request)

    cache = ex.ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport, config=config(3))
    features = ex.screen_features({"records": source}, cache)
    assert len(sent) == 1
    assert len(cache.calls) == 6
    assert sum(not call["cache_hit"] for call in cache.calls) == 1
    assert sum(call["native_usage"]["requests"] for call in cache.calls) == 1
    assert {feature["record_id"] for feature in features} == {row["id"] for row in source}
    assert all(feature["retained"] for feature in features)


def test_identical_failed_requests_do_not_retry_or_deadlock_in_waiting_workers(tmp_path):
    source = records(6)
    for row in source:
        row["text"] = source[0]["text"]
    sent = []

    def transport(provider, request):
        sent.append(request)
        time.sleep(0.02)
        raise RuntimeError("synthetic failure")

    cache = ex.ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport, config=config(3))
    with pytest.raises(RuntimeError, match="request failed"):
        ex.screen_features({"records": source}, cache)
    assert len(sent) == 1
    assert len(cache.calls) == 1
    assert not list((tmp_path / "calls").glob("*.json"))
