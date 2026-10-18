"""Provider cache identity must survive escaped credentials in JSON evidence."""

import json

import pytest

from agent_hotwash.memory.extraction import ProviderCache, _normalize_provider_request


def test_serialized_evidence_redacts_before_hash_send_and_cache(tmp_path):
    secret = "unit-test-credential"
    evidence = {"records": [{"text": f'Use OPENAI_API_KEY="{secret}"; preserve "quoted prose" and \\paths.'}]}
    request = {"model": "test", "input": json.dumps(evidence)}
    original = json.loads(json.dumps(request))
    observed = []

    def transport(provider, sent):
        observed.append((provider, sent))
        assert secret not in sent["input"]
        assert json.loads(sent["input"])["records"][0]["text"].endswith('preserve "quoted prose" and \\paths.')
        return {
            "output": [
                {"content": [{"type": "output_text", "text": json.dumps({"draft": f'KEY: API_KEY="{secret}"'})}]}
            ]
        }

    first = ProviderCache(tmp_path, allow_paid=True, transport=transport).call("openai", request)
    saved_path = next(tmp_path.glob("*.json"))
    saved = json.loads(saved_path.read_text())
    assert saved["request"] == observed[0][1]
    assert saved["response"] == first
    assert json.loads(first["output"][0]["content"][0]["text"])["draft"] == 'KEY: API_KEY="[REDACTED]"'
    assert secret not in saved_path.read_text()
    assert _normalize_provider_request("openai", saved["request"]) == saved["request"]
    assert ProviderCache(tmp_path).call("openai", request) == first
    assert len(observed) == 1
    assert request == original  # normalization never mutates the caller's evidence


def test_already_redacted_escaped_assignment_remains_valid_nested_json(tmp_path):
    request = {"input": json.dumps({"text": 'OPENAI_API_KEY="[REDACTED]"'})}
    ProviderCache(tmp_path, allow_paid=True, transport=lambda *_: {}).call("openai", request)
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert json.loads(saved["request"]["input"]) == {"text": 'OPENAI_API_KEY="[REDACTED]"'}
    assert ProviderCache(tmp_path).call("openai", request) == {}


def test_cache_mismatch_still_blocks_without_retry(tmp_path):
    request = {"input": json.dumps({"text": 'API_KEY="[REDACTED]"'})}
    cache = ProviderCache(tmp_path, allow_paid=True, transport=lambda *_: {})
    cache.call("openai", request)
    path = next(tmp_path.glob("*.json"))
    saved = json.loads(path.read_text())
    saved["request"]["input"] = "corrupted serialized evidence"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="cache request mismatch"):
        ProviderCache(tmp_path, allow_paid=True, transport=lambda *_: pytest.fail("must not retry")).call(
            "openai", request
        )


def test_plain_nonjson_provider_input_remains_supported(tmp_path):
    request = {"input": 'Keep "ordinary prose" with API_KEY="sample-secret".'}
    expected = 'Keep "ordinary prose" with API_KEY="[REDACTED]".'
    ProviderCache(tmp_path, allow_paid=True, transport=lambda _, sent: {"echo": sent["input"]}).call("openai", request)
    assert ProviderCache(tmp_path).call("openai", request) == {"echo": expected}


def test_configured_secrets_are_removed_from_preparation_providers_and_reports(tmp_path):
    from agent_hotwash.config import load_config
    from agent_hotwash.memory.pipeline import run_audit

    tokens = ["AKIA" + "A" * 16, "company-credential-123"]
    config = load_config()
    config = config.model_copy(
        update={"lexicons": config.lexicons.model_copy(update={"secret": [*config.lexicons.secret, tokens[1]]})}
    )
    source = tmp_path / "rollout-secret.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"type": "session_meta", "payload": {"id": "redaction", "cwd": "/repo"}},
                {
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": "No, never print " + " ".join(tokens)},
                },
            ]
        )
        + "\n"
    )
    sent = []

    def transport(provider, request):
        sent.append((provider, request))
        if provider == "jev":
            return {"answers": {key: {"noul": 0.99} for key in request["questions"]}}
        return {"output": [{"content": [{"type": "output_text", "text": '{"candidates":[]}'}]}]}

    output = tmp_path / "audit"
    cache = ProviderCache(output / "calls", allow_paid=True, transport=transport, config=config)
    assert run_audit([source], output, config=config, extract=True, cache=cache)["status"] == "complete"
    assert {provider for provider, _ in sent} == {"jev", "openai"}
    for token in tokens:
        assert token not in json.dumps(sent)
        assert all(token not in path.read_text() for path in output.rglob("*") if path.is_file())
    prepared = json.loads((output / "prepared.json").read_text())
    assert all(row["retained_chars"] == len(row["text"]) for row in prepared["records"])


def test_stricter_redaction_policy_cannot_reuse_unsanitized_response(tmp_path):
    from agent_hotwash.config import Config, LexiconConfig

    token = "company-credential-123"
    request = {"input": "public evidence"}
    ProviderCache(tmp_path, allow_paid=True, transport=lambda *_: {"echo": token}).call("openai", request)
    stricter = Config(lexicons=LexiconConfig(secret=[token]))
    with pytest.raises(RuntimeError, match="cache miss"):
        ProviderCache(tmp_path, config=stricter).call("openai", request)
    response = ProviderCache(tmp_path, config=stricter, allow_paid=True, transport=lambda *_: {"echo": token}).call(
        "openai", request
    )
    assert response == {"echo": "[REDACTED]"}
    assert ProviderCache(tmp_path, config=stricter).call("openai", request) == response


def test_repair_references_the_actual_cached_request(tmp_path):
    from agent_hotwash.memory.extraction import propose_candidates

    requests = []

    def transport(provider, request):
        requests.append(request)
        output = {"candidates": [{"invalid": True}]} if len(requests) == 1 else {"candidates": []}
        return {"output": [{"content": [{"type": "output_text", "text": json.dumps(output)}]}]}

    cache = ProviderCache(tmp_path, allow_paid=True, transport=transport)
    assert propose_candidates(cache, "test-model", {"records": []}, []) == []
    reference = cache.validation_events[0]["request_sha256"]
    saved = json.loads((tmp_path / (reference + ".json")).read_text())
    assert saved["request"] == requests[0]
    assert reference == cache.calls[0]["request_sha256"]
    assert json.loads(requests[1]["input"])["repair"]["previous_request_sha256"] == reference
