"""Learning test: code-bench PI stdout.jsonl format.

Target: runs/baseline-pi-*/<task>/<attempt>/stdout.jsonl

Verified facts:
- Very chatty. Line types: session, agent_start, agent_end, turn_start,
  turn_end, message_start, message_update, message_end,
  tool_execution_start, tool_execution_update, tool_execution_end, and
  (in some runs) auto_retry_start / auto_retry_end.
- session: version, id, timestamp (ISO), cwd.
- tool_execution_start: toolCallId, toolName, args.
- tool_execution_end: toolCallId, toolName, result, isError (TOP-LEVEL bool;
  result.isError is null — SURPRISE vs briefing which implied nested).
  result.content is a list of {type:"text", text:...}.
- message_end.message.usage: {input, output, cacheRead, cacheWrite, reasoning,
  totalTokens, cost{...}}. Epoch-ms `timestamp` on message.
- SURPRISE: the glm-5.2 "zero/null usage" caveat is FALSE for these runs.
  glm-5.2 pi runs emit real usage; only rare individual messages show 0.
"""

from __future__ import annotations

from ._helpers import codebench_stdout_files, read_jsonl, require


def _first_file(model_glob: str):
    files = codebench_stdout_files(model_glob, limit=1)
    require(files, f"code-bench pi ({model_glob})")
    return files[0]


def test_line_type_inventory():
    from ._helpers import codebench_stdout_files

    expected = {
        "session",
        "agent_start",
        "agent_end",
        "turn_start",
        "turn_end",
        "message_start",
        "message_update",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
        "auto_retry_start",
        "auto_retry_end",
    }
    # Scan across the available runs rather than only the first on disk: every run
    # must stay within the known type inventory, and at least one must exhibit the
    # chatty *_update stream (which not every short run emits — hence no order
    # dependency on whichever run sorts first).
    files = codebench_stdout_files("baseline-pi-opus48-*")
    require(files, "code-bench pi (baseline-pi-opus48-*)")
    saw_update = False
    for f in files:
        types = {r["type"] for r in read_jsonl(f)}
        assert types <= expected, f"unexpected pi types in {f}: {types - expected}"
        if "message_update" in types:
            saw_update = True
    if not saw_update:
        import pytest

        pytest.skip("no baseline-pi-opus48 run on disk contains message_update")


def test_session_record():
    records = read_jsonl(_first_file("baseline-pi-opus48-*"))
    session = next(r for r in records if r["type"] == "session")
    assert {"version", "id", "timestamp", "cwd"} <= set(session)
    assert session["timestamp"].endswith("Z"), "session timestamp is ISO-8601 Z"


def test_tool_execution_end_error_is_top_level():
    """isError lives at top level; result.isError is null."""
    records = read_jsonl(_first_file("baseline-pi-opus48-*"))
    ends = [r for r in records if r["type"] == "tool_execution_end"]
    assert ends
    sample = ends[0]
    assert "isError" in sample, "isError must be a top-level field"
    assert isinstance(sample["isError"], bool)
    # result carries content list; its own isError is null / absent.
    assert sample["result"].get("isError") is None


def test_opus_usage_shape():
    records = read_jsonl(_first_file("baseline-pi-opus48-*"))
    msgs = [r for r in records if r["type"] == "message_end" and r["message"].get("role") == "assistant"]
    assert msgs
    usage = msgs[0]["message"]["usage"]
    assert {"input", "output", "cacheRead", "cacheWrite", "totalTokens", "cost"} <= set(usage)
    assert "total" in usage["cost"]


def test_glm52_has_real_usage_not_zero():
    """SURPRISE: briefing said glm-5.2 emits zero/null usage. Reality: it doesn't."""
    files = codebench_stdout_files("baseline-pi-glm52-xhigh", limit=4)
    require(files, "code-bench pi glm-5.2")
    nonzero_totals = 0
    zero_totals = 0
    for f in files:
        for r in read_jsonl(f):
            if r["type"] == "message_end" and r["message"].get("role") == "assistant":
                total = r["message"].get("usage", {}).get("totalTokens")
                if total:
                    nonzero_totals += 1
                elif total in (0, None):
                    zero_totals += 1
    assert nonzero_totals > 0, "glm-5.2 should emit real usage"
    # Real usage vastly dominates; the caveat as stated is wrong.
    assert nonzero_totals > zero_totals * 5, (
        f"expected mostly-real usage, got nonzero={nonzero_totals} zero={zero_totals}"
    )
    print(f"\nPI glm-5.2 usage: nonzero={nonzero_totals} zero/null={zero_totals}")


def test_error_serialization_and_collect_strings():
    error_samples: list[str] = []
    for glob_ in ("baseline-pi-opus48-*", "baseline-pi-glm52-xhigh"):
        for f in codebench_stdout_files(glob_, limit=3):
            for r in read_jsonl(f):
                if r["type"] == "tool_execution_end" and r.get("isError"):
                    texts = [c.get("text", "") for c in r["result"].get("content", []) if isinstance(c, dict)]
                    joined = " ".join(texts)
                    if joined.strip():
                        error_samples.append(joined[:200])
    assert error_samples, "expected some pi tool errors"
    # Bash failures embed 'Command exited with code N' in the text.
    assert any("exited with code" in s for s in error_samples)
    print("\nPI real error strings:")
    for s in error_samples[:6]:
        print(" -", repr(s[:140]))
