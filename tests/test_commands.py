"""Shell command classifier tests."""

from __future__ import annotations

import pytest

from agent_hotwash.primitives.commands import (
    classify_command,
    command_intents,
    is_compound,
    segment_intent,
)

CASES = [
    ("ls -la", "inspect"),
    ("cat file.py", "inspect"),
    ("grep -r foo .", "inspect"),
    ("git status", "inspect"),
    ("git diff HEAD~1", "inspect"),
    ("rm -rf build", "mutate"),
    ("git commit -m 'x'", "mutate"),
    ("git checkout main", "mutate"),
    ("sed -i 's/a/b/' f.txt", "mutate"),
    ("sed 's/a/b/' f.txt", "inspect"),
    ("pytest tests/", "build_test"),
    ("uv run pytest", "build_test"),
    ("pnpm test", "build_test"),
    ("npm run build", "build_test"),
    ("npm install", "other"),
    ("./run.sh", "other"),
    # codex shell wrapper is stripped before classifying
    ("bash -lc 'pytest -q'", "build_test"),
    # compound: highest-priority intent wins (build_test > mutate > inspect)
    ("cat f && pytest", "build_test"),
    ("grep x f && rm y", "mutate"),
    ("cd src && ls", "inspect"),
    ("", "other"),
]


@pytest.mark.parametrize(("cmd", "expected"), CASES)
def test_classify_command(cmd: str, expected: str) -> None:
    assert classify_command(cmd) == expected


def test_segment_intent_skips_prefixes() -> None:
    assert segment_intent("sudo rm -rf x") == "mutate"
    assert segment_intent("FOO=bar pytest") == "build_test"
    assert segment_intent("cd /tmp") is None


def test_command_intents_and_compound() -> None:
    assert command_intents("cat f && pytest") == {"inspect", "build_test"}
    assert is_compound("a && b")
    assert not is_compound("just one")
