"""Shell command classifier tests."""

from __future__ import annotations

import pytest

from agent_hotwash.primitives.commands import (
    classify_command,
    command_intents,
    exit1_is_signal_free,
    is_compound,
    segment_head,
    segment_intent,
    split_segments,
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
    # modern runners
    ("bun run check", "build_test"),
    ("bun test", "build_test"),
    ("bunx vitest run src/x.test.ts", "build_test"),
    ("bun install", "other"),
    ("python -m pytest -q tests", "build_test"),
    ("python3 -m unittest discover -s tests", "build_test"),
    ("python -m http.server", "other"),
    ("ruff check .", "build_test"),
    ("ruff format --check src", "build_test"),
    ("ty check src", "build_test"),
    ("mypy src", "build_test"),
    ("cargo test", "build_test"),
    ("cargo clippy", "build_test"),
    ("cargo run", "other"),
    ("go test ./...", "build_test"),
    ("go vet ./...", "build_test"),
    ("go run main.go", "other"),
    ("browser-pilot eval flows/login.toml", "build_test"),
    ("browser-pilot snapshot", "other"),
    ("uv run python -m pytest tests", "build_test"),
    ("uv run ruff check .", "build_test"),
    # newline-separated commands are segments too
    ("cat a.py\npytest", "build_test"),
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
    assert is_compound("a\nb")
    assert not is_compound("just one")


def test_split_segments_on_newlines() -> None:
    assert split_segments("sed -n '1,9p' a.py\nprintf x\nrg -n -i foo src") == [
        "sed -n '1,9p' a.py",
        "printf x",
        "rg -n -i foo src",
    ]
    assert segment_head("cd x && FOO=1 sudo sed -i s/a/b/ f") is not None


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("rg needle src", True),
        ("grep -r x .", True),
        ("diff -u a b", True),
        ("cmp a b", True),
        ("test -f x", True),
        ("[ -d x ]", True),
        ("git diff --no-index --check a b", True),
        ("git grep foo", True),
        ("cat f && rg x f", True),  # tail segment decides
        ("git commit -m x", False),
        ("pytest", False),
        ("rg x f && pytest", False),
        ("", False),
    ],
)
def test_exit1_is_signal_free(cmd: str, expected: bool) -> None:
    assert exit1_is_signal_free(cmd) is expected
