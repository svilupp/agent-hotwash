"""Arg-normalizer + edit-distance tests."""

from __future__ import annotations

from agent_hotwash.primitives.argnorm import edit_distance, norm_args


def test_norm_args_order_independent() -> None:
    a = norm_args({"b": 2, "a": 1})
    b = norm_args({"a": 1, "b": 2})
    assert a == b == "a=1;b=2"


def test_norm_args_whitespace_collapsed() -> None:
    assert norm_args({"cmd": "  ls   -la  "}) == "cmd=ls -la"


def test_norm_args_nested_and_empty() -> None:
    assert norm_args(None) == ""
    assert norm_args({}) == ""
    assert norm_args({"x": {"z": 1, "a": 2}, "y": [1, "b "]}) == "x={a=2,z=1};y=[1,b]"


def test_norm_args_idempotent() -> None:
    args = {"path": "src/x.py", "content": "a\n b"}
    once = norm_args(args)
    # normalizing the same dict again is stable
    assert norm_args(dict(args)) == once


def test_edit_distance_basic() -> None:
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3


def test_edit_distance_cap() -> None:
    assert edit_distance("a" * 200, "b" * 10, cap=5) == 5
    assert edit_distance("aaaaaaaa", "bbbbbbbb", cap=3) == 3
