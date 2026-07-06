"""Sliding-window counter tests."""

from __future__ import annotations

import pytest

from agent_hotwash.primitives.window import SlidingWindow


def _odd(n: int) -> bool:
    return n % 2 == 1


def test_max_count_basic() -> None:
    w = SlidingWindow[int](3)
    # windows of 3: [1,2,3]->2 odd, [2,3,4]->1, [3,4,5]->2
    assert w.max_count([1, 2, 3, 4, 5], _odd) == 2


def test_max_count_shorter_than_window() -> None:
    w = SlidingWindow[int](10)
    assert w.max_count([1, 3], _odd) == 2
    assert w.max_count([], _odd) == 0


def test_max_count_all_match() -> None:
    w = SlidingWindow[int](2)
    assert w.max_count([1, 3, 5, 7], _odd) == 2


def test_first_window_reaching() -> None:
    w = SlidingWindow[int](3)
    # need 2 odds within any 3-window; first at indices (0,2) with 1 and 3
    assert w.first_window_reaching([1, 2, 3, 4, 5], _odd, 2) == (0, 2)


def test_first_window_reaching_none() -> None:
    w = SlidingWindow[int](2)
    assert w.first_window_reaching([2, 4, 6], _odd, 1) is None
    # threshold never reached within window size
    assert w.first_window_reaching([1, 2, 1, 2, 1], _odd, 3) is None


def test_bad_size() -> None:
    with pytest.raises(ValueError, match="size"):
        SlidingWindow[int](0)
