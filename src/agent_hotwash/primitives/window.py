"""Sliding-window counter over an event stream.

Windows are measured in *event count* (by ``idx``), never wall clock, because
``ts`` may be absent. Powers CONTEXT_ROT #1, CORRECTION_LOOP #3, RETRY_STORM #7,
RATE_LIMIT_LOOP #17.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class SlidingWindow[T]:
    """Count predicate matches within any contiguous window of ``size`` items.

    All methods are pure and take the sequence explicitly so a single window can
    be reused across detectors without carrying state.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("window size must be >= 1")
        self.size = size

    def max_count(self, items: Sequence[T], predicate: Callable[[T], bool]) -> int:
        """Maximum number of matches found in any window of ``size`` consecutive
        items. For sequences shorter than ``size`` the whole sequence is one
        window."""
        flags = [1 if predicate(x) else 0 for x in items]
        if not flags:
            return 0
        window = self.size
        running = sum(flags[:window])
        best = running
        for i in range(window, len(flags)):
            running += flags[i] - flags[i - window]
            best = max(best, running)
        return best

    def first_window_reaching(
        self, items: Sequence[T], predicate: Callable[[T], bool], threshold: int
    ) -> tuple[int, int] | None:
        """``(start, end)`` index bounds (inclusive) of the earliest window whose
        match count is ``>= threshold``, or ``None`` if never reached.

        ``end`` is the index of the item that pushed the count to the threshold -
        useful for anchoring a finding's span at the moment it fired.
        """
        flags = [1 if predicate(x) else 0 for x in items]
        window = self.size
        running = 0
        matched: list[int] = []  # idx positions of matches inside the live window
        for i, f in enumerate(flags):
            if f:
                matched.append(i)
            # drop matches that fell out of the window
            lo = i - window + 1
            while matched and matched[0] < lo:
                matched.pop(0)
            running = len(matched)
            if running >= threshold:
                return matched[0], i
        return None
