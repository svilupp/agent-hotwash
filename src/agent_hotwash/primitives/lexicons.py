"""Compiled regex sets built from config lexicons.

One hot-swappable place that turns the ``[lexicons]`` word/phrase lists into
compiled ``re.Pattern`` objects. Word/phrase lexicons are matched
case-insensitively with word boundaries; ``secret`` is a set of raw regex
patterns matched as-is (case-sensitive).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_hotwash.config import Config, LexiconConfig


def _compile_words(words: list[str]) -> re.Pattern[str]:
    """Compile a set of words/phrases into one alternation with word boundaries.

    An empty list yields a pattern that never matches.
    """
    if not words:
        return re.compile(r"(?!x)x")  # matches nothing
    parts = sorted((re.escape(w.strip()) for w in words if w.strip()), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def _compile_raw(patterns: list[str]) -> re.Pattern[str]:
    if not patterns:
        return re.compile(r"(?!x)x")
    return re.compile("|".join(f"(?:{p})" for p in patterns))


@dataclass(frozen=True)
class Lexicons:
    """Compiled lexicon patterns. Build via :meth:`from_config`."""

    correction: re.Pattern[str]
    completion: re.Pattern[str]
    positive: re.Pattern[str]
    task_intro: re.Pattern[str]
    interrogative: re.Pattern[str]
    secret: re.Pattern[str]

    @classmethod
    def from_lexicon_config(cls, lex: LexiconConfig) -> Lexicons:
        return cls(
            correction=_compile_words(lex.correction),
            completion=_compile_words(lex.completion),
            positive=_compile_words(lex.positive),
            task_intro=_compile_words(lex.task_intro),
            interrogative=_compile_words(lex.interrogative),
            secret=_compile_raw(lex.secret),
        )

    @classmethod
    def from_config(cls, config: Config) -> Lexicons:
        return cls.from_lexicon_config(config.lexicons)


def is_interrogative(text: str, lex: Lexicons) -> bool:
    """A turn is interrogative if it ends with '?' or opens with an
    interrogative word (research #12 ACTING_ON_QUESTION)."""
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return lex.interrogative.match(t) is not None
