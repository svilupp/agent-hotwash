"""Lexicon compilation + matching tests."""

from __future__ import annotations

from agent_hotwash.config import load_config
from agent_hotwash.primitives.lexicons import Lexicons, is_interrogative


def _lex() -> Lexicons:
    return Lexicons.from_config(load_config())


def test_correction_matches() -> None:
    lex = _lex()
    assert lex.correction.search("no, that is wrong")
    assert lex.correction.search("please undo that")
    assert lex.correction.search("still broken")
    assert not lex.correction.search("looks perfect")


def test_positive_and_completion() -> None:
    lex = _lex()
    assert lex.positive.search("thanks, lgtm")
    assert lex.completion.search("all set now")
    # word boundary: 'undone' should not match 'undo'
    assert not lex.correction.search("the task is undoneable")


def test_secret_patterns() -> None:
    lex = _lex()
    assert lex.secret.search("key AKIA1234567890ABCDEF here")
    assert lex.secret.search("token ghp_" + "a" * 36)
    assert lex.secret.search("slack xoxb-123456789012-abcdefg")
    assert lex.secret.search("-----BEGIN RSA PRIVATE KEY-----")
    assert not lex.secret.search("no secrets in this line at all")


def test_is_interrogative() -> None:
    lex = _lex()
    assert is_interrogative("why is this failing?", lex)
    assert is_interrogative("what does this do", lex)
    assert is_interrogative("is it working?", lex)
    assert not is_interrogative("implement the feature", lex)
    assert not is_interrogative("", lex)


def test_empty_lexicon_never_matches() -> None:
    from agent_hotwash.config import LexiconConfig

    lex = Lexicons.from_lexicon_config(LexiconConfig())
    assert not lex.correction.search("no wrong undo")
    assert not lex.secret.search("AKIA1234567890ABCDEF")
