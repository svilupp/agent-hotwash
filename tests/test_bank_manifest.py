"""Milestone feature bank load + uniqueness."""

from __future__ import annotations

from agent_hotwash.semantic.bank import load_bank


def test_load_bank_unique_ids_and_count() -> None:
    feats = load_bank()
    ids = [f.id for f in feats]
    assert len(ids) == len(set(ids))
    assert len(feats) >= 40
    scopes = {f.scope for f in feats}
    assert scopes >= {"task", "turn", "episode"}


def test_every_feature_has_inspect_paths_and_structured_instructions() -> None:
    for feat in load_bank():
        assert feat.inspect, feat.id
        ins = feat.question.get("instructions") if isinstance(feat.question, dict) else None
        assert isinstance(ins, dict), feat.id
        assert ins["question"] == feat.question_text
        assert "inspect" in ins
        assert "focus" in ins
