"""Label store resume, duplicates, non-TTY skip, criteria-hash migration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_hotwash.cli import app
from agent_hotwash.labels import (
    LabelRecord,
    SplitConflict,
    append_records,
    check_split_pinned,
    eval_store,
    labelled_keys,
    load_records,
    pinned_splits,
    record_key,
)
from agent_hotwash.semantic.bank import load_bank

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"
CLAUDE_RUN = FIXTURES / "codebench" / "claude_run"


def _rec(**over: Any) -> LabelRecord:
    base: dict[str, Any] = {
        "item_id": "t0:task:s0:task0",
        "root_trace_id": "t0",
        "scope": "task",
        "object_id": "s0:task0",
        "source_kind": "codebench",
        "digest_hash": "abc",
        "digest_schema_version": 1,
        "feature_id": "task.intent.inquire",
        "feature_version": 1,
        "criteria_hash": "hash-a",
        "answer": "yes",
        "annotator": "ann-1",
        "split": "dev",
        "provenance": "synthetic",
    }
    base.update(over)
    return LabelRecord.model_validate(base)


def test_resume_skips_existing_key(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    first = _rec()
    append_records(store, [first])
    keys = labelled_keys(load_records(store))
    assert record_key(first) in keys
    # Re-appending the same key is the store's job to avoid; CLI resume uses labelled_keys.
    dup = _rec(answer="no", annotator="ann-1")
    assert record_key(dup) == record_key(first)


def test_duplicate_handling_same_key(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    a = _rec(annotator="ann-1")
    b = _rec(annotator="ann-1", ts="2026-09-02T00:00:00+00:00")
    append_records(store, [a])
    existing = labelled_keys(load_records(store))
    written = []
    if record_key(b) not in existing:
        written.append(b)
    assert written == []
    assert len(load_records(store)) == 1


def test_second_annotator_is_new_work_not_resume_skip(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    a = _rec(annotator="ann-1")
    append_records(store, [a])
    existing = labelled_keys(load_records(store))
    b = _rec(annotator="ann-2", answer="no")
    assert record_key(b) != record_key(a)
    assert record_key(b) not in existing
    append_records(store, [b])
    rows = load_records(store)
    assert len(rows) == 2
    assert {r.annotator for r in rows} == {"ann-1", "ann-2"}
    # Same annotator again is still a resume-skip.
    assert record_key(_rec(annotator="ann-1", answer="maybe")) in labelled_keys(rows)


def test_second_annotator_cli_writes_double_labels(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    feat = "task.intent.inquire"
    args = ["label", str(CLAUDE_RUN), "--store", str(store), "--feature", feat, "--answer", "true"]
    first = runner.invoke(app, [*args, "--annotator", "ann-1"])
    assert first.exit_code == 0, first.stdout + first.stderr
    n = len(load_records(store))
    assert n > 0
    second = runner.invoke(app, [*args, "--annotator", "ann-2"])
    assert second.exit_code == 0, second.stdout + second.stderr
    rows = load_records(store)
    assert len(rows) == 2 * n
    assert {r.annotator for r in rows} == {"ann-1", "ann-2"}
    assert "resumed-skip 0" in second.stderr + second.stdout


def test_split_is_pinned_by_root_trace_id(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    append_records(store, [_rec(split="dev")])
    assert pinned_splits(load_records(store)) == {"t0": "dev"}
    check_split_pinned({"t0": "dev"}, "t0", "dev")  # same split: fine
    check_split_pinned({"t0": "dev"}, "t1", "held-out")  # unknown root: fine
    with pytest.raises(SplitConflict) as ei:
        check_split_pinned({"t0": "dev"}, "t0", "held-out")
    assert ei.value.pinned == "dev" and ei.value.requested == "held-out"
    # The store itself refuses a contradicting row.
    with pytest.raises(SplitConflict):
        append_records(store, [_rec(annotator="ann-2", split="held-out")])
    assert len(load_records(store)) == 1


def test_split_change_rejected_by_cli(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    feat = "task.intent.inquire"
    base = ["label", str(CLAUDE_RUN), "--store", str(store), "--feature", feat, "--answer", "true"]
    ok = runner.invoke(app, [*base, "--split", "dev", "--annotator", "ann-1"])
    assert ok.exit_code == 0, ok.stdout + ok.stderr
    n = len(load_records(store))
    bad = runner.invoke(app, [*base, "--split", "held-out", "--annotator", "ann-2"])
    assert bad.exit_code == 1
    assert "pinned" in bad.stderr + bad.stdout
    assert len(load_records(store)) == n  # nothing written


def test_criteria_hash_change_is_new_record(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    a = _rec(criteria_hash="hash-a")
    b = _rec(criteria_hash="hash-b", answer="no")
    append_records(store, [a, b])
    rows = load_records(store)
    assert len(rows) == 2
    assert record_key(a) != record_key(b)
    assert {r.criteria_hash for r in rows} == {"hash-a", "hash-b"}


def test_non_tty_skip_and_resume_cli(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    feat = "task.intent.inquire"
    result = runner.invoke(
        app,
        ["label", str(CLAUDE_RUN), "--store", str(store), "--feature", feat, "--annotator", "bot"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    rows = load_records(store)
    assert rows
    assert all(r.skip_reason == "non_tty" for r in rows)
    n = len(rows)
    # Resume: already-labelled keys are not written again.
    again = runner.invoke(
        app,
        ["label", str(CLAUDE_RUN), "--store", str(store), "--feature", feat, "--annotator", "bot"],
    )
    assert again.exit_code == 0, again.stdout + again.stderr
    assert len(load_records(store)) == n


def test_answer_flag_writes_label(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    result = runner.invoke(
        app,
        [
            "label",
            str(CLAUDE_RUN),
            "--store",
            str(store),
            "--feature",
            "task.intent.change",
            "--answer",
            "true",
            "--annotator",
            "a1",
            "--provenance",
            "synthetic",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    rows = load_records(store)
    assert rows
    assert all(r.answer == "true" and r.skip_reason is None for r in rows)
    assert all(r.provenance == "synthetic" for r in rows)


def test_eval_agreement_and_held_out_burn(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    bank = load_bank()
    feat = next(f for f in bank if f.id == "task.intent.inquire")
    a = _rec(annotator="ann-1", answer=0.9, criteria_hash=feat.id)
    b = _rec(annotator="ann-2", answer=0.9, criteria_hash=feat.id)
    c = _rec(item_id="t0:task:other", object_id="other", annotator="ann-1", answer=0.1, criteria_hash=feat.id)
    append_records(store, [a, b, c])
    report = eval_store(load_records(store))
    stats = report["features"]["task.intent.inquire"]
    assert stats["double_labelled"] == 1
    assert stats["agreement"] == 1.0
    assert stats["n"] == 3

    burned = store.with_name(store.name + ".burned")
    result = runner.invoke(app, ["eval", "--store", str(store), "--held-out"])
    assert result.exit_code == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert "features" in data
    assert burned.is_file()
    again = runner.invoke(app, ["eval", "--store", str(store), "--held-out"])
    assert again.exit_code == 1
