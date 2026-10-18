"""Offline Markdown and HTML reviews of persisted memory-audit evidence."""

from __future__ import annotations

import re
import unicodedata
from html import escape
from typing import Any
from urllib.parse import quote

from agent_hotwash.report.highlights import _CSS

CATEGORIES = ("user_corrections", "repeated_discovery", "repeated_reading")


def _display(value: Any) -> str:
    return "".join(c for c in str(value) if c in "\n\t" or not unicodedata.category(c).startswith("C"))


def _md(value: Any) -> str:
    text = " ".join(_display(value).splitlines())
    text = re.sub(r"(?<!\w)_|_(?!\w)", "&#95;", escape(text, quote=False))
    text = re.sub(r"^(\s*\d+)\.", r"\1&#46;", text)
    return re.sub(r"[\\`*{}\[\]()#+!|~]|^-", lambda m: f"&#{ord(m[0])};", text)


def _code(value: str) -> str:
    text = _display(value)
    fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", text)), default=0))
    return f"{fence}text\n{text}\n{fence}"


def _selected_matches(assessment: dict[str, Any]) -> list[dict[str, Any]]:
    status = assessment.get("guidance_status")

    def rank(match: dict[str, Any]) -> tuple[float, float]:
        coverage, conflict = match["coverage_probability"], match["conflict_probability"]
        if status == "possible_conflict":
            return conflict, coverage
        if status == "already_covered":
            return coverage if match.get("context_complete", True) else -1, conflict
        return max(coverage, conflict), conflict

    return sorted(assessment.get("guidance_matches", []), key=rank, reverse=True)[:3]


def _evidence_link(ref, record, sources):
    path = record.get("source_path") or sources.get(record["session_id"])
    line = record.get("line_number")
    if path and line:
        path = _display(path)
        if not path.startswith("/") or path.startswith("//"):
            path = "./" + path
        return f"- [{_md(ref)}](<{quote(path, safe='/')}#L{quote(str(line), safe='')}>)"
    return (
        f"- {_md(ref)} (canonical event {_md(record.get('canonical_event_index', 'unknown'))}; source line unavailable)"
    )


def render_memory_markdown(traces: dict[str, Any], result: dict[str, Any]) -> str:
    signal_count = sum(f.get("evidence_use") == "candidate_signal" for f in result.get("features", []))
    context_count = sum(f.get("evidence_use") == "supporting_context" for f in result.get("features", []))
    lines = [
        "# Memory extraction review",
        "",
        f"Status: {_md(result['status'])}.",
        "",
        "Proposals require review before installation. Screening thresholds are provisional and uncalibrated.",
        "",
        (
            (
                "Guidance comparison covers only explicitly selected file snapshots. Matches "
                "require human review; absence of a match does not establish novelty."
            )
            if result.get("guidance_assessments")
            else (
                "Novelty has not been checked against current repository guidance. Compare every "
                "draft with existing instructions and skills before accepting it."
            )
        ),
        "",
        f"Sessions: {len(traces['sessions'])}. Records: {len(traces['records'])}. "
        f"Retained evidence: {sum(f['retained'] for f in result.get('features', []))}.",
        "",
        f"Candidate-signal messages: {signal_count}. Supporting context records: {context_count}.",
        "",
    ]
    if result.get("error"):
        lines += [_md(result["error"]), ""]
    if result.get("failing_stage"):
        lines += ["Failing stage: " + _md(result["failing_stage"]) + ".", ""]
    groups = {
        record.get("independence_group_id")
        for record in traces["records"]
        if record.get("independence_confidence") == "known" and record.get("independence_group_id")
    }
    lines += [f"Known independent evidence groups: {len(groups)}.", ""]
    omitted = result.get("synthesis_evidence", {})
    if omitted.get("truncated_record_ids"):
        lines += [
            f"Synthesis omitted text from {len(omitted['truncated_record_ids'])} oversized records, "
            f"including {omitted['truncated_reasoning_records']} reasoning records. "
            "Full normalized text remains in `prepared.json`; proposal coverage is limited to the supplied excerpts.",
            "",
        ]
    coverage = traces.get("input_coverage", [])
    if coverage:
        lines += ["## Input coverage", ""]
        for item in coverage:
            lines += [f"- {_md(item['input'])}: {_md(item['parser'])}. " + _md(item.get("limitations", ""))]
        if traces.get("diagnostics"):
            lines += [
                "",
                f"Source diagnostics: {len(traces['diagnostics'])}; inspect `prepared.json` before accepting drafts.",
            ]
        lines += [""]
    records = {r["id"]: r for r in traces["records"]}
    sources = {s["session_id"]: s["source_path"] for s in traces["sessions"]}
    for category in CATEGORIES:
        lines += ["## " + category.replace("_", " ").title(), ""]
        candidates = [c for c in result["candidates"] if c["category"] == category]
        if not candidates:
            lines += ["No supported proposals in this run.", ""]
        for candidate in candidates:
            assessment = result.get("proposal_assessments", {}).get(candidate["id"], {})
            scope = (
                "Source task only in session " + ", ".join(assessment["scope_session_ids"])
                if assessment.get("scope_status") == "source_task_only"
                else "Requires scope review"
            )
            lines += [
                "### " + _md(candidate["statement"]),
                "",
                _md(candidate["reason"]),
                "",
                "Evidence scope: " + _md(scope) + ".",
                "",
                "Model-proposed applicability (requires review): " + _md(candidate["applicability"]),
                "",
                f"Intervention: {_md(candidate['intervention'])}. Supporting independent groups: "
                f"{len(candidate.get('supporting_independence_groups', []))}.",
                "",
                "Draft:",
                "",
                _code(candidate["draft"]),
                "",
                "Evidence:",
                "",
            ]
            for ref in candidate["evidence_ids"]:
                r = records[ref]
                lines += [_evidence_link(ref, r, sources) + f", session {_md(r['session_id'])}."]
            if assessment.get("guidance_status"):
                lines += [
                    "",
                    "Guidance comparison: "
                    + _md(assessment["guidance_status"])
                    + ". Scope: selected files only. Novelty is not established.",
                    "",
                ]
                if assessment.get("context_warning"):
                    lines += [_md(assessment["context_warning"]), ""]
                for match in _selected_matches(assessment):
                    lines += [
                        f"- {_md(match['path'])}:{_md(match['start_line'])}-{_md(match['end_line'])}, "
                        f"chunk {_md(match['chunk_id'])}, "
                        f"SHA-256 {_md(match['source_sha256'])}. Coverage {match['coverage_probability']:.3f}; "
                        f"conflict {match['conflict_probability']:.3f}.",
                        "Guidance context: "
                        + (
                            "complete selected file."
                            if match.get("context_complete", True)
                            else "partial excerpt; omitted context requires review."
                        ),
                        "",
                        _code(match["excerpt"]),
                        "",
                    ]
            if assessment.get("context_evidence_ids"):
                lines += ["", "Preceding agent context (interpretation only; not human correction evidence):", ""]
                for ref in assessment["context_evidence_ids"]:
                    r = records[ref]
                    lines += [_evidence_link(ref, r, sources) + "."]
            lines += [""]
    lines += [
        "## File and harness analytics",
        "",
        "Read frequency measures navigation effort. It does not establish that a file answered a question.",
        "Relative path labels may combine different project files. Absolute paths identify files more precisely.",
        "",
        "| File | Reads | Sessions | Repeated reads within sessions |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in traces["analytics"].get("most_opened_files", [])[:20]:
        path = _md(item["path"])
        lines += [f"| {path} | {item['read_count']} | {item['session_count']} | {item['repeated_within_sessions']} |"]
    hooks = result.get("command_hooks", {})
    if hooks:
        lines += [
            "",
            "## Command hooks",
            "",
            f"Submitted shell statements: {hooks['call_count']}. Hook matches: {hooks['match_count']}.",
            "",
            "Counts describe submitted shell statements. Conditional branches may never execute. "
            "Execution, success, and usefulness are unverified.",
            "",
            "| Hook or extracted value | Statements or occurrences | Sessions |",
            "| --- | ---: | ---: |",
        ]
        for item in hooks.get("summaries", []):
            lines += [f"| {_md(item['name'])} | {item['call_count']} | {item['session_count']} |"]
            for value in item.get("values", [])[:20]:
                lines += [
                    f"| {_md(item['name'])} / {_md(value['value'])} | {value['count']} | {value['session_count']} |"
                ]
        lines += [
            "",
            "Full calls, extracted values, and source references are in `command-hooks.json`, "
            "`command-calls.jsonl` / `.csv`, and `command-matches.jsonl` / `.csv`.",
            "",
            f"Command extraction diagnostics: {len(hooks.get('diagnostics', []))}. "
            "Inspect the dataset for limitations.",
        ]
    lines += ["", "Full counts and evidence IDs are in `prepared.json`.", "", "## Reasoning availability", ""]
    for session in traces["sessions"]:
        lines += [f"- {_md(session['session_id'])}: {_md(session.get('reasoning_visibility', 'unspecified'))}."]
    if result.get("provider_usage"):
        lines += ["", "## Provider execution", ""]
        for provider, usage in result["provider_usage"].items():
            lines += [
                f"- {_md(provider)}: {usage.get('requests', 0)} network requests; "
                f"{usage.get('cache_hits', 0)} native cache hits."
            ]
    return "\n".join(lines)


def _e(value: Any) -> str:
    return escape(_display(value), quote=True)


def render_memory_html(traces: dict[str, Any], result: dict[str, Any]) -> str:
    """Render a self-contained review without reading traces or calling providers."""
    records = {record["id"]: record for record in traces["records"]}
    features = result.get("features", [])
    groups = {
        record.get("independence_group_id")
        for record in records.values()
        if record.get("independence_confidence") == "known" and record.get("independence_group_id")
    }
    calls = result.get("calls", [])
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Skill and memory review</title><style>" + _CSS + "</style>",
        '<a class="skip" href="#main">Skip to review</a><main id="main">',
        "<h1>Skill and memory review</h1>",
        '<nav aria-label="Review sections"><a href="#drafts">Drafts</a>',
        '<a href="#inputs">Inputs</a><a href="#files">File reads</a>',
        '<a href="#providers">Provider usage</a></nav>',
        f"<p>Status <strong>{_e(result['status'])}</strong>.</p>",
        '<p class="note">Proposals are review drafts. Selected-file comparison does not establish novelty. '
        "Read frequency is an observation; suggested interventions require review.</p>",
        '<div class="metrics">',
        f'<div class="metric"><strong>{len(traces["sessions"])}</strong>Sessions</div>',
        f'<div class="metric"><strong>{len(records)}</strong>Evidence records</div>',
        f'<div class="metric"><strong>{len(groups)}</strong>Known independent groups</div>',
        f'<div class="metric"><strong>{sum(bool(f.get("retained")) for f in features)}</strong>Retained records</div>',
        "</div>",
    ]
    if result.get("error"):
        parts.append('<p class="note">Incomplete audit. ' + _e(result["error"]) + "</p>")
    if result.get("failing_stage"):
        parts.append("<p>Failing stage <code>" + _e(result["failing_stage"]) + "</code>.</p>")
    parts.append('<h2 id="drafts">Review drafts</h2>')
    for category in CATEGORIES:
        parts.append("<h3>" + _e(category.replace("_", " ").title()) + "</h3>")
        selected = [candidate for candidate in result.get("candidates", []) if candidate["category"] == category]
        if not selected:
            parts.append("<p>No supported proposals.</p>")
        for candidate in selected:
            assessment = result.get("proposal_assessments", {}).get(candidate["id"], {})
            parts.extend(
                [
                    '<article id="proposal-' + _e(candidate["id"]) + '">',
                    "<h4>" + _e(candidate["statement"]) + "</h4>",
                    "<p>" + _e(candidate["reason"]) + "</p>",
                    "<dl><dt>Proposed intervention</dt><dd>" + _e(candidate["intervention"]) + "</dd>",
                    "<dt>Applicability</dt><dd>" + _e(candidate["applicability"]) + "</dd>",
                    "<dt>Scope</dt><dd>" + _e(assessment.get("scope_status", "requires_review")) + "</dd>",
                    "<dt>Selected guidance</dt><dd>" + _e(assessment.get("guidance_status", "unchecked")) + "</dd>",
                    "<dt>Installation</dt><dd>Review draft</dd></dl>",
                    "<pre><code>" + _e(candidate["draft"]) + "</code></pre>",
                    "<details><summary>Supporting source evidence</summary>",
                ]
            )
            for ref in candidate["evidence_ids"]:
                record = records[ref]
                coordinate = (
                    str(record.get("source_path") or "source unavailable")
                    + ":"
                    + str(record.get("line_number") or "physical line unavailable")
                )
                parts.append("<p><code>" + _e(ref) + "</code> " + _e(coordinate) + "</p>")
                parts.append("<pre>" + _e(record["text"]) + "</pre>")
            parts.append("</details>")
            matches = _selected_matches(assessment)
            if matches:
                parts.append("<details><summary>Selected-guidance citations</summary>")
                for match in matches:
                    parts.append(
                        "<p>"
                        + _e(match["path"])
                        + ":"
                        + _e(match["start_line"])
                        + "-"
                        + _e(match["end_line"])
                        + "; chunk <code>"
                        + _e(match["chunk_id"])
                        + "</code>; SHA-256 <code>"
                        + _e(match["source_sha256"])
                        + "</code>.</p>"
                    )
                    if not match.get("context_complete", True):
                        parts.append('<p class="note">Partial context. Omitted text requires review.</p>')
                    parts.append("<pre>" + _e(match["excerpt"]) + "</pre>")
                parts.append("</details>")
            parts.append("</article>")
    parts.append('<h2 id="inputs">Inputs and visibility</h2><ul>')
    for item in traces.get("input_coverage", []):
        parts.append("<li>" + _e(item["input"]) + "; " + _e(item["parser"]) + "</li>")
    for session in traces["sessions"]:
        parts.append(
            "<li>"
            + _e(session["session_id"])
            + "; reasoning "
            + _e(session.get("reasoning_visibility", "unspecified"))
            + "</li>"
        )
    parts.append("</ul>")
    if traces.get("diagnostics"):
        parts.append("<p>Source diagnostics " + str(len(traces["diagnostics"])) + ". Inspect prepared.json.</p>")
    parts.append('<h2 id="files">Observed file reads</h2>')
    parts.append(
        "<p>Relative path labels may combine different project files. "
        'Absolute paths identify files more precisely.</p><div class="scroll"><table>'
    )
    parts.append("<caption>Most frequently read files</caption><tr><th>File</th><th>Reads</th><th>Sessions</th></tr>")
    for item in traces["analytics"].get("most_opened_files", [])[:20]:
        parts.append(
            "<tr><td><code>"
            + _e(item["path"])
            + "</code></td><td>"
            + _e(item["read_count"])
            + "</td><td>"
            + _e(item["session_count"])
            + "</td></tr>"
        )
    parts.append("</table></div>")
    hooks = result.get("command_hooks", {})
    if hooks:
        parts.append('<h2 id="command-hooks">Command hooks</h2>')
        parts.append(
            "<p>Submitted shell statements "
            + _e(hooks["call_count"])
            + "; hook matches "
            + _e(hooks["match_count"])
            + ". Counts describe submitted shell statements. Conditional branches may never execute. "
            "Execution, success, and usefulness are unverified.</p>"
        )
        parts.append(
            '<div class="scroll"><table><caption>Command counts and extracted values</caption>'
            "<tr><th>Hook or extracted value</th><th>Statements or occurrences</th><th>Sessions</th></tr>"
        )
        for item in hooks.get("summaries", []):
            parts.append(
                "<tr><td>"
                + _e(item["name"])
                + "</td><td>"
                + _e(item["call_count"])
                + "</td><td>"
                + _e(item["session_count"])
                + "</td></tr>"
            )
            for value in item.get("values", [])[:20]:
                parts.append(
                    "<tr><td>"
                    + _e(item["name"])
                    + " / "
                    + _e(value["value"])
                    + "</td><td>"
                    + _e(value["count"])
                    + "</td><td>"
                    + _e(value["session_count"])
                    + "</td></tr>"
                )
        parts.append(
            "</table></div><p>Full calls, extracted values, and source references are in command-hooks.json, "
            "command-calls.jsonl / .csv, and command-matches.jsonl / .csv. Extraction diagnostics "
            + _e(len(hooks.get("diagnostics", [])))
            + ".</p>"
        )
    parts.append('<h2 id="providers">Provider and cache usage</h2>')
    parts.append(
        "<p>Audit response cache hits "
        + str(sum(bool(call.get("cache_hit")) for call in calls))
        + "; misses "
        + str(sum(not call.get("cache_hit") for call in calls))
        + ". Audit cache misses do not establish network requests. "
        "See the stage manifest for actual provider usage.</p>"
    )
    for provider, usage in result.get("provider_usage", {}).items():
        parts.append(
            "<p>"
            + _e(provider)
            + " network requests "
            + _e(usage.get("requests", 0))
            + "; native cache hits "
            + _e(usage.get("cache_hits", 0))
            + ".</p>"
        )
    parts.append(
        "<p>Complete machine-readable evidence and results remain in prepared.json and memory.json.</p></main></html>"
    )
    return "".join(parts)
