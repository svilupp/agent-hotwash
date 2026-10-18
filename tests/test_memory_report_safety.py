from html.parser import HTMLParser

import pytest
from markdown_it import MarkdownIt

from agent_hotwash.report.memory import render_memory_html, render_memory_markdown


def review(matches):
    return (
        {"sessions": [], "records": [], "analytics": {}},
        {
            "status": "complete",
            "candidates": [
                {
                    "id": "proposal",
                    "category": "user_corrections",
                    "statement": "Review this rule",
                    "reason": "Observed correction",
                    "applicability": "This task",
                    "intervention": "instruction",
                    "draft": "Use the rule",
                    "evidence_ids": [],
                }
            ],
            "proposal_assessments": {"proposal": {"guidance_matches": matches}},
        },
    )


def match(index, coverage=0.99, conflict=0.0, complete=True):
    return {
        "path": "guidance.md",
        "start_line": index,
        "end_line": index,
        "chunk_id": str(index),
        "source_sha256": "a" * 64,
        "coverage_probability": coverage,
        "conflict_probability": conflict,
        "context_complete": complete,
        "excerpt": f"Citation number {index}",
    }


@pytest.mark.parametrize("status", ["possible_conflict", "already_covered"])
@pytest.mark.parametrize("renderer", [render_memory_markdown, render_memory_html])
def test_report_shows_decisive_citation_after_display_limit(status, renderer):
    matches = [match(i, complete=False) for i in range(3)]
    matches.append(match(4, coverage=0.91, conflict=0.6 if status == "possible_conflict" else 0))
    traces, result = review(matches)
    result["proposal_assessments"]["proposal"]["guidance_status"] = status
    output = renderer(traces, result)
    assert "Citation number 4" in output
    assert sum(f"Citation number {i}" in output for i in range(5)) == 3


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "a":
            self.links.append(dict(attrs)["href"])


def test_markdown_preserves_nested_fences_as_literal_evidence():
    excerpt = "Before\n```sh\necho hello\n```\n# Source heading\n````\n<script>alert(1)</script>"
    traces, result = review([match(1)])
    assessment = result["proposal_assessments"]["proposal"]
    assessment["guidance_status"] = "needs_review"
    assessment["guidance_matches"][0]["excerpt"] = excerpt
    result["candidates"][0]["draft"] = excerpt
    tokens = MarkdownIt().parse(render_memory_markdown(traces, result))
    assert [token.content for token in tokens if token.type == "fence"] == [excerpt + "\n"] * 2
    assert not any(token.content == "Source heading" for token in tokens)


@pytest.mark.parametrize("renderer", [render_memory_markdown, render_memory_html])
def test_report_content_cannot_create_active_markup_or_links(renderer):
    attack = "<img src=x onerror=alert(1)>\n\n# forged\n[click](javascript:alert(1))\x00\u202e"
    traces, result = review([match(1)])
    for field in ("statement", "reason", "applicability", "draft"):
        result["candidates"][0][field] = attack
    result["error"] = attack
    traces["records"] = [
        {
            "id": "evidence",
            "session_id": "session",
            "text": attack,
            "source_path": "javascript:alert(1)",
            "line_number": 2,
        }
    ]
    result["candidates"][0]["evidence_ids"] = ["evidence"]
    output = renderer(traces, result)
    assert "\x00" not in output and "\u202e" not in output
    html = MarkdownIt("commonmark", {"html": True}).render(output) if renderer == render_memory_markdown else output
    elements = Elements()
    elements.feed(html)
    assert "img" not in elements.tags and "script" not in elements.tags
    assert all(not link.startswith("javascript:") for link in elements.links)
    if renderer == render_memory_markdown:
        assert elements.links == ["./javascript%3Aalert%281%29#L2"]


def test_preparation_failure_uses_safe_report_rendering(tmp_path):
    from agent_hotwash.config import Config
    from agent_hotwash.memory.pipeline import run_audit

    attack = "<img src=x onerror=alert(1)>\n```\n# forged\n[click](javascript:alert(1))\u202e"
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="trace input does not exist"):
        run_audit([tmp_path / attack], output, config=Config())
    for filename in ("report.md", "report.html"):
        text = (output / filename).read_text()
        assert "\u202e" not in text
        html = MarkdownIt("commonmark", {"html": True}).render(text) if filename.endswith(".md") else text
        elements = Elements()
        elements.feed(html)
        assert "img" not in elements.tags
        assert all(link in {"#main", "#drafts", "#inputs", "#files", "#providers"} for link in elements.links)
        assert "preparation" in text
