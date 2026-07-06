"""Layer 1 analytics — deterministic per-trace metrics.

``analyze(trace, config) -> Analysis`` folds a normalized :class:`Trace` into a
JSON-serializable metric surface: turn/tool/error counts, token totals and cost,
wall-clock timing, file-operation ratios, and the inferred outcome. Metrics are
computed for the root session and summarized for each linked subagent session.

Every metric that depends on data a source may not carry (per-event ``ts``,
token ``usage``) yields ``None`` rather than a misleading ``0``. The names of
those degraded metrics are recorded in ``Analysis.degraded`` so a reader can
tell "no idle time" from "we could not measure idle time".
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from agent_hotwash.events import AgentKind, EventKind, ToolCategory
from agent_hotwash.primitives.commands import classify_command
from agent_hotwash.primitives.lexicons import Lexicons
from agent_hotwash.primitives.outcome import Outcome, label_outcome

if TYPE_CHECKING:
    from agent_hotwash.config import Config, PriceEntry
    from agent_hotwash.events import Event, Session, Trace

_TEST_MARKERS = ("pytest", "vitest", "jest", "tsc", "unittest", " test", "test ")


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class TokenTotals(BaseModel):
    """Summed per-event token deltas. ``None`` fields mean the source carried no
    usage for that dimension (never a fabricated ``0``)."""

    input: int | None = None
    output: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None

    @property
    def total(self) -> int | None:
        parts = [self.input, self.output, self.cache_read, self.cache_write]
        present = [p for p in parts if p is not None]
        return sum(present) if present else None


class SessionMetrics(BaseModel):
    """The full L1 metric set for a single session (root or subagent)."""

    session_id: str
    agent: AgentKind
    model: str | None = None
    is_subagent: bool = False

    has_timestamps: bool = False
    has_any_timestamps: bool = False
    ts_coverage: float = 0.0
    usage_reliable: bool = True

    event_count: int = 0
    user_turns: int = 0
    assistant_turns: int = 0
    thinking_events: int = 0
    thinking_chars: int = 0
    compaction_count: int = 0

    tool_calls_total: int = 0
    tools_by_name: dict[str, int] = Field(default_factory=dict)
    tools_by_category: dict[str, int] = Field(default_factory=dict)
    tool_calls_per_turn: float | None = None

    tool_results_total: int = 0
    tool_error_count: int = 0
    tool_error_rate: float | None = None
    errors_by_tool: dict[str, int] = Field(default_factory=dict)
    error_categories: dict[str, int] = Field(default_factory=dict)
    error_severity: dict[str, int] = Field(default_factory=dict)

    tokens: TokenTotals = Field(default_factory=TokenTotals)
    cache_hit_ratio: float | None = None

    read_count: int = 0
    write_count: int = 0
    edit_count: int = 0
    bash_count: int = 0
    unique_files_touched: int = 0
    unique_files_read: int = 0
    unique_files_written: int = 0
    files_by_edits: dict[str, int] = Field(default_factory=dict)
    edit_write_ratio: float | None = None
    read_before_first_edit: int = 0

    events_to_first_tool_call: int | None = None
    corrections_count: int = 0

    lines_added: int | None = None
    lines_removed: int | None = None

    test_run_count: int = 0
    test_pass_count: int = 0
    test_fail_count: int = 0
    test_pass_fail_transitions: int = 0

    retry_after_error: int = 0
    max_error_streak: int = 0
    edit_test_cycles: int = 0
    distinct_bash_commands: int = 0

    duration_seconds: float | None = None
    active_seconds: float | None = None
    idle_seconds: float | None = None


class Analysis(BaseModel):
    """Trace-level analytics: root metrics, subagent summaries, cost and outcome."""

    trace_id: str
    agent: AgentKind
    model: str | None = None
    experiment: str | None = None
    instance_id: str | None = None
    resolved: bool | None = None

    root: SessionMetrics
    subagents: list[SessionMetrics] = Field(default_factory=list)
    subagent_count: int = 0
    subagent_event_total: int = 0
    subagent_fanout: int = 0

    total_tokens: TokenTotals = Field(default_factory=TokenTotals)
    cost: float | None = None
    cost_source: str | None = None  # "provenance" | "estimated" | None
    # Always the token x price-table estimate when tokens + a price are known,
    # independent of `cost`. When `cost_source == "provenance"` this lets a
    # reader see drift between the harness figure and our recomputation.
    cost_estimated: float | None = None

    outcome: Outcome = Field(default_factory=Outcome)
    degraded: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _bash_command(ev: Event) -> str | None:
    if ev.kind is not EventKind.tool_call or ev.tool_category is not ToolCategory.execute:
        return None
    args = ev.tool_args or {}
    for key in ("command", "cmd", "script"):
        val = args.get(key)
        if isinstance(val, str):
            return val
    return None


def _is_test_command(cmd: str) -> bool:
    if classify_command(cmd) == "build_test":
        return True
    low = cmd.lower()
    return any(m in low for m in _TEST_MARKERS)


def _sum_optional(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


# ---------------------------------------------------------------------------
# Per-session metrics
# ---------------------------------------------------------------------------


def _result_ok_by_call(session: Session) -> dict[str, bool | None]:
    return {ev.call_id: ev.ok for ev in session.events if ev.kind is EventKind.tool_result and ev.call_id is not None}


def analyze_session(session: Session, config: Config, *, is_subagent: bool = False) -> SessionMetrics:
    """Compute the full L1 metric set for one session."""
    events = session.events
    m = SessionMetrics(
        session_id=session.session_id,
        agent=session.agent,
        model=session.model,
        is_subagent=is_subagent,
        has_timestamps=session.has_timestamps,
        has_any_timestamps=session.has_any_timestamps,
        ts_coverage=session.ts_coverage,
        usage_reliable=session.usage_reliable,
        event_count=len(events),
    )
    lex = Lexicons.from_config(config)

    # --- turn / thinking / compaction / tool counts -----------------------
    first_tool_idx: int | None = None
    reads_before_edit = 0
    seen_edit = False

    la_present: list[int | None] = []
    lr_present: list[int | None] = []

    for ev in events:
        if ev.kind is EventKind.user_msg:
            m.user_turns += 1
            if ev.text and lex.correction.search(ev.text):
                m.corrections_count += 1
        elif ev.kind is EventKind.assistant_msg:
            m.assistant_turns += 1
        elif ev.kind is EventKind.thinking:
            m.thinking_events += 1
            m.thinking_chars += len(ev.text or "")
        elif ev.kind is EventKind.compaction:
            m.compaction_count += 1
        elif ev.kind is EventKind.tool_call:
            m.tool_calls_total += 1
            name = ev.tool_name or "?"
            m.tools_by_name[name] = m.tools_by_name.get(name, 0) + 1
            cat = ev.tool_category or ToolCategory.other
            m.tools_by_category[cat.value] = m.tools_by_category.get(cat.value, 0) + 1
            if first_tool_idx is None:
                first_tool_idx = ev.idx
            if cat is ToolCategory.read:
                m.read_count += 1
                if not seen_edit:
                    reads_before_edit += 1
            elif cat is ToolCategory.write:
                if (ev.tool_name or "").lower() == "write":
                    m.write_count += 1
                else:
                    m.edit_count += 1
                seen_edit = True
                if ev.path:
                    m.files_by_edits[ev.path] = m.files_by_edits.get(ev.path, 0) + 1
                la_present.append(ev.lines_added)
                lr_present.append(ev.lines_removed)
            elif cat is ToolCategory.execute and _bash_command(ev) is not None:
                m.bash_count += 1

    m.events_to_first_tool_call = first_tool_idx
    m.read_before_first_edit = reads_before_edit
    m.lines_added = _sum_optional(la_present)
    m.lines_removed = _sum_optional(lr_present)

    if m.assistant_turns:
        m.tool_calls_per_turn = m.tool_calls_total / m.assistant_turns

    # --- file uniqueness --------------------------------------------------
    fs = session.file_state
    m.unique_files_touched = len(fs)
    m.unique_files_read = sum(1 for st in fs.values() if st.ever_read)
    m.unique_files_written = sum(1 for st in fs.values() if st.edit_count or st.write_count)
    if m.write_count:
        m.edit_write_ratio = m.edit_count / m.write_count

    # --- errors -----------------------------------------------------------
    call_by_id = {ev.call_id: ev for ev in events if ev.kind is EventKind.tool_call and ev.call_id}
    from agent_hotwash.primitives.errors import classify_error

    streak = 0
    for ev in events:
        if ev.kind is not EventKind.tool_result:
            continue
        m.tool_results_total += 1
        if ev.ok is False:
            m.tool_error_count += 1
            streak += 1
            m.max_error_streak = max(m.max_error_streak, streak)
            call = call_by_id.get(ev.call_id) if ev.call_id else None
            tool = (call.tool_name if call else ev.tool_name) or "?"
            m.errors_by_tool[tool] = m.errors_by_tool.get(tool, 0) + 1
            cat = ev.error_category or "other"
            m.error_categories[cat] = m.error_categories.get(cat, 0) + 1
            cmd = (_bash_command(call) if call else None) or ""
            _c, severity, _conf = classify_error(tool, ev.exit_code, ev.error_text or ev.output or "", cmd)
            m.error_severity[severity] = m.error_severity.get(severity, 0) + 1
        elif ev.ok is True:
            streak = 0
    if m.tool_results_total:
        m.tool_error_rate = m.tool_error_count / m.tool_results_total

    # retry-after-error: errored results with any later tool_call
    error_result_idxs = [ev.idx for ev in events if ev.kind is EventKind.tool_result and ev.ok is False]
    tool_call_idxs = [ev.idx for ev in events if ev.kind is EventKind.tool_call]
    for eidx in error_result_idxs:
        if any(c > eidx for c in tool_call_idxs):
            m.retry_after_error += 1

    # --- tokens -----------------------------------------------------------
    usages = [ev.usage for ev in events if ev.usage is not None]
    if usages:
        m.tokens = TokenTotals(
            input=_sum_optional([u.input for u in usages]),
            output=_sum_optional([u.output for u in usages]),
            cache_read=_sum_optional([u.cache_read for u in usages]),
            cache_write=_sum_optional([u.cache_write for u in usages]),
        )
        cr, inp = m.tokens.cache_read, m.tokens.input
        if cr is not None and inp is not None and (cr + inp) > 0:
            m.cache_hit_ratio = cr / (cr + inp)

    # --- tests + edit/test cycles ----------------------------------------
    result_ok = _result_ok_by_call(session)
    test_states: list[bool] = []
    pending_edit = False
    seen_bash: set[str] = set()
    for ev in events:
        cmd = _bash_command(ev)
        if ev.kind is EventKind.tool_call and ev.tool_category is ToolCategory.write:
            pending_edit = True
        if cmd is None:
            continue
        seen_bash.add(cmd.strip())
        if not _is_test_command(cmd):
            continue
        ok = result_ok.get(ev.call_id) if ev.call_id else None
        if ok is None:
            ok = ev.ok
        m.test_run_count += 1
        if ok is True:
            m.test_pass_count += 1
            test_states.append(True)
        elif ok is False:
            m.test_fail_count += 1
            test_states.append(False)
        if pending_edit:
            m.edit_test_cycles += 1
            pending_edit = False
    m.distinct_bash_commands = len(seen_bash)
    m.test_pass_fail_transitions = sum(1 for a, b in pairwise(test_states) if a != b)

    # --- timing (only when timestamps are present) ------------------------
    if session.has_timestamps:
        ts = [ev.ts for ev in events if ev.ts is not None]
        if len(ts) >= 2:
            ts_sorted = sorted(ts)
            m.duration_seconds = (ts_sorted[-1] - ts_sorted[0]).total_seconds()
            gap_cap = config.analytics.idle_gap_minutes * 60.0
            active = 0.0
            idle = 0.0
            for a, b in pairwise(ts_sorted):
                delta = (b - a).total_seconds()
                if delta > gap_cap:
                    idle += delta
                else:
                    active += delta
            m.active_seconds = active
            m.idle_seconds = idle
        else:
            m.duration_seconds = 0.0
            m.active_seconds = 0.0
            m.idle_seconds = 0.0

    return m


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def _cost_of(tokens: TokenTotals, price: PriceEntry) -> float:
    def part(n: int | None, per_mtok: float) -> float:
        return (n or 0) / 1_000_000.0 * per_mtok

    return (
        part(tokens.input, price.input)
        + part(tokens.output, price.output)
        + part(tokens.cache_read, price.cache_read)
        + part(tokens.cache_write, price.cache_write)
    )


def _as_cost(val: Any) -> float | None:
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None


def _provenance_cost(trace: Trace) -> float | None:
    """Harness-authoritative cost for a trace, ``None`` if none is recorded.

    Traverses the real ``harness_meta`` nesting the code-bench parser produces
    (``{"run", "metrics", "verification"}`` plus any parser-provided cost the
    stdout stream carried, e.g. claude's ``result`` line ``total_cost_usd``).
    """
    hm = trace.provenance.harness_meta if trace.provenance else {}
    if not isinstance(hm, dict):
        return None

    # 1. Parser-provided cost lifted straight off the stream (flat keys).
    for key in ("total_cost_usd", "cost_usd", "cost", "total_cost"):
        val = hm.get(key)
        cost = _as_cost(val)
        if cost is not None:
            return cost

    # 2. metrics.json cost — nested as metrics["cost"]["total"|...].
    metrics = hm.get("metrics")
    if isinstance(metrics, dict):
        cost_field = metrics.get("cost")
        if isinstance(cost_field, dict):
            for key in ("total", "cost_reported", "computed"):
                cost = _as_cost(cost_field.get(key))
                if cost is not None:
                    return cost
        else:
            cost = _as_cost(cost_field)
            if cost is not None:
                return cost
        for key in ("total_cost_usd", "cost_usd", "total_cost"):
            cost = _as_cost(metrics.get(key))
            if cost is not None:
                return cost
    return None


# ---------------------------------------------------------------------------
# Trace-level entry point
# ---------------------------------------------------------------------------


def analyze(trace: Trace, config: Config) -> Analysis:
    """Fold a Trace into an :class:`Analysis` metric surface."""
    root = analyze_session(trace.root, config)
    subs = [analyze_session(s, config, is_subagent=True) for s in trace.subagents]

    # aggregate tokens across every session
    all_metrics = [root, *subs]
    total_tokens = TokenTotals(
        input=_sum_optional([mm.tokens.input for mm in all_metrics]),
        output=_sum_optional([mm.tokens.output for mm in all_metrics]),
        cache_read=_sum_optional([mm.tokens.cache_read for mm in all_metrics]),
        cache_write=_sum_optional([mm.tokens.cache_write for mm in all_metrics]),
    )

    # Estimate cost from tokens x the model price table, always when computable,
    # so it can be cross-checked against a provenance figure (drift detection).
    est = 0.0
    priced_any = False
    for mm in all_metrics:
        if mm.tokens.total is None:
            continue
        price = config.price_for(mm.model or trace.model)
        if price is None:
            continue
        est += _cost_of(mm.tokens, price)
        priced_any = True
    cost_estimated = est if priced_any else None

    # Headline cost: prefer harness-authoritative provenance, else the estimate.
    cost: float | None = None
    cost_source: str | None = None
    prov_cost = _provenance_cost(trace)
    if prov_cost is not None:
        cost, cost_source = prov_cost, "provenance"
    elif cost_estimated is not None:
        cost, cost_source = cost_estimated, "estimated"

    outcome = label_outcome(trace, config)

    degraded: list[str] = []
    if not trace.root.has_timestamps:
        degraded += ["duration_seconds", "active_seconds", "idle_seconds"]
    if root.tokens.total is None:
        degraded.append("tokens")
    if cost is None:
        degraded.append("cost")
    if not trace.root.usage_reliable:
        degraded.append("usage_reliable")

    return Analysis(
        trace_id=trace.trace_id,
        agent=trace.agent,
        model=trace.model,
        experiment=trace.experiment,
        instance_id=trace.instance_id,
        resolved=trace.resolved,
        root=root,
        subagents=subs,
        subagent_count=len(subs),
        subagent_event_total=sum(mm.event_count for mm in subs),
        subagent_fanout=len(subs),
        total_tokens=total_tokens,
        cost=cost,
        cost_source=cost_source,
        cost_estimated=cost_estimated,
        outcome=outcome,
        degraded=degraded,
    )


__all__ = ["Analysis", "SessionMetrics", "TokenTotals", "analyze", "analyze_session"]
