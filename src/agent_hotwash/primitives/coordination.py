"""Recorded orchestration operations, without inferring the purpose of reasoning."""

from agent_hotwash.events import Event, ToolCategory

_SPAWN = {"agent", "spawn_agent", "create_thread", "fork_thread"}
_STEER = {"steer_subagent", "send_message", "send_message_to_agent", "send_message_to_thread"}
_WAIT = {"get_subagent_result", "wait_agent", "wait_threads"}
_CONTROL = {"interrupt_agent", "close_agent"}


def tool_name(event: Event) -> str:
    name = (event.tool_name or "").lower().rsplit(".", 1)[-1]
    return name.rsplit("__", 1)[-1] if name.startswith("mcp__codex_app__") else name


def status_probe(event: Event) -> bool:
    name = tool_name(event)
    if name == "get_subagent_result":
        return event.tool_args.get("wait") is False
    if name == "write_stdin":
        return not event.tool_args.get("chars") and event.tool_args.get("yield_time_ms") == 0
    if name == "list_agents" or event.op_kind == "agent.status":
        return True
    return (event.op_kind == "agent.wait" or name in _WAIT) and event.tool_args.get("timeout_ms") == 0


def is_coordination(event: Event) -> bool:
    return (
        event.tool_category == ToolCategory.subagent
        or (event.op_kind or "").startswith("agent.")
        or tool_name(event) in _SPAWN | _STEER | _WAIT | _CONTROL | {"list_agents"}
    )


def coordination_kind(event: Event) -> str:
    name = tool_name(event)
    if status_probe(event):
        return "poll"
    if name == "agent" and event.tool_args.get("resume"):
        return "resume"
    if name in _SPAWN or event.op_kind == "agent.spawn":
        return "spawn"
    if name in _STEER or event.op_kind == "agent.message":
        return "steer"
    if name in _WAIT or event.op_kind == "agent.wait":
        return "wait"
    if name in _CONTROL:
        return "control"
    return "other"


def wait_mode(event: Event) -> str:
    if status_probe(event):
        return "nonblocking"
    if event.tool_args.get("wait") is True:
        return "blocking"
    timeout = event.tool_args.get("timeout_ms")
    if isinstance(timeout, (int, float)) and timeout > 0:
        return "blocking"
    return "unspecified"
