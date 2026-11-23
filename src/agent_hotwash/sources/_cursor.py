"""Normalize the tool data shared by Cursor's CLI stream and chat store."""

from __future__ import annotations

import json
import posixpath
from typing import Any

from agent_hotwash.events import ArtifactInteraction, ArtifactOp, Event, EventKind, RoleHint
from agent_hotwash.primitives.reads import read_paths
from agent_hotwash.sources._common import (
    diagnostic_excerpt,
    flatten_text,
    output_metadata,
    truncate,
    truncate_head_tail,
)

_TOOL_OPS = {
    "read": "cmd.read",
    "readfile": "cmd.read",
    "glob": "cmd.search",
    "grep": "cmd.search",
    "semsearch": "cmd.search",
    "ls": "cmd.list",
    "shell": "cmd.exec",
    "bash": "cmd.exec",
    "edit": "file.edit",
    "editfile": "file.edit",
    "strreplace": "file.edit",
    "write": "file.write",
    "writefile": "file.write",
    "delete": "file.delete",
    "deletefile": "file.delete",
    "websearch": "web.search",
    "fetch": "web.open",
    "webfetch": "web.open",
    "task": "agent.spawn",
    "createagent": "agent.spawn",
    "await": "agent.wait",
    "sendmessage": "agent.message",
    "updatetodos": "plan.update",
    "todowrite": "plan.update",
    "createplan": "plan.update",
}
_FILE_OPS = {
    "cmd.read": ArtifactOp.read,
    "file.edit": ArtifactOp.update,
    "file.write": ArtifactOp.add,
    "file.delete": ArtifactOp.delete,
}
_FAILURE_KEYS = (
    "error",
    "failure",
    "rejected",
    "timeout",
    "fileNotFound",
    "permissionDenied",
    "spawnError",
    "commandFailed",
)
_CONTEXT_TAGS = ("<user_info>", "<system_reminder>", "<system_notification>", "<timestamp>")


def object_field(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def string_field(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def integer_field(value: Any) -> int | None:
    return value if type(value) is int else None


def user_prompt(text: str, *, has_request: bool = False) -> tuple[str, RoleHint]:
    stripped = text.lstrip()
    if has_request or stripped.startswith("<user_query>"):
        _prefix, opening, tail = stripped.partition("<user_query>")
        query, closing, _suffix = tail.rpartition("</user_query>")
        if opening and closing:
            return query.strip(), RoleHint.user
    if stripped.startswith(_CONTEXT_TAGS) and not has_request:
        return text, RoleHint.injected
    return text, RoleHint.user


def _resolve_path(path: str, cwd: str | None) -> str:
    if cwd and posixpath.isabs(cwd) and not path.startswith(("/", "~", "$")):
        return posixpath.normpath(posixpath.join(cwd, path))
    return path


def tool_call(name: str, args: dict[str, Any], cwd: str | None, **fields: Any) -> Event:
    normalized = name.lower().replace("_", "")
    op_kind = _TOOL_OPS.get(normalized)
    if normalized in ("calldynamictool", "mcp"):
        namespace = string_field(args.get("namespace")) or string_field(args.get("server"))
        tool_name = string_field(args.get("toolName"))
        if namespace and tool_name:
            op_kind = f"mcp.{namespace}.{tool_name}"
    path = next(
        (value for key in ("path", "file_path", "filePath", "targetFile") if (value := string_field(args.get(key)))),
        None,
    )
    if path and op_kind in _FILE_OPS:
        artifacts = [ArtifactInteraction(path=_resolve_path(path, cwd), op=_FILE_OPS[op_kind])]
    elif op_kind == "cmd.exec":
        directory = string_field(args.get("workingDirectory")) or string_field(args.get("working_directory")) or cwd
        artifacts = [
            ArtifactInteraction(path=_resolve_path(path, directory), op=ArtifactOp.read)
            for path in read_paths(name, args)
        ]
    else:
        artifacts = []
    return Event(
        kind=EventKind.tool_call, tool_name=name, tool_args=args, op_kind=op_kind, artifacts=artifacts, **fields
    )


def _output_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return flatten_text(value) or json.dumps(value, ensure_ascii=True, sort_keys=True)
    if value is not None:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return None


def tool_result(
    name: str | None,
    outcome: dict[str, Any],
    *,
    fallback: Any = None,
    is_error: bool | None = None,
    **fields: Any,
) -> Event:
    success_value = outcome.get("success")
    success = object_field(success_value)
    failure_key = next(
        (key for key in _FAILURE_KEYS if (value := outcome.get(key)) is not None and value is not False), None
    )
    failure = outcome.get(failure_key) if failure_key else None
    error = object_field(failure)
    payload = error or success
    exit_code = integer_field(payload.get("exitCode"))
    if exit_code is None:
        exit_code = integer_field(payload.get("exit_code"))
    stdout, stderr = payload.get("stdout"), payload.get("stderr")
    output = next(
        (
            value
            for key in ("interleavedOutput", "content", "markdown")
            if isinstance(value := payload.get(key), str) and value
        ),
        None,
    )
    if output is None and (isinstance(stdout, str) or isinstance(stderr, str)):
        output = "\n".join(text for text in (stdout, stderr) if isinstance(text, str) and text) or None
    error_text = next(
        (text for key in ("message", "reason", "errorMessage", "error") if (text := string_field(error.get(key)))),
        None,
    ) or string_field(failure)
    if output is None:
        structured = {key: success[key] for key in ("files", "workspaceResults", "references") if key in success}
        captured = fallback if fallback is not None else structured or failure or (outcome if not success else None)
        output = error_text or _output_text(captured)
    ok = None
    if (success_value is not None and success_value is not False) or is_error is False:
        ok = True
    if failure_key or is_error is True:
        ok = False
    if exit_code is not None and exit_code != 0:
        ok = False
    return Event(
        kind=EventKind.tool_result,
        tool_name=name,
        ok=ok,
        exit_code=exit_code,
        error_text=truncate(error_text or output) if ok is False else None,
        output=truncate_head_tail(output),
        diagnostic_excerpt=diagnostic_excerpt(output),
        **output_metadata(output),
        **fields,
    )


def attach_file_change(call: Event, result: Event, outcome: dict[str, Any], cwd: str | None) -> None:
    if result.ok is not True or call.op_kind not in ("file.edit", "file.write", "file.delete"):
        return
    data = object_field(outcome.get("success"))
    if not call.artifacts and (path := string_field(data.get("path"))):
        call.artifacts.append(ArtifactInteraction(path=_resolve_path(path, cwd), op=_FILE_OPS[call.op_kind]))
    if not call.artifacts:
        return
    artifact = call.artifacts[0]
    added = integer_field(data.get("linesAdded"))
    if added is None:
        added = integer_field(data.get("linesCreated"))
    removed = integer_field(data.get("linesRemoved"))
    diff = string_field(data.get("diff")) or string_field(data.get("unifiedDiff"))
    if added is not None and added >= 0:
        artifact.lines_added = call.lines_added = added
    if removed is not None and removed >= 0:
        artifact.lines_removed = call.lines_removed = removed
    if diff:
        artifact.diff_head = truncate(diff)
