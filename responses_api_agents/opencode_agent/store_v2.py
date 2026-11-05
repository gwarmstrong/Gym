# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read an OpenCode 2.x session store through the 1.x tables the adapter parses.

FEASIBILITY SPIKE, not reviewed for a PR. OpenCode 2.x keeps sessions in ``session_v2``, ``session_message`` and
``session_part``; each message row is typed (``user``, ``assistant``, ``compaction``, ``idle`` and control markers)
and an assistant's text, reasoning and tool calls are its ``content``. ``project_v1`` writes the 1.x ``session``,
``message`` and ``part`` tables with 1.x-shaped JSON so the existing parsers read 2.x runs unchanged. Known losses:
compaction boundaries (``tail_start_id``), subagent spawn links and tool ``time.compacted`` have no mapping yet, and
2.x error types are mapped onto 1.x names only for provider status and context overflow.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


_V1_SCHEMA = """
create table session (id text primary key, parent_id text, time_created integer);
create table message (id text primary key, session_id text, time_created integer, data text);
create table part (id text primary key, message_id text, session_id text, time_created integer, data text);
"""


def is_v2_store(database: Path) -> bool:
    """Whether ``database`` holds OpenCode 2.x session tables."""
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as source:
        return (
            source.execute("select 1 from sqlite_master where type='table' and name='session_v2'").fetchone()
            is not None
        )


def _v1_error(error: dict[str, Any] | None) -> dict[str, Any] | None:
    """A 1.x-shaped error: provider failures become ``APIError`` with their HTTP status."""
    if not isinstance(error, dict):
        return None
    kind = str(error.get("type") or "unknown")
    data: dict[str, Any] = {"message": error.get("message"), "type": kind}
    if isinstance(error.get("status"), int):
        data["statusCode"] = error["status"]
    if kind == "provider.transport" or kind == "provider.timeout":
        data["isRetryable"] = True
    name = "APIError" if kind.startswith("provider.") else kind
    return {"name": name, "data": data}


def _tool_output(state: dict[str, Any]) -> str | None:
    content = state.get("content")
    if not isinstance(content, list):
        return None
    texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
    return "\n".join(texts)


def _v1_tool(content: dict[str, Any]) -> dict[str, Any]:
    state = content.get("state") if isinstance(content.get("state"), dict) else {}
    timing = content.get("time") if isinstance(content.get("time"), dict) else {}
    status = state.get("status")
    v1_state: dict[str, Any] = {
        # 2.x streams arguments as "streaming"; 1.x called that state "pending".
        "status": "pending" if status == "streaming" else status,
        "input": state.get("input") if isinstance(state.get("input"), dict) else {},
        "metadata": state.get("metadata") if isinstance(state.get("metadata"), dict) else {},
        "time": {"start": timing.get("ran", timing.get("created")), "end": timing.get("completed")},
    }
    output = _tool_output(state)
    if status == "completed":
        v1_state["output"] = output
    elif status == "error":
        error = state.get("error") if isinstance(state.get("error"), dict) else {}
        v1_state["error"] = error.get("message") or output
    return {"type": "tool", "callID": content.get("id"), "tool": content.get("name"), "state": v1_state}


def _rows(source: sqlite3.Connection) -> tuple[list[tuple], list[tuple], list[tuple]]:
    sessions = source.execute(
        "select id, parent_id, time_created from session_v2 order by time_created, id"
    ).fetchall()
    stored_parts: dict[str, list[dict[str, Any]]] = {}
    for message_id, data in source.execute("select message_id, data from session_part order by message_id, ordinal"):
        stored_parts.setdefault(message_id, []).append(json.loads(data))
    messages: list[tuple] = []
    parts: list[tuple] = []
    last_user: dict[str, str] = {}
    for message_id, session_id, kind, created, raw in source.execute(
        "select id, session_id, type, time_created, data from session_message order by session_id, seq"
    ):
        data = json.loads(raw)
        times = data.get("time") if isinstance(data.get("time"), dict) else {}

        def add_part(ordinal: int, part: dict[str, Any], message_id: str = message_id) -> None:
            # Zero-padded ordinals keep the parsers' (time_created, id) order equal to OpenCode's.
            parts.append((f"{message_id}-{ordinal:05d}", message_id, session_id, created, json.dumps(part)))

        if kind in ("user", "synthetic"):
            last_user[session_id] = message_id
            info = {"role": "user", "time": {"created": times.get("created")}}
            messages.append((message_id, session_id, created, json.dumps(info)))
            add_part(
                0,
                {"type": "text", "text": data.get("text", ""), **({"synthetic": True} if kind == "synthetic" else {})},
            )
        elif kind == "assistant":
            model = data.get("model") if isinstance(data.get("model"), dict) else {}
            info = {
                "role": "assistant",
                "parentID": last_user.get(session_id),
                "agent": data.get("agent"),
                "mode": data.get("agent"),
                "modelID": model.get("id"),
                "providerID": model.get("providerID"),
                "time": {"created": times.get("created"), "completed": times.get("completed")},
                "finish": data.get("finish"),
                "cost": data.get("cost"),
                "tokens": data.get("tokens"),
            }
            error = _v1_error(data.get("error"))
            if error is not None:
                info["error"] = error
            messages.append((message_id, session_id, created, json.dumps(info)))
            content = (
                data.get("content") if isinstance(data.get("content"), list) else stored_parts.get(message_id, [])
            )
            add_part(0, {"type": "step-start"})
            for ordinal, item in enumerate(content, start=1):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "tool":
                    add_part(ordinal, _v1_tool(item))
                elif item.get("type") in ("text", "reasoning"):
                    add_part(ordinal, {"type": item["type"], "text": item.get("text", "")})
            if data.get("finish") is not None:
                add_part(
                    len(content) + 1,
                    {"type": "step-finish", "reason": data.get("finish"), "tokens": data.get("tokens")},
                )
        elif kind == "compaction" and data.get("status") == "completed":
            # 1.x recorded a compaction as a user message with a compaction part answered by a summary assistant.
            messages.append((message_id, session_id, created, json.dumps({"role": "user", "time": times})))
            add_part(0, {"type": "compaction", "auto": data.get("reason") == "auto"})
            summary_id = f"{message_id}-summary"
            info = {"role": "assistant", "summary": True, "parentID": message_id, "time": times, "finish": "stop"}
            messages.append((summary_id, session_id, created, json.dumps(info)))
            add_part(0, {"type": "text", "text": data.get("summary", "")}, summary_id)
    return sessions, messages, parts


def project_v1(source_path: Path, destination: Path) -> None:
    """Write the 1.x ``session``, ``message`` and ``part`` tables of the 2.x store ``source_path`` to ``destination``."""
    with closing(sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)) as source:
        sessions, messages, parts = _rows(source)
    with closing(sqlite3.connect(destination)) as target:
        target.executescript(_V1_SCHEMA)
        target.executemany("insert into session values (?, ?, ?)", sessions)
        target.executemany("insert into message values (?, ?, ?, ?)", messages)
        target.executemany("insert into part values (?, ?, ?, ?, ?)", parts)
        target.commit()
