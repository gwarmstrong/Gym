# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from responses_api_agents.opencode_agent.artifacts import parse_opencode_session
from responses_api_agents.opencode_agent.snapshot import snapshot, stored_message_ids, stored_sessions
from responses_api_agents.opencode_agent.store_v2 import is_v2_store


def write_v2_store(path: Path) -> None:
    """The OpenCode 2.2.5 tables the projection reads, holding one shell call and a final answer."""
    with closing(sqlite3.connect(path)) as db:
        db.executescript(
            """
            create table session_v2 (id text primary key, parent_id text, time_created integer);
            create table session_message (id text primary key, session_id text, type text, seq integer,
                time_created integer, data text);
            create table session_part (message_id text, session_id text, ordinal integer, type text, call_id text,
                time_created integer, data text);
            """
        )
        db.execute("insert into session_v2 values ('ses_1', null, 1)")
        rows = [
            ("msg_0", "agent-switched", 1, {"time": {"created": 1}, "agent": "build"}),
            ("msg_1", "user", 2, {"time": {"created": 2}, "text": "Make out.txt", "files": []}),
            (
                "msg_2",
                "assistant",
                3,
                {
                    "time": {"created": 3, "completed": 9},
                    "agent": "build",
                    "model": {"id": "dummy_model", "providerID": "nemo_gym"},
                    "content": [
                        {"type": "reasoning", "text": "Run it."},
                        {
                            "type": "tool",
                            "id": "call_1",
                            "name": "shell",
                            "state": {
                                "status": "completed",
                                "input": {"command": "false"},
                                "content": [{"type": "text", "text": "(no output)"}],
                                "metadata": {"exit": 1},
                            },
                            "time": {"created": 4, "ran": 5, "completed": 6},
                        },
                    ],
                    "finish": "tool-calls",
                    "tokens": {"input": 10, "output": 2, "reasoning": 1, "cache": {"read": 3, "write": 0}},
                },
            ),
            (
                "msg_3",
                "assistant",
                4,
                {
                    "time": {"created": 10, "completed": 11},
                    "agent": "build",
                    "model": {"id": "dummy_model", "providerID": "nemo_gym"},
                    "content": [{"type": "text", "text": "DONE"}],
                    "finish": "stop",
                    "tokens": {"input": 20, "output": 1, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                },
            ),
            ("msg_4", "idle", 5, {"time": {"created": 12}, "outcome": "succeeded"}),
        ]
        db.executemany(
            "insert into session_message values (?, 'ses_1', ?, ?, ?, ?)",
            [(i, kind, seq, data["time"]["created"], json.dumps(data)) for i, kind, seq, data in rows],
        )
        db.commit()


def test_v2_store_is_read_through_the_v1_tables(tmp_path: Path) -> None:
    store = tmp_path / "opencode.db"
    write_v2_store(store)
    assert is_v2_store(store)
    assert stored_sessions(store) == [{"id": "ses_1", "parent_id": None}]
    assert stored_message_ids(store) == ["msg_1", "msg_2", "msg_3"]

    snapshot(store, tmp_path, harness_session_id=None, previous_message_ids=[])
    export = json.loads((tmp_path / "export.json").read_text())
    assert export["session_id"] == "ses_1"
    assert [message["info"]["role"] for message in export["messages"]] == ["user", "assistant", "assistant"]
    tool = export["messages"][1]["parts"][2]
    assert tool["tool"] == "shell" and tool["callID"] == "call_1"
    assert tool["state"]["output"] == "(no output)"
    assert tool["state"]["time"] == {"start": 5, "end": 6}

    items, usage = parse_opencode_session(tmp_path / "observations.db")
    assert [item.type for item in items] == ["function_call", "function_call_output", "message"]
    assert items[-1].content[0].text == "DONE"
    # 2.x, like 1.17.11, stores input without cache and output without reasoning.
    assert usage == {"input_tokens": 33, "output_tokens": 4, "reasoning_tokens": 1}
