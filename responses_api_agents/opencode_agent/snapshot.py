# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export an OpenCode session database copied out of the sandbox after the CLI stopped."""

import json
import logging
import sqlite3
from collections.abc import Collection
from contextlib import closing
from pathlib import Path

from responses_api_agents.opencode_agent.store_v2 import is_v2_store, project_v1


LOG = logging.getLogger(__name__)


def snapshot(
    database: Path, directory: Path, *, harness_session_id: str | None, previous_message_ids: Collection[str]
) -> None:
    """Export root messages and retain the full session tree for observations."""
    previous_ids = set(previous_message_ids)
    if is_v2_store(database):
        # OpenCode 2.x keeps another schema; read it through the 1.x tables (see store_v2).
        projected = directory / "opencode-v1.db"
        project_v1(database, projected)
        database = projected
    # Close both connections even on errors; the caller removes the directory afterwards.
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as source:
        source.row_factory = sqlite3.Row
        sessions = source.execute(
            "select id from session where parent_id is null order by time_created, id"
        ).fetchall()
        if len(sessions) != 1:
            raise RuntimeError(f"Expected one OpenCode root session, found {len(sessions)}")
        rows = source.execute(
            "select id, data from message where session_id=? order by time_created, id", (sessions[0]["id"],)
        ).fetchall()
        session_id = sessions[0]["id"]
        if harness_session_id is not None and harness_session_id != session_id:
            raise RuntimeError(f"OpenCode resumed unexpected session {session_id!r}")
        messages = []
        for row in rows:
            if row["id"] in previous_ids:
                continue
            info = json.loads(row["data"])
            parts = [
                json.loads(part[0])
                for part in source.execute(
                    "select data from part where message_id=? order by time_created, id", (row["id"],)
                )
            ]
            messages.append({"id": row["id"], "info": info, "parts": parts})
        all_messages = source.execute("select id, data from message order by time_created, id").fetchall()
        usage_messages = [json.loads(row["data"]) for row in all_messages if row["id"] not in previous_ids]
        (directory / "export.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "message_ids": [row["id"] for row in all_messages],
                    "activation_message_ids": [row["id"] for row in all_messages if row["id"] not in previous_ids],
                    "messages": messages,
                    "usage_messages": usage_messages,
                }
            )
        )
        # A missing observation store costs only observation evidence, not the export.
        try:
            with closing(sqlite3.connect(directory / "observations.db")) as destination:
                source.backup(destination)
        except sqlite3.Error:
            LOG.exception("Failed to copy OpenCode observations")


def stored_sessions(database: Path) -> list[dict[str, str | None]]:
    """Every stored session's ID and parent ID, oldest first, as the owner's ``sessions.json`` lists them."""
    table = "session_v2" if is_v2_store(database) else "session"
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as source:
        rows = source.execute(f"select id, parent_id from {table} order by time_created, id").fetchall()  # noqa: S608
    return [{"id": session_id, "parent_id": parent_id} for session_id, parent_id in rows]


def stored_message_ids(database: Path) -> list[str]:
    """Every stored message's ID, oldest first."""
    table = "session_message where type in ('user', 'synthetic', 'assistant')" if is_v2_store(database) else "message"
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as source:
        return [row[0] for row in source.execute(f"select id from {table} order by time_created, id")]  # noqa: S608
