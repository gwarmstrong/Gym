# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeMo Skills Tools Resources Server.

This resources server provides:
- Stateful Python code execution via in-process multiprocessing (no sandbox)
- Verification delegation to math_with_judge
"""

import asyncio
import io
import json
import logging
import multiprocessing
import signal
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import SESSION_ID_KEY


logger = logging.getLogger(__name__)


# ============================================================
# In-process Python execution
# ============================================================


def _session_worker(child_conn, max_execution_time: int):
    """Runs forever in its own process, keeping globals between calls."""
    exec_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "__import__": __import__,
        },
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "pd": pd,
        "pandas": pd,
    }
    exec_locals = {}
    while True:
        msg = child_conn.recv()
        if msg["cmd"] == "exec":
            code = msg["code"]
            try:
                out, err, res = _run_code_in_existing_env(code, exec_globals, exec_locals, max_execution_time)
                child_conn.send({"ok": True, "out": out, "err": err, "res": res})
            except Exception as e:
                child_conn.send({"ok": False, "error": str(e)})
        elif msg["cmd"] == "close":
            break


def _run_code_in_existing_env(code, globals_d, locals_d, timeout_s):
    """Re-uses the same globals/locals dictionary between calls."""
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()

    def _handle_timeout(signum, frame):
        raise TimeoutError("code timed-out")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_s)
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, globals_d, locals_d)
            result = _get_last_expr_value(code, globals_d, locals_d)
    finally:
        signal.alarm(0)
    return stdout_capture.getvalue(), stderr_capture.getvalue(), result


def _get_last_expr_value(code: str, globals_dict: dict, locals_dict: dict):
    """Try to evaluate the last line as a bare expression and return its repr."""
    lines = code.strip().split("\n")
    if not lines:
        return None

    last_line = lines[-1].strip()

    if last_line.startswith(("print", "import", "from", "def", "class", "if", "for", "while", "try", "with")):
        return None

    try:
        return str(eval(last_line, globals_dict, locals_dict))
    except Exception:
        return None


class _SessionHandle:
    """Light wrapper around one long-lived worker process."""

    def __init__(self, max_execution_time: int):
        parent_conn, child_conn = multiprocessing.Pipe()
        self._conn = parent_conn
        self._proc = multiprocessing.Process(
            target=_session_worker,
            args=(child_conn, max_execution_time),
            daemon=True,
        )
        self._proc.start()
        self.last_used = time.time()

    def exec(self, code: str):
        self._conn.send({"cmd": "exec", "code": code})
        reply = self._conn.recv()
        self.last_used = time.time()
        if reply["ok"]:
            return reply["out"], reply["err"], reply["res"]
        raise RuntimeError(reply["error"])

    def close(self):
        try:
            self._conn.send({"cmd": "close"})
        except (BrokenPipeError, EOFError):
            pass
        self._proc.join(timeout=1)


# ============================================================
# Configuration
# ============================================================


class NSToolsConfig(BaseResourcesServerConfig):
    """Config for the NeMo Skills tools resources server."""

    # Default verifier (typically math_with_judge)
    default_verifier: str = "math_with_judge"

    # Map of verifier names to server references
    # At minimum, should include math_with_judge
    verifiers: Dict[str, ResourcesServerRef] = Field(default_factory=dict)

    # Max execution time per code cell (seconds)
    max_execution_time: int = 10

    # Verbose logging for tool execution timing (disabled by default)
    verbose_tool_logging: bool = False


# ============================================================
# Run/Verify Request/Response Models
# ============================================================


class NSToolsRunRequest(BaseRunRequest):
    """Run request that allows extra fields from the sample."""

    model_config = ConfigDict(extra="allow")

    # Per-sample verifier selection (optional, falls back to default_verifier)
    verifier_type: Optional[str] = None

    # Fields for math_with_judge verifier
    question: Optional[str] = None
    expected_answer: Optional[str] = None


class NSToolsVerifyRequest(NSToolsRunRequest, BaseVerifyRequest):
    pass


class NSToolsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    delegated_response: Optional[Dict[str, Any]] = None

    # Timing metrics for tool execution
    total_tool_execution_time_seconds: float = 0.0
    num_tool_calls: int = 0
    avg_tool_call_time_seconds: float = 0.0
    tool_timeout_count: int = 0


# ============================================================
# Resources Server Implementation
# ============================================================


class NSToolsResourcesServer(SimpleResourcesServer):
    config: NSToolsConfig

    _sessions: Dict[str, _SessionHandle] = PrivateAttr(default_factory=dict)
    _timing_by_session: Dict[str, list] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/stateful_python_code_exec")(self.execute_tool)
        app.post("/end_session")(self.end_session)
        return app

    async def execute_tool(self, request: Request) -> PlainTextResponse:
        """
        Execute Python code in a stateful per-session worker process.

        Uses the nemo-gym session ID to maintain state across tool calls.
        Returns the result as plain text for simple_agent compatibility.
        Tracks execution timing and timeout detection per session.
        """
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.warning(f"No session ID found, using fallback: {session_id}")

        if session_id not in self._timing_by_session:
            self._timing_by_session[session_id] = []

        body = await request.json()
        code = body.get("code", "")

        start_time = time.perf_counter()
        is_timeout = False

        try:
            if session_id not in self._sessions:
                self._sessions[session_id] = _SessionHandle(self.config.max_execution_time)
            handle = self._sessions[session_id]

            loop = asyncio.get_running_loop()
            stdout, stderr, result = await loop.run_in_executor(None, handle.exec, code)

            response_data = {
                "success": True,
                "stdout": stdout,
                "stderr": stderr,
                "result": result,
            }
        except TimeoutError as e:
            is_timeout = True
            response_data = {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error_message": str(e),
            }
        except Exception as e:
            response_data = {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error_message": str(e),
            }

        elapsed = time.perf_counter() - start_time
        self._timing_by_session[session_id].append(
            {
                "tool_name": "stateful_python_code_exec",
                "execution_time_seconds": elapsed,
                "is_internal_timeout": is_timeout,
            }
        )
        if self.config.verbose_tool_logging:
            timeout_info = " [TIMEOUT]" if is_timeout else ""
            logger.info(f"Tool executed in {elapsed:.3f}s{timeout_info} (session={session_id[:8]}...)")

        return PlainTextResponse(json.dumps(response_data))

    async def end_session(self, request: Request) -> PlainTextResponse:
        """Clean up a session's worker process."""
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id and session_id in self._sessions:
            self._sessions[session_id].close()
            del self._sessions[session_id]
        return PlainTextResponse(json.dumps({"success": True}))

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    def _aggregate_timing_metrics(self, session_id: Optional[str]) -> Dict[str, Any]:
        """Aggregate tool execution timing metrics for a session."""
        tool_timings = self._timing_by_session.pop(session_id, []) if session_id else []

        total_tool_time = sum(t["execution_time_seconds"] for t in tool_timings)
        num_tool_calls = len(tool_timings)
        avg_tool_time = total_tool_time / num_tool_calls if num_tool_calls > 0 else 0.0
        tool_timeout_count = sum(1 for t in tool_timings if t.get("is_internal_timeout"))

        return {
            "total_tool_execution_time_seconds": total_tool_time,
            "num_tool_calls": num_tool_calls,
            "avg_tool_call_time_seconds": avg_tool_time,
            "tool_timeout_count": tool_timeout_count,
        }

    async def verify(self, request: Request, body: NSToolsVerifyRequest) -> NSToolsVerifyResponse:
        """
        Verify the model's response by delegating to the configured verifier.

        The verifier is selected by:
        1. Per-sample `verifier_type` field (if present)
        2. Config `default_verifier` (fallback)

        Always aggregates and returns tool execution timing metrics for this session.
        Detailed per-call and summary logging is controlled by verbose_tool_logging.
        """
        session_id = request.session.get(SESSION_ID_KEY) if request else None
        metrics = self._aggregate_timing_metrics(session_id)

        # Clean up session worker
        if session_id and session_id in self._sessions:
            self._sessions[session_id].close()
            del self._sessions[session_id]

        if self.config.verbose_tool_logging:
            logger.info(
                f"Session {session_id[:8] if session_id else 'unknown'}... metrics: "
                f"{metrics['num_tool_calls']} tool calls, total={metrics['total_tool_execution_time_seconds']:.3f}s, "
                f"avg={metrics['avg_tool_call_time_seconds']:.3f}s, "
                f"timeouts={metrics['tool_timeout_count']}"
            )

        # Select verifier
        verifier_type = body.verifier_type or self.config.default_verifier

        if verifier_type not in self.config.verifiers:
            raise ValueError(
                f"Unknown verifier: {verifier_type}. Configure it in 'verifiers' or check 'default_verifier'."
            )

        verifier_ref = self.config.verifiers[verifier_type]

        # Delegate to the verifier
        response = await self.server_client.post(
            server_name=verifier_ref.name,
            url_path="/verify",
            json=body.model_dump(),
        )

        result = await response.json()

        # Hard fail if no reward in response
        if "reward" not in result:
            raise ValueError(f"Verifier did not return 'reward' field. Response: {result}")

        return NSToolsVerifyResponse(
            **body.model_dump(),
            reward=result["reward"],
            delegated_response=result,
            **metrics,
        )


if __name__ == "__main__":
    NSToolsResourcesServer.run_webserver()
