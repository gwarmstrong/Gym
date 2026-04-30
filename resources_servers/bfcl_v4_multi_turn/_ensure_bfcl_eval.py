# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotent runtime install of bfcl_eval into the current Python's venv.

Each rollout starts with `uv sync` (in the run-cmd wrapper) which strips
anything not declared in pyproject.toml — and bfcl_eval is intentionally
NOT declared (it conflicts with pinned core deps in the Gym tree). This
helper runs at agent / resource-server startup AFTER `uv sync` has
finished, restoring bfcl_eval before any runtime use.

Pinned to the same Gorilla commit Skills uses
(nemo_skills/dataset/bfcl_v3/prepare.py).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path


LOG = logging.getLogger(__name__)

REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
BFCL_GIT_COMMIT = "86d0374d0db52623c5092a73f82c22b87b7e9a25"
BFCL_EVAL_SUBDIR = "berkeley-function-call-leaderboard"
BFCL_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cpu"

# Gym-container extras. Agent parser-build avoids the registry entirely
# by importing QwenFCHandler directly, BUT the resource server's
# compute_metrics() shells out to `python -m bfcl_eval evaluate`, which
# is the CLI entrypoint that DOES import the full registry. So the
# resource-server-side install still needs the registry's transitive
# deps. Keep the list minimal — only deps that have actually surfaced
# during cluster probes.
EXTRA_RUNTIME_DEPS = [
    "cffi>=1.17",
    "cryptography>=43",
    "soundfile",  # qwen_agent.llm.base
    "Pillow",  # qwen_agent.tools.image_zoom_in_qwen3vl
    # bfcl_eval grader replays the web_search backend on scored rows
    # (see _evaluate_single_agentic_entry → _load_scenario), so the
    # resource-server venv needs the same backend dep as the agent venv.
    "ddgs",
]


def ensure_bfcl_eval_installed() -> None:
    # Probe ALL imports the runtime needs:
    #   - bfcl_eval itself
    #   - QwenFCHandler (used directly by agents, bypasses the registry)
    #   - bfcl_eval.constants.model_config (the CLI subprocess uses this)
    #     — touches qwen_agent which transitively imports PIL/soundfile,
    #     so the import_module success implies all those deps are present.
    try:
        import bfcl_eval  # noqa: F401
        from bfcl_eval.constants.model_config import (  # noqa: F401
            local_inference_model_map,
        )
        from bfcl_eval.model_handler.local_inference.qwen_fc import (  # noqa: F401
            QwenFCHandler,
        )

        return
    except (ModuleNotFoundError, ImportError):
        pass

    LOG.info("Installing bfcl_eval at runtime (commit %s)", BFCL_GIT_COMMIT)
    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / "gorilla"
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
        subprocess.run(["git", "checkout", BFCL_GIT_COMMIT], check=True, cwd=str(repo_dir))
        # Two-stage install:
        #   1. bfcl_eval alone — let it pull its own pinned deps
        #      (google-genai==1.24.0, qwen-agent, anthropic, etc.). uv's
        #      strict resolver rejects mixing those with our looser pins
        #      from SKILLS_BFCL_REQUIREMENTS.
        #   2. Top up Gym-container extras (cffi/cryptography/soundfile)
        #      separately. These don't conflict with bfcl_eval's pins.
        for stage_args in (
            [str(repo_dir / BFCL_EVAL_SUBDIR)],
            list(EXTRA_RUNTIME_DEPS),
        ):
            uv_cmd = [
                "uv",
                "pip",
                "install",
                "--no-cache-dir",
                "--python",
                sys.executable,
                *stage_args,
                "--extra-index-url",
                BFCL_EXTRA_INDEX_URL,
            ]
            try:
                subprocess.run(uv_cmd, check=True)
            except FileNotFoundError:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-cache-dir",
                        *stage_args,
                        "--extra-index-url",
                        BFCL_EXTRA_INDEX_URL,
                    ],
                    check=True,
                )
    LOG.info("bfcl_eval install complete")
