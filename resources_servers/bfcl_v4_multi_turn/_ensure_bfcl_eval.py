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
    # memory_vector.py imports sentence_transformers → sklearn.
    "scikit-learn",
]


def _pip_install(stage_args: list[str]) -> None:
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


def _bfcl_eval_importable() -> bool:
    try:
        import bfcl_eval  # noqa: F401
        from bfcl_eval.constants.model_config import (  # noqa: F401
            local_inference_model_map,
        )
        from bfcl_eval.model_handler.local_inference.qwen_fc import (  # noqa: F401
            QwenFCHandler,
        )
    except (ModuleNotFoundError, ImportError):
        return False
    return True


def _extras_importable() -> bool:
    # Each EXTRA_RUNTIME_DEP gates a code path the registry / web_search
    # backend reaches at request time. Verify the pip name resolves to an
    # importable module so a freshly-added extra forces a top-up install
    # even when the resource server's persisted venv already has bfcl_eval.
    extra_module_names = ["cffi", "cryptography", "soundfile", "PIL", "ddgs", "sklearn"]
    for mod in extra_module_names:
        try:
            __import__(mod)
        except (ModuleNotFoundError, ImportError):
            return False
    return True


def ensure_bfcl_eval_installed() -> None:
    # Probe imports separately — bfcl_eval persists in the resource
    # server's lustre venv across runs, so the early-return path used to
    # silently skip newly-added EXTRA_RUNTIME_DEPS. Run each install only
    # when needed.
    if not _bfcl_eval_importable():
        LOG.info("Installing bfcl_eval at runtime (commit %s)", BFCL_GIT_COMMIT)
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "gorilla"
            subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
            subprocess.run(["git", "checkout", BFCL_GIT_COMMIT], check=True, cwd=str(repo_dir))
            # bfcl_eval alone — let it pull its own pinned deps
            # (google-genai==1.24.0, qwen-agent, anthropic, etc.). uv's
            # strict resolver rejects mixing those with our looser pins.
            _pip_install([str(repo_dir / BFCL_EVAL_SUBDIR)])
        LOG.info("bfcl_eval install complete")

    if not _extras_importable():
        LOG.info("Installing bfcl_eval extras: %s", EXTRA_RUNTIME_DEPS)
        _pip_install(list(EXTRA_RUNTIME_DEPS))
        LOG.info("bfcl_eval extras install complete")
