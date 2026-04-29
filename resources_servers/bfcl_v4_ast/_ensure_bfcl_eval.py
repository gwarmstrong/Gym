# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotent runtime install of bfcl_eval into the current Python's venv.

Each rollout starts with `uv sync` (in the run-cmd wrapper) which strips
anything not declared in pyproject.toml — and bfcl_eval is intentionally
NOT declared (it's heavy, transitive, and conflicts with pinned core
deps in the Gym tree). So this helper runs at agent / resource-server
startup AFTER `uv sync` has finished, restoring bfcl_eval before any
runtime use.

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

# bfcl_eval.constants.model_config eagerly imports every handler
# (Gemini, Anthropic, Cohere, ...) at module load. We only need local
# inference handlers but pay for all of them. cryptography is a
# transitive of google.auth required by the Gemini handler — bfcl_eval
# doesn't pull it via setup.py, so importing the module fails without it.
EXTRA_RUNTIME_DEPS = ["cffi>=1.17", "cryptography>=43"]


def ensure_bfcl_eval_installed() -> None:
    try:
        import _cffi_backend  # noqa: F401  # cryptography native backend
        import bfcl_eval  # noqa: F401

        # Smoke-import the failing path explicitly so we catch C-ext ABI
        # mismatches up front instead of on the first /run.
        from cryptography.hazmat.bindings._rust import exceptions  # noqa: F401

        return
    except (ModuleNotFoundError, ImportError):
        pass

    LOG.info("Installing bfcl_eval at runtime (commit %s)", BFCL_GIT_COMMIT)
    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / "gorilla"
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
        subprocess.run(["git", "checkout", BFCL_GIT_COMMIT], check=True, cwd=str(repo_dir))
        cmd = [
            "uv",
            "pip",
            "install",
            "--no-cache-dir",
            "--python",
            sys.executable,
            str(repo_dir / BFCL_EVAL_SUBDIR),
            *EXTRA_RUNTIME_DEPS,
            "--extra-index-url",
            BFCL_EXTRA_INDEX_URL,
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    str(repo_dir / BFCL_EVAL_SUBDIR),
                    *EXTRA_RUNTIME_DEPS,
                    "--extra-index-url",
                    BFCL_EXTRA_INDEX_URL,
                ],
                check=True,
            )
    LOG.info("bfcl_eval install complete")
