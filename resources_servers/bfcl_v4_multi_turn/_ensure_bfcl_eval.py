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

# Mirror Skills' BFCL_REQUIREMENTS in nemo_skills/inference/eval/bfcl.py.
# bfcl_eval/constants/model_config.py eagerly imports every handler at
# module load, so we need the SDKs / runtime deps even though we only
# use the local-inference Qwen handler at parse time. cryptography +
# cffi added on top because the Gym container lacks them and Gemini's
# import chain ends at google.auth -> cryptography.
SKILLS_BFCL_REQUIREMENTS = [
    "requests",
    "tqdm",
    "numpy==1.26.4",
    "pandas",
    "huggingface_hub",
    "pydantic>=2.8.2",
    "python-dotenv>=1.0.1",
    "tree_sitter==0.21.3",
    "tree-sitter-java==0.21.0",
    "tree-sitter-javascript==0.21.4",
    "openai>=1.86.0",
    "mistralai==1.7.0",
    "anthropic>=0.75.0",
    "cohere==5.18.0",
    "typer>=0.12.5",
    "tabulate>=0.9.0",
    "datamodel-code-generator==0.25.7",
    "google-genai>=1.52.0",
    "mpmath==1.3.0",
    "tenacity>=8.5.0",
    "writer-sdk>=2.1.0",
    "overrides",
    "boto3",
    "beautifulsoup4",
    "html2text",
    "rank_bm25==0.2.2",
    "google-search-results",
    "faiss-cpu==1.11.0",
    "networkx==3.3",
    "filelock==3.20.0",
]
# Gym-container extras that Skills' container has by default but we don't.
EXTRA_RUNTIME_DEPS = [
    "cffi>=1.17",
    "cryptography>=43",
    # qwen_agent.llm.base imports soundfile at module load; bfcl_eval
    # transitively imports qwen_agent. Skills has it via its broader
    # BFCL_REQUIREMENTS chain.
    "soundfile",
]


def ensure_bfcl_eval_installed() -> None:
    try:
        import _cffi_backend  # noqa: F401  # cryptography native backend
        import bfcl_eval  # noqa: F401

        # Probe the failing import chain end-to-end.
        from bfcl_eval.constants.model_config import (  # noqa: F401
            local_inference_model_map,
        )
        from cryptography.hazmat.bindings._rust import (  # noqa: F401
            exceptions as _rust_exc,
        )

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
            *SKILLS_BFCL_REQUIREMENTS,
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
                    *SKILLS_BFCL_REQUIREMENTS,
                    *EXTRA_RUNTIME_DEPS,
                    "--extra-index-url",
                    BFCL_EXTRA_INDEX_URL,
                ],
                check=True,
            )
    LOG.info("bfcl_eval install complete")
