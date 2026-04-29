# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Prepare BFCL v4 AST family data.

Pulls the upstream Gorilla repo, walks the AST scoring categories
(simple_python, simple_java, simple_javascript, parallel, multiple,
parallel_multiple, irrelevance, live_*), runs BFCL's per-category
preprocessing, and writes ONE consolidated JSONL covering all 13 splits.
Each row carries a `test_category` field so the resource server can
dispatch to the right BFCL grader.
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path


LOG = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "bfcl_v4_ast_benchmark.jsonl"

# Pinned Gorilla commit — must match Skills (nemo_skills/dataset/bfcl_v3/prepare.py).
REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
BFCL_GIT_COMMIT = "86d0374d0db52623c5092a73f82c22b87b7e9a25"
BFCL_EVAL_SUBDIR = "berkeley-function-call-leaderboard"
BFCL_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cpu"
DATA_FOLDER_PATH = Path("berkeley-function-call-leaderboard/bfcl_eval/data")

AST_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "multiple",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]


def _ensure_bfcl_eval_installed() -> None:
    try:
        import bfcl_eval  # noqa: F401

        return
    except (ModuleNotFoundError, ImportError):
        LOG.info("Installing bfcl_eval at runtime from pinned commit %s", BFCL_GIT_COMMIT)
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "gorilla"
            subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
            subprocess.run(["git", "checkout", BFCL_GIT_COMMIT], check=True, cwd=str(repo_dir))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    str(repo_dir / BFCL_EVAL_SUBDIR),
                    "--extra-index-url",
                    BFCL_EXTRA_INDEX_URL,
                ],
                check=True,
            )


def _load_category(target_folder: Path, category: str) -> list[dict]:
    """Replicate Skills' load_dataset_entry for AST categories.

    Same fields the BFCL grader and FC handler rely on:
      id, question, function, tools, single_turn (always True for AST).
    """
    from bfcl_eval.utils import (  # noqa
        is_format_sensitivity,
        is_memory,
        is_web_search,
        load_file,
        populate_test_cases_with_predefined_functions,
        process_agentic_test_case,
    )
    from nemo_skills.dataset.bfcl_v3.utils import (
        convert_to_tool,
        func_doc_language_specific_pre_processing,
    )

    if is_format_sensitivity(category) or is_web_search(category) or is_memory(category):
        raise ValueError(f"{category} is not an AST category")

    file_name = f"BFCL_v4_{category}.json"
    entries = load_file(target_folder / file_name)
    entries = process_agentic_test_case(entries)
    entries = populate_test_cases_with_predefined_functions(entries)
    for instance in entries:
        instance["single_turn"] = True
        if "function" in instance:
            instance["function"] = func_doc_language_specific_pre_processing(instance["function"], category)
            instance["tools"] = convert_to_tool(instance["function"])
    return entries


def _to_gym_row(entry: dict, category: str) -> dict:
    """Convert a Skills-style BFCL row into a Gym JSONL row.

    The agent re-applies the HF chat template + tools client-side, so
    `responses_create_params.input` is left empty and the fields the
    agent needs (question messages, tools, single_turn) ride in
    `verifier_metadata`.
    """
    return {
        "responses_create_params": {"input": []},
        "verifier_metadata": {
            "id": entry["id"],
            "test_category": category,
            "single_turn": entry["single_turn"],
            "question": entry["question"],
            "function": entry.get("function", []),
            "tools": entry.get("tools", []),
        },
    }


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_bfcl_eval_installed()

    with tempfile.TemporaryDirectory() as tmp:
        LOG.info("Cloning Gorilla repo to %s", tmp)
        subprocess.run(
            ["git", "clone", "--depth=1", REPO_URL, tmp],
            check=True,
            capture_output=True,
        )
        target_folder = Path(tmp) / DATA_FOLDER_PATH
        if not target_folder.exists():
            raise FileNotFoundError(f"BFCL data folder missing in {REPO_URL}: {DATA_FOLDER_PATH}")

        with OUTPUT_FPATH.open("w") as f_out:
            count_per_cat: dict[str, int] = {}
            for category in AST_CATEGORIES:
                entries = _load_category(target_folder, category)
                for e in entries:
                    f_out.write(json.dumps(_to_gym_row(e, category)) + "\n")
                count_per_cat[category] = len(entries)

    total = sum(count_per_cat.values())
    LOG.info("Wrote %d AST rows to %s. Per-category counts: %s", total, OUTPUT_FPATH, count_per_cat)
    return OUTPUT_FPATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare()
