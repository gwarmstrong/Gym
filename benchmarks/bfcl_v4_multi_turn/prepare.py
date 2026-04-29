# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BFCL v4 multi_turn family data.

4 splits: multi_turn_base, multi_turn_miss_func, multi_turn_miss_param,
multi_turn_long_context. Per-turn execution loop happens inside the
agent (`bfcl_v4_multi_turn_agent`).
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
OUTPUT_FPATH = DATA_DIR / "bfcl_v4_multi_turn_benchmark.jsonl"

REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
BFCL_GIT_COMMIT = "86d0374d0db52623c5092a73f82c22b87b7e9a25"
BFCL_EVAL_SUBDIR = "berkeley-function-call-leaderboard"
BFCL_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cpu"
DATA_FOLDER_PATH = Path("berkeley-function-call-leaderboard/bfcl_eval/data")

MULTI_TURN_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]


def _ensure_bfcl_eval_installed() -> None:
    try:
        import bfcl_eval  # noqa: F401

        return
    except (ModuleNotFoundError, ImportError):
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
                        "--extra-index-url",
                        BFCL_EXTRA_INDEX_URL,
                    ],
                    check=True,
                )


def _load_category(target_folder: Path, category: str) -> list[dict]:
    """Mirror Skills' load_dataset_entry for multi_turn categories."""
    from bfcl_eval.utils import (
        load_file,
        populate_test_cases_with_predefined_functions,
        process_agentic_test_case,
    )
    from nemo_skills.dataset.bfcl_v3.utils import (
        convert_to_tool,
        func_doc_language_specific_pre_processing,
    )

    file_name = f"BFCL_v4_{category}.json"
    entries = load_file(target_folder / file_name)
    entries = process_agentic_test_case(entries)
    entries = populate_test_cases_with_predefined_functions(entries)
    for instance in entries:
        instance["single_turn"] = False
        if "function" in instance:
            instance["function"] = func_doc_language_specific_pre_processing(instance["function"], category)
            instance["tools"] = convert_to_tool(instance["function"])
    return entries


def _to_gym_row(entry: dict, category: str) -> dict:
    return {
        "responses_create_params": {"input": []},
        "verifier_metadata": {
            "id": entry["id"],
            "test_category": category,
            "single_turn": False,
            "question": entry["question"],
            "function": entry.get("function", []),
            "tools": entry.get("tools", []),
            "initial_config": entry.get("initial_config", {}),
            "involved_classes": entry.get("involved_classes", []),
            "missed_function": entry.get("missed_function", {}),
        },
    }


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_bfcl_eval_installed()

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth=1", REPO_URL, tmp],
            check=True,
            capture_output=True,
        )
        target_folder = Path(tmp) / DATA_FOLDER_PATH

        with OUTPUT_FPATH.open("w") as f_out:
            counts: dict[str, int] = {}
            for category in MULTI_TURN_CATEGORIES:
                entries = _load_category(target_folder, category)
                for e in entries:
                    f_out.write(json.dumps(_to_gym_row(e, category)) + "\n")
                counts[category] = len(entries)

    LOG.info("Wrote %d multi-turn rows. Per-category: %s", sum(counts.values()), counts)
    return OUTPUT_FPATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare()
