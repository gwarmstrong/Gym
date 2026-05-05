# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BFCL v4 memory family data.

3 splits: memory_kv, memory_vector, memory_rec_sum. Memory tasks have
a *prereq* concept: each memory test entry has zero or more "_prereq_<n>"
sibling rows that must be executed (in order) before the scored row,
to populate stateful memory. We emit prereq rows interleaved with
scored rows, ordered the way Skills' BFCLGenerationTask::load_data
orders them — prereqs first (sorted by their numeric suffix), then the
scored rows.

The agent's run() detects memory categories and:
  1. for prereq rows, runs the multi-turn loop and lets the
     MemoryAPI._flush_memory_to_local_file() hook persist state.
  2. for scored rows, runs the multi-turn loop reading from the
     populated state and submits to verify().

This mirrors NeMo Skills' memory pipeline exactly.
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
OUTPUT_FPATH = DATA_DIR / "bfcl_v4_memory_benchmark.jsonl"

REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
BFCL_GIT_COMMIT = "86d0374d0db52623c5092a73f82c22b87b7e9a25"
BFCL_EVAL_SUBDIR = "berkeley-function-call-leaderboard"
BFCL_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cpu"
DATA_FOLDER_PATH = Path("berkeley-function-call-leaderboard/bfcl_eval/data")

MEMORY_CATEGORIES = ["memory_kv", "memory_vector", "memory_rec_sum"]


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


def _load_memory_category(target_folder: Path, category: str) -> list[dict]:
    """Mirror Skills' load_dataset_entry for memory categories."""
    from bfcl_eval.constants.category_mapping import MEMORY_SCENARIO_NAME
    from bfcl_eval.utils import (
        load_file,
        populate_initial_settings_for_memory_test_cases,
        populate_test_cases_with_predefined_functions,
        process_agentic_test_case,
        process_memory_test_case,
    )
    from nemo_skills.dataset.bfcl_v3.utils import (
        convert_to_tool,
        func_doc_language_specific_pre_processing,
    )

    all_entries = load_file(target_folder / "BFCL_v4_memory.json")
    for scenario in MEMORY_SCENARIO_NAME:
        all_entries = process_memory_test_case(all_entries, category, scenario, include_prereq=True)
    all_entries = process_agentic_test_case(all_entries)
    all_entries = populate_test_cases_with_predefined_functions(all_entries)
    all_entries = populate_initial_settings_for_memory_test_cases(all_entries, str(target_folder))

    for instance in all_entries:
        instance["single_turn"] = False
        if "function" in instance:
            instance["function"] = func_doc_language_specific_pre_processing(instance["function"], category)
            instance["tools"] = convert_to_tool(instance["function"])
    return all_entries


def _is_prereq(entry_id: str) -> bool:
    return "_prereq_" in entry_id


def _prereq_sort_key(entry_id: str) -> int:
    suffix = entry_id.split("_prereq_")[1].split("-")[0]
    return int(suffix)


def _to_gym_row(entry: dict, category: str) -> dict:
    return {
        "responses_create_params": {"input": []},
        "verifier_metadata": {
            "id": entry["id"],
            "test_category": category,
            "single_turn": False,
            "is_prereq": _is_prereq(entry["id"]),
            "scenario": entry.get("scenario", ""),
            "question": entry["question"],
            "function": entry.get("function", []),
            "tools": entry.get("tools", []),
            "initial_config": entry.get("initial_config", {}),
            "involved_classes": entry.get("involved_classes", []),
            # `depends_on` is BFCL's per-row list of prereq ids that must
            # populate memory state before the scored row runs. The agent
            # needs this to await prereq completion at runtime; without
            # it the async rollout client will fire scored rollouts
            # before their prereqs land their _flush_memory_to_local_file()
            # output.
            "depends_on": entry.get("depends_on", []),
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
            for category in MEMORY_CATEGORIES:
                entries = _load_memory_category(target_folder, category)
                # Skills' ordering: all prereqs first (sorted), then scored.
                prereqs = sorted(
                    [e for e in entries if _is_prereq(e["id"])],
                    key=lambda e: _prereq_sort_key(e["id"]),
                )
                scored = [e for e in entries if not _is_prereq(e["id"])]
                ordered = prereqs + scored
                for e in ordered:
                    f_out.write(json.dumps(_to_gym_row(e, category)) + "\n")
                counts[category] = len(ordered)

    LOG.info("Wrote %d memory rows. Per-category: %s", sum(counts.values()), counts)
    return OUTPUT_FPATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare()
