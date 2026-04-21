# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Prepare SWE-bench Pro benchmark data.

Downloads ScaleAI/SWE-bench_Pro from HuggingFace and writes a single
benchmark JSONL in the shape the Gym swe_agents agent expects. Matches
Skills' nemo_skills/dataset/swe-bench-pro/prepare.py:
  - Remaps repo_language → language using the multilingual code set.
  - Appends "Requirements:" and "New interfaces introduced:" blocks to
    problem_statement.
  - Builds a per-row container_formatter pointing to the jefzda/sweap-images
    docker registry tag.

Skills additionally splits into Alpine vs Ubuntu files because its host
nemo-skills container must match the instance's libc. For the Gym port we
keep a single JSONL — the apptainer-in-apptainer step here uses the
swebench_pro_openhands.yaml agent container, not the host OS-specific
nemo-skills container, so the split is unnecessary.
"""

import json
from pathlib import Path

import datasets


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "swe_bench_pro_benchmark.jsonl"

# Maps ScaleAI/SWE-bench_Pro repo_language values to the language names Skills'
# SWE-bench_Multilingual prompting expects. Kept identical to Skills.
LANGUAGE_MAP = {
    "js": "javascript",
    "ts": "typescript",
    "go": "go",
    "python": "python",
}

DATASET_NAME = "ScaleAI/SWE-bench_Pro"
SPLIT = "test"
CONTAINER_FORMATTER_TEMPLATE = "docker://jefzda/sweap-images:{dockerhub_tag}"


def _build_problem_statement(row: dict) -> str:
    """Match Skills' reformatted problem statement verbatim."""
    return (
        f"{row['problem_statement']}\n\n"
        f"Requirements:\n{row['requirements']}\n\n"
        f"New interfaces introduced:\n{row['interface']}"
    )


def prepare() -> Path:
    """Download and prepare SWE-bench Pro data. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(path=DATASET_NAME, split=SPLIT)

    count = 0
    with OUTPUT_FPATH.open("w") as f:
        for row in dataset:
            language = LANGUAGE_MAP[row["repo_language"]]
            problem_statement = _build_problem_statement(row)
            container_formatter = CONTAINER_FORMATTER_TEMPLATE.format(dockerhub_tag=row["dockerhub_tag"])

            # The swe_agents agent reads problem_info from body.metadata. It also
            # needs instance_dict as a JSON-encoded string (mounted as the
            # per-instance dataset file inside the apptainer container). Bake
            # the full row into instance_dict so the eval harness has everything
            # it needs (patch, test_patch, fail_to_pass, repo, base_commit, etc.).
            instance_dict = {
                **{k: v for k, v in row.items() if k not in ("repo_language", "interface", "requirements")},
                "problem_statement": problem_statement,
                "language": language,
            }

            metadata = {
                "instance_id": row["instance_id"],
                "dataset_name": DATASET_NAME,
                "split": SPLIT,
                "problem_statement": problem_statement,
                "language": language,
                "container_formatter": container_formatter,
                "container_repo_dir": "/app",
                "instance_dict": json.dumps(instance_dict),
            }

            entry = {
                "responses_create_params": {
                    # Empty input: swe_agents constructs prompts internally via OpenHands
                    # using the problem_statement from metadata. Matches the shape of
                    # responses_api_agents/swe_agents/data/example.jsonl.
                    "input": [],
                    "metadata": metadata,
                },
                # Keep top-level copies for compatibility with existing JSONL consumers.
                "instance_id": row["instance_id"],
                "problem_statement": problem_statement,
                "language": language,
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
