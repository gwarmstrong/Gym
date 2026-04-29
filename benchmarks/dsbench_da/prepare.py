# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Prepare DSBench-DA benchmark data for NeMo Gym.

Mirrors `nemo_skills/dataset/dsbench_da/prepare.py`:

  1. Downloads `liqiang888/DSBench` (`data_analysis/data.zip`) and the
     accompanying `data_analysis/data.json` task metadata from HuggingFace.
  2. Extracts task directories under
     `Gym/benchmarks/dsbench_da/data/extracted/<task_id>/`. Each task has
     an `introduction.txt`, one or more `<question>.txt` question files,
     and one or more Excel workbooks (.xlsx / .xlsb / .xlsm).
  3. Emits one row per question to
     `data/dsbench_da_benchmark.jsonl`. Each row has Skills' `problem` and
     `expected_answer` fields plus `excel_paths` (formatted as absolute
     in-container paths). At rollout time the prompt template
     interpolates `{problem}` and `{excel_paths}`.

The prompt's `excel_paths` placeholder is filled with absolute paths that
resolve inside the runtime container (`/opt/Gym/benchmarks/dsbench_da/...`).
The cluster mounts the per-recipe Gym worktree at `/opt/Gym`, so both the
agent's and the python sandbox's view of the file is the same string.
"""

import json
import zipfile
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import hf_hub_download


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
OUTPUT_FPATH = DATA_DIR / "dsbench_da_benchmark.jsonl"

# In-container path prefix the prompt should reference. The prepare job runs
# inside the same container family (nemo-rl), and the eval/rollout jobs see
# the same recipe-local Gym worktree at /opt/Gym, so absolute paths
# rooted at /opt/Gym/benchmarks/dsbench_da/data/extracted resolve in the
# python sandbox at rollout time.
DISPLAY_ROOT = Path("/opt/Gym/benchmarks/dsbench_da/data/extracted")


def _format_excel_paths(excel_files: Iterable[Path], actual_root: Path, display_root: Path) -> str:
    """Map on-disk paths to in-container absolute paths for the prompt."""
    formatted: List[str] = []
    for path in excel_files:
        try:
            rel = path.relative_to(actual_root)
            disp = display_root / rel
        except ValueError:
            disp = path
        formatted.append(str(disp))
    return " ".join(formatted)


def _gather_excel_files(task_dir: Path) -> List[Path]:
    excel_files: List[Path] = []
    for ext in ("*.xlsx", "*.xlsb", "*.xlsm"):
        excel_files.extend(task_dir.glob(ext))
    return [f for f in excel_files if "answer" not in f.name.lower()]


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not EXTRACTED_DIR.exists():
        print("Downloading DSBench from HuggingFace (liqiang888/DSBench)...")
        zip_path = Path(
            hf_hub_download(
                repo_id="liqiang888/DSBench",
                filename="data_analysis/data.zip",
                repo_type="dataset",
            )
        )
        print(f"Extracting {zip_path} -> {DATA_DIR} ...")
        EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # The zip contains a top-level "data/" directory; extract into DATA_DIR
            # and rename to "extracted" to keep the layout self-contained and not
            # collide with potential future siblings.
            zf.extractall(DATA_DIR)
        unpacked = DATA_DIR / "data"
        if not unpacked.exists():
            raise FileNotFoundError(f"Expected {unpacked} after unzip; layout changed?")
        # Move contents into extracted/ then remove the now-empty parent.
        for child in unpacked.iterdir():
            target = EXTRACTED_DIR / child.name
            if target.exists():
                continue
            child.rename(target)
        unpacked.rmdir()
    else:
        print(f"Using cached extracted DSBench data at {EXTRACTED_DIR}")

    metadata_path = Path(
        hf_hub_download(
            repo_id="liqiang888/DSBench",
            filename="data_analysis/data.json",
            repo_type="dataset",
        )
    )
    metadata: List[dict] = []
    with open(metadata_path, "r") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line.strip()))

    print(f"Processing {len(metadata)} tasks ...")
    rows = []
    for task in metadata:
        task_id = task["id"]
        task_dir = EXTRACTED_DIR / task_id
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory missing: {task_dir}")

        if len(task["answers"]) != len(task["questions"]):
            raise ValueError(
                f"Task {task_id}: mismatched questions ({len(task['questions'])}) and answers ({len(task['answers'])})."
            )

        intro_file = task_dir / "introduction.txt"
        introduction = intro_file.read_text(encoding="utf-8", errors="ignore") if intro_file.exists() else ""

        excel_files = _gather_excel_files(task_dir)
        excel_paths = _format_excel_paths(excel_files, actual_root=EXTRACTED_DIR, display_root=DISPLAY_ROOT)

        for idx, question_name in enumerate(task["questions"]):
            question_file = task_dir / f"{question_name}.txt"
            if not question_file.exists():
                print(f"  Warning: {task_id}/{question_name}.txt not found, skipping")
                continue

            question_text = question_file.read_text(encoding="utf-8", errors="ignore").strip()
            problem_text = ""
            if introduction:
                problem_text += f"The introduction is detailed as follows.\n{introduction}\n\n"
            problem_text += f"The question for this task is detailed as follows.\n{question_text}"

            row = {
                # Prompt template placeholders.
                "problem": problem_text,
                "excel_paths": excel_paths,
                # Verifier inputs (read directly by dsbench_da server).
                "question": question_text,
                "expected_answer": str(task["answers"][idx]),
                # Per-row metadata so verify response carries it through.
                "verifier_type": "dsbench_da",
                "task_id": task_id,
                "question_id": question_name,
                "task_name": task["name"],
                "task_url": task["url"],
                "task_year": task["year"],
            }
            rows.append(row)

    if not rows:
        raise ValueError("No DSBench-DA rows produced — extraction failure?")

    with open(OUTPUT_FPATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
