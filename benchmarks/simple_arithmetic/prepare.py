# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Prepare a small synthetic arithmetic benchmark.

A deliberately Gym-only smoke benchmark: 20 deterministic two-operand
arithmetic questions exercising the `+`, `-`, `*` operations on
two-digit operands. Reuses the existing `math_with_judge` resource
server — no new server, no HF download, no Skills counterpart.

Used as the end-to-end test fixture for the Gym-only `ns eval --backend=gym`
path (see NeMo Skills' `eval_gym.py` dispatcher). The expected answers are
unambiguous, so parity with any sane policy model is trivially observable:
on a reasoning-capable model every row should score reward=1.0.
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "simple_arithmetic_benchmark.jsonl"


def prepare() -> Path:
    """Emit 20 deterministic arithmetic problems."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Deterministic across runs — a fixed seed-equivalent pattern using a
    # small set of operands so every (a, op, b) triple is reproducible.
    operands = [(12, 3), (47, 83), (15, 27), (64, 19), (8, 7), (33, 11), (100, 25), (9, 9), (50, 50), (17, 23)]
    ops = ["+", "-", "*", "+", "*", "-", "+", "*", "-", "+"]
    pairs = list(zip(operands, ops))

    def _eval(a: int, b: int, op: str) -> int:
        return {"+": a + b, "-": a - b, "*": a * b}[op]

    rows = []
    for (a, b), op in pairs:
        rows.append(
            {
                "question": f"What is {a} {op} {b}? Reply with just the integer.",
                "expected_answer": str(_eval(a, b, op)),
            }
        )
        # Same operands, swapped order — doubles the count to 20.
        rows.append(
            {
                "question": f"What is {b} {op} {a}? Reply with just the integer.",
                "expected_answer": str(_eval(b, a, op)),
            }
        )

    with open(OUTPUT_FPATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT_FPATH}")
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 rows, got {len(rows)}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
