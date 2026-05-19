"""Merge AAI sub-benchmark JSONLs into one rollout-ready file.

Run AFTER `ng_prepare_benchmark "+config_paths=[benchmarks/aai/config.yaml]"` has
produced each sub-benchmark's data JSONL.

For each AAI sub-benchmark:
  1. Read its prepared JSONL.
  2. Apply the sub's prompt template (AAI override path if any) to build
     `responses_create_params.input` on every row.
  3. Tag every row with `agent_ref: {name: <sub_agent_name>}`.
  4. Append to the merged JSONL.

The merged JSONL is the single input to `ng_collect_rollouts`. Per-sub rollout
counts come from the dict-form `+num_repeats={agent_name: N, ...}` CLI flag
(PR #1356).
"""

import json
from pathlib import Path

from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config


GYM_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FPATH = DATA_DIR / "aai_merged.jsonl"

# Maximum rows to keep per sub-benchmark in the merged output (before
# per-agent num_repeats fan-out at rollout time). Set via env
# `AAI_PER_SUB_LIMIT`; 0/unset = keep all. Used to fit a balanced parity
# run inside the 4h cluster slot.
PER_SUB_LIMIT_ENV = "AAI_PER_SUB_LIMIT"


# (sub_short_name, agent_name, jsonl_relpath_from_gym_root, prompt_relpath_from_gym_root_or_None)
AAI_SUBS = [
    (
        "mmlu_pro",
        "mmlu_pro_mcqa_simple_agent",
        "benchmarks/mmlu_pro/data/mmlu_pro_benchmark.jsonl",
        "benchmarks/mmlu_pro/prompts/default.yaml",
    ),
    (
        "hle",
        "hle_equivalence_llm_judge_simple_agent",
        "benchmarks/hle/data/hle_benchmark.jsonl",
        "benchmarks/hle/prompts/default.yaml",
    ),
    (
        "gpqa",
        "gpqa_mcqa_simple_agent",
        "benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl",
        "benchmarks/gpqa/prompts/default.yaml",
    ),
    (
        "aime25",
        "aime25_math_with_judge_simple_agent",
        "benchmarks/aime25/data/aime25_benchmark.jsonl",
        "benchmarks/aai/prompts/math.yaml",
    ),
    (
        "livecodebench",
        "livecodebench_v5_2407_2412_code_gen_simple_agent",
        "benchmarks/livecodebench/v5_2407_2412/data/livecodebench_v5_2407_2412_validation.jsonl",
        "benchmarks/aai/prompts/livecodebench.yaml",
    ),
    (
        "ifbench",
        "ifbench_benchmark_simple_agent",
        "benchmarks/ifbench/data/ifbench_benchmark.jsonl",
        "benchmarks/ifbench/prompts/default.yaml",
    ),
    (
        "aalcr",
        "aalcr_benchmark_simple_agent",
        "benchmarks/aalcr/data/aalcr_benchmark.jsonl",
        None,
    ),
]


def merge() -> Path:
    import os

    per_sub_limit = int(os.environ.get(PER_SUB_LIMIT_ENV, "0") or "0")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    # Load each sub fully (capped if per_sub_limit > 0), then INTERLEAVE
    # round-robin so the merged file mixes all 7 agents from the head. This
    # matters when ng_collect_rollouts processes rows in batches: a
    # mmlu_pro-first dump would have us spend hours before any aalcr/aime25
    # rollout completes.
    per_sub_rows: dict = {}
    for short, agent_name, jsonl_rel, prompt_rel in AAI_SUBS:
        jsonl_fpath = GYM_ROOT / jsonl_rel
        if not jsonl_fpath.exists():
            raise FileNotFoundError(
                f"Sub-benchmark {short} JSONL missing at {jsonl_fpath}. "
                f"Run `ng_prepare_benchmark +config_paths=[benchmarks/aai/config.yaml]` first."
            )
        prompt_cfg = load_prompt_config(prompt_rel) if prompt_rel else None
        rows = []
        with open(jsonl_fpath) as fin:
            for line in fin:
                row = json.loads(line)
                if prompt_cfg is not None:
                    row = apply_prompt_to_row(row, prompt_cfg)
                row["agent_ref"] = {"name": agent_name}
                rows.append(row)
                if per_sub_limit and len(rows) >= per_sub_limit:
                    break
        per_sub_rows[short] = (agent_name, rows)
        counts[short] = len(rows)
        print(f"[merge] {short:18s} → {agent_name:60s} : {len(rows)} rows")

    # Round-robin interleave
    written = 0
    with open(OUTPUT_FPATH, "w") as fout:
        max_len = max(len(r) for _, r in per_sub_rows.values())
        for i in range(max_len):
            for short, (_, rows) in per_sub_rows.items():
                if i < len(rows):
                    fout.write(json.dumps(rows[i]) + "\n")
                    written += 1
    print(f"[merge] Wrote interleaved merged JSONL: {OUTPUT_FPATH} (total {written} rows)")
    return OUTPUT_FPATH


if __name__ == "__main__":
    merge()
