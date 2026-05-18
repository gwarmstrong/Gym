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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(OUTPUT_FPATH, "w") as fout:
        for short, agent_name, jsonl_rel, prompt_rel in AAI_SUBS:
            jsonl_fpath = GYM_ROOT / jsonl_rel
            if not jsonl_fpath.exists():
                raise FileNotFoundError(
                    f"Sub-benchmark {short} JSONL missing at {jsonl_fpath}. "
                    f"Run `ng_prepare_benchmark +config_paths=[benchmarks/aai/config.yaml]` first."
                )
            prompt_cfg = load_prompt_config(prompt_rel) if prompt_rel else None
            n = 0
            with open(jsonl_fpath) as fin:
                for line in fin:
                    row = json.loads(line)
                    if prompt_cfg is not None:
                        row = apply_prompt_to_row(row, prompt_cfg)
                    row["agent_ref"] = {"name": agent_name}
                    fout.write(json.dumps(row) + "\n")
                    n += 1
            counts[short] = n
            print(f"[merge] {short:18s} → {agent_name:60s} : {n} rows")
    print(f"[merge] Wrote merged JSONL: {OUTPUT_FPATH} (total {sum(counts.values())} rows)")
    return OUTPUT_FPATH


if __name__ == "__main__":
    merge()
