# BFCL v4 — AST family

13 single-turn function-calling splits from Berkeley Function Calling
Leaderboard v4 ([gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html)):

- non-live AST: `simple_python`, `simple_java`, `simple_javascript`,
  `parallel`, `multiple`, `parallel_multiple`, `irrelevance`
- live AST: `live_simple`, `live_multiple`, `live_parallel`,
  `live_parallel_multiple`, `live_irrelevance`, `live_relevance`

Each row carries `verifier_metadata.test_category` so the resource server
dispatches to the right BFCL grader. Rollouts go through `bfcl_v4_ast_agent`,
which formats the prompt via HF `apply_chat_template(messages, tools=...)`,
sends to the model's text-completions endpoint, and parses tool calls with
BFCL's per-model FC handler (`Qwen/Qwen3-8B-FC` for Qwen3 family) — matching
NeMo Skills' `BFCLGenerationTask` parsing path.

## Example usage

```bash
# Prepare benchmark data (clones Gorilla, runs BFCL preprocessing,
# writes data/bfcl_v4_ast_benchmark.jsonl).
ng_prepare_benchmark "+config_paths=[benchmarks/bfcl_v4_ast/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/bfcl_v4_ast/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=bfcl_v4_ast_benchmark_agent \
    +input_jsonl_fpath=benchmarks/bfcl_v4_ast/data/bfcl_v4_ast_benchmark.jsonl \
    +output_jsonl_fpath=results/bfcl_v4_ast_rollouts.jsonl \
    +num_repeats=4
```

The vLLM server must be started without `--tool-call-parser` /
`--enable-auto-tool-choice` — BFCL's per-model FC handler does the
tool-call extraction, and a vLLM-side parser would fight with it.

A reasoning parser (`--reasoning-parser deepseek_r1`) is recommended
for any reasoning-emitting model so `<think>…</think>` is stripped
server-side before the FC handler runs.
