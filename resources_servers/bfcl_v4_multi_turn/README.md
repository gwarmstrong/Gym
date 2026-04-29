# bfcl_v4_multi_turn

Resource server shared by the BFCL v4 multi_turn, memory, and web_search
benchmarks. Wraps `bfcl_eval`'s multi-turn / memory / web-search graders;
per-rollout `verify()` records the agent's per-turn `generation` shape
and `compute_metrics()` shells out to `python -m bfcl_eval evaluate`
once per `test_category`.

The agent (`bfcl_v4_multi_turn_agent`) handles the per-turn execute-and-
feed-back loop, prereq pre-pass for memory, and DuckDuckGo web search.

## Example usage

```bash
# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/bfcl_v4_multi_turn/configs/bfcl_v4_multi_turn.yaml,\
responses_api_agents/bfcl_v4_multi_turn_agent/configs/bfcl_v4_multi_turn_agent.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=bfcl_v4_multi_turn_agent \
    +input_jsonl_fpath=resources_servers/bfcl_v4_multi_turn/data/example.jsonl \
    +output_jsonl_fpath=results/bfcl_v4_multi_turn_rollouts.jsonl \
    +num_repeats=1
```

Start the vLLM server WITHOUT `--tool-call-parser` and WITHOUT
`--enable-auto-tool-choice` — the agent's BFCL FC handler does the
tool-call extraction and a vLLM-side parser would fight with it.
A reasoning parser (`--reasoning-parser deepseek_r1`) is recommended
so `<think>…</think>` is stripped server-side before the FC handler runs.
