# bfcl_v4_ast

BFCL v4 AST family resource server. Wraps the upstream `bfcl_eval`
package's AST checker — same package NeMo Skills uses
(`nemo_skills/evaluation/evaluator/bfcl.py::eval_bfcl`).

Per-rollout `verify()` records the agent's parsed tool calls and
metadata; actual grading runs in batch in `compute_metrics()` via one
`python -m bfcl_eval evaluate` subprocess per `test_category`. Per-rollout
reward in the JSONL is a placeholder (0.0) — read `pass@k/accuracy` from
the metrics file for real scores.

The server's `model_handler` config field MUST match the FC handler the
agent uses for parsing; otherwise the structured-call format the grader
expects will diverge from what the agent emits.

## Example usage

```bash
# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/bfcl_v4_ast/configs/bfcl_v4_ast.yaml,\
responses_api_agents/bfcl_v4_ast_agent/configs/bfcl_v4_ast_agent.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=bfcl_v4_ast_agent \
    +input_jsonl_fpath=resources_servers/bfcl_v4_ast/data/example.jsonl \
    +output_jsonl_fpath=results/bfcl_v4_ast_rollouts.jsonl \
    +num_repeats=1
```

Start the vLLM server WITHOUT `--tool-call-parser` and WITHOUT
`--enable-auto-tool-choice` — the agent's BFCL FC handler does the
tool-call extraction. A reasoning parser
(`--reasoning-parser deepseek_r1` for Qwen3 / Nemotron-3 / DeepSeek-R1
families) is recommended so `<think>…</think>` is stripped server-side
before the FC handler runs.
