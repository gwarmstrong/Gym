# dsbench_da

DSBench data-analysis verifier — a Skills-parity port of
`nemo_skills/evaluation/evaluator/dsbench.py::DSBenchEvaluator`.

The server only implements `verify()`. It does **not** execute python
itself — the agent's python tool calls go through the existing
`ns_tools` resources server (which bundles `DirectPythonTool` against a
local sandbox). Wire `ns_tools.verifiers.dsbench_da` and either set
`ns_tools.default_verifier=dsbench_da` or include
`verifier_type=dsbench_da` per row in the JSONL.

## Verification semantics

Reproduces NeMo Skills' DSBench logic:

1. **Extraction.** Apply Skills' `extract_answer(generation, relaxed=True)`
   with `extract_regex=r"(?:The final answer is |\\boxed=)(.+)$"`.
   `relaxed=True` tries the regex first (returning the LAST match),
   falling back to `\boxed{...}`.
2. **`math_equal`.** Skills' MCQ-aware, latex-normalizing,
   `math_verify`-backed comparison of expected and predicted answers.
3. **`relaxed_equal` fallback** — only applied if `math_equal` returns
   False. JSON-parses both sides and recurses on dict / list, then
   case-insensitive single-letter MCQ matching, then `math_equal` of
   stringified args. This is what handles dsbench-da's dict-valued and
   list-valued ground-truth answers.

## Example usage

```bash
# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/ns_tools/configs/ns_tools.yaml,\
resources_servers/dsbench_da/configs/dsbench_da.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=dsbench_da_simple_agent \
    +input_jsonl_fpath=resources_servers/dsbench_da/data/example.jsonl \
    +output_jsonl_fpath=results/dsbench_da_rollouts.jsonl \
    +num_repeats=1
```

## Two agents in `configs/dsbench_da.yaml`

`dsbench_da_simple_agent` (defined alongside the resource server in
this config) is a verifier-only smoke agent: it points at the
`dsbench_da` server directly with no tool execution and consumes
`data/example.jsonl`. Use it for `ng_prepare_data … +mode=example_validation`
and for end-to-end smoke runs against a remote OpenAI-compatible
endpoint without spinning up a python sandbox.

The dsbench_da **benchmark** is wired up in
`benchmarks/dsbench_da/config.yaml` with a separate
`dsbench_da_benchmark_agent` that inherits from `ns_tools_simple_agent`
so it gets the local DirectPythonTool sandbox (which the model needs
to read the per-task Excel files).

## Default verifier override at the CLI

`ns_tools.yaml` ships `default_verifier: math_with_judge`. Chained
configs load AFTER the entry-point file in OmegaConf's merge order,
so a top-level `ns_tools.resources_servers.ns_tools.default_verifier:
dsbench_da` in `benchmarks/dsbench_da/config.yaml` would be silently
overwritten back to `math_with_judge`. The recipe's
`run_dsbench_da_gym.py` therefore passes
`++ns_tools.resources_servers.ns_tools.default_verifier=dsbench_da` at
the CLI. Each prepared row also carries `verifier_type: dsbench_da`,
so per-sample dispatch is unambiguous regardless of the default.

## Reasoning-parser note

When serving a reasoning-style policy (Nemotron-3, DeepSeek-R1, etc.),
start the model server with `--reasoning-parser <name>` so vLLM strips
`<think>...</think>` before the verifier runs. Without it, both the
DSBench extractor regex and `\boxed{...}` extraction will be more likely
to grab in-progress reasoning rather than the final answer.
