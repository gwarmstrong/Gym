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

## Reasoning-parser note

When serving a reasoning-style policy (Nemotron-3, DeepSeek-R1, etc.),
start the model server with `--reasoning-parser <name>` so vLLM strips
`<think>...</think>` before the verifier runs. Without it, both the
DSBench extractor regex and `\boxed{...}` extraction will be more likely
to grab in-progress reasoning rather than the final answer.
