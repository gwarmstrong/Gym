# dsbench_da

DSBench data-analysis benchmark — agentic Excel-spreadsheet QA. Ported
from NeMo Skills' `nemo_skills/dataset/dsbench_da/`.

The data preparation step downloads `liqiang888/DSBench` from HuggingFace,
unzips per-task directories under `data/extracted/<task_id>/` (each with
an `introduction.txt`, one or more `<question>.txt` question files, and
one or more Excel workbooks), and emits one row per question to
`data/dsbench_da_benchmark.jsonl`. Each row carries Skills' `problem`
text + `excel_paths` (absolute in-container paths the python sandbox
opens at rollout time) plus the per-question `expected_answer`.

Verification routes through the new `dsbench_da` resource server (Skills
parity: `extract_answer(relaxed=True)` with the DSBench regex,
`math_equal` first, then `relaxed_equal` — JSON-aware dict/list
comparison, case-insensitive single-letter MCQ, math fallback). Tool
execution (stateful Python over the Excel files) is delegated to the
existing `ns_tools` resources server; this benchmark's `config.yaml`
overrides `ns_tools.default_verifier=dsbench_da` and adds
`ns_tools.verifiers.dsbench_da`.

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/dsbench_da/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/dsbench_da/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=dsbench_da_simple_agent \
    +input_jsonl_fpath=benchmarks/dsbench_da/data/dsbench_da_benchmark.jsonl \
    +output_jsonl_fpath=results/dsbench_da_rollouts.jsonl \
    +num_repeats=4
```

## Reasoning-parser note

When serving a reasoning-style policy (Nemotron-3, DeepSeek-R1, etc.),
start the model server with `--reasoning-parser <name>` so vLLM strips
`<think>...</think>` before the verifier runs. Without it, both the
DSBench extractor regex and `\boxed{...}` extraction will be more likely
to grab in-progress reasoning rather than the final answer.
