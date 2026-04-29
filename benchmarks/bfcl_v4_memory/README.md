# BFCL v4 — memory family

3 memory-augmented function-calling splits from BFCL v4:

- `memory_kv` — key/value memory backend.
- `memory_vector` — vector memory backend.
- `memory_rec_sum` — recursive-summarization memory backend.

Memory tasks need a *prereq pre-pass* — each memory test row has zero
or more `_prereq_<n>` siblings that must be executed (in order) before
the scored row to populate the persistent state. `prepare.py` emits
prereqs first (sorted by their numeric suffix), then scored rows;
`bfcl_v4_multi_turn_agent` runs the per-turn loop for each in order
and lets BFCL's `MemoryAPI._flush_memory_to_local_file()` hook persist
state across rows. Same flow as NeMo Skills' `BFCLGenerationTask::load_data`.

This benchmark reuses `bfcl_v4_multi_turn_agent` (same per-turn loop;
agent switches internally on `test_category` to inject the memory
instruction system prompt) and `bfcl_v4_multi_turn` resource server
(BFCL grader handles memory state checks).

## Example usage

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/bfcl_v4_memory/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/bfcl_v4_memory/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=bfcl_v4_memory_agent \
    +input_jsonl_fpath=benchmarks/bfcl_v4_memory/data/bfcl_v4_memory_benchmark.jsonl \
    +output_jsonl_fpath=results/bfcl_v4_memory_rollouts.jsonl \
    +num_repeats=4
```
