# BFCL v4 — multi-turn family

4 multi-turn tool-execution splits from BFCL v4:

- `multi_turn_base` — base multi-turn dialogues against
  Gorilla's Python class backends.
- `multi_turn_miss_func` — extra functions appear partway through
  the dialogue and the model must adapt.
- `multi_turn_miss_param` — required parameters are deliberately
  ambiguous; the model must ask for them.
- `multi_turn_long_context` — long-context multi-turn variant.

Rollouts go through `bfcl_v4_multi_turn_agent`, which runs the
per-turn execute-and-feed-back loop against `bfcl_eval`'s in-process
class instances (`GorillaFileSystem`, `MathAPI`, `MessageAPI`,
`TwitterAPI`, `TicketAPI`, `TradingBot`, `TravelAPI`,
`VehicleControlAPI`). Grading is batched in
`bfcl_v4_multi_turn`'s `compute_metrics()` via
`python -m bfcl_eval evaluate`.

## Example usage

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/bfcl_v4_multi_turn/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/bfcl_v4_multi_turn/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=bfcl_v4_multi_turn_benchmark_agent \
    +input_jsonl_fpath=benchmarks/bfcl_v4_multi_turn/data/bfcl_v4_multi_turn_benchmark.jsonl \
    +output_jsonl_fpath=results/bfcl_v4_multi_turn_rollouts.jsonl \
    +num_repeats=4
```
