# SWE-bench Pro

NeMo Gym port of [ScaleAI/SWE-bench_Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro),
the enterprise-grade successor to SWE-bench that ships per-instance docker images
(`jefzda/sweap-images:<tag>`) and a bespoke evaluation harness at
[wasiahmad/SWE-bench_Pro-os](https://github.com/wasiahmad/SWE-bench_Pro-os).

This benchmark reuses the existing `responses_api_agents/swe_agents` wrapper
(OpenHands agent framework) with a new `SweBenchProDatasetProcessor` that
invokes the Pro harness CLI (`--raw_sample_path` / `--patch_path` /
`--scripts_dir`) instead of the stock SWE-bench CLI.

## Example usage

```bash
# Prepare benchmark data (downloads ScaleAI/SWE-bench_Pro from HuggingFace)
ng_prepare_benchmark "+config_paths=[benchmarks/swe_bench_pro/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/swe_bench_pro/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=swe_bench_pro_benchmark_agent \
    +input_jsonl_fpath=benchmarks/swe_bench_pro/data/swe_bench_pro_benchmark.jsonl \
    +output_jsonl_fpath=results/swe_bench_pro_rollouts.jsonl \
    +num_repeats=4
```

## Metrics

Per-row reward is 1.0 iff the patch produced by the agent resolves the upstream
issue (matches Skills' `issues_resolved`). `SWEBenchWrapper.compute_metrics` aggregates
rollouts into:

- `pass@1[avg-of-k]/issues_resolved`
- `pass@k/issues_resolved`
- `majority@k/issues_resolved`
- `pass@k/no_patch`, `pass@k/patch_cant_apply`

These mirror the fields produced by Skills' `SweBenchMetrics` class.
