# AAI Intelligence Index — meta-benchmark

Reproduces the [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking) on Gym. Composes 7 sub-benchmarks (scicode skipped vs Skills' 8) into one rollout-collection job, then computes the composite `overall_score`, `math_score`, and `code_score` post-hoc via `score.py`.

Sub-benchmarks (and per-task rollout counts from AAI):
- `mmlu_pro` (1), `hle` (1), `gpqa` (1)
- `aime25` (10), `livecodebench` v5_2407_2412 (3)
- `ifbench` (5), `aalcr` (4)

This benchmark depends on the agent-specific `num_repeats` primitive from PR [#1356](https://github.com/NVIDIA-NeMo/Gym/pull/1356) so the per-sub rollout counts can be expressed in one `ng_collect_rollouts` call.

# Prepare data
```bash
ng_prepare_benchmark "+config_paths=[benchmarks/aai/config.yaml]"
```

# Collect rollouts
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/aai/config.yaml"

ng_collect_rollouts \
    "+config_paths=[$config_paths]" \
    +output_jsonl_fpath=results/aai_rollouts.jsonl \
    '+num_repeats={mmlu_pro_mcqa_simple_agent: 1, hle_equivalence_llm_judge_simple_agent: 1, gpqa_mcqa_simple_agent: 1, aime25_math_with_judge_simple_agent: 10, livecodebench_v5_2407_2412_code_gen_simple_agent: 3, ifbench_benchmark_simple_agent: 5, aalcr_benchmark_simple_agent: 4}' \
    +num_repeats_add_seed=true
```

# Compute composite score
```bash
python benchmarks/aai/score.py --aggregate-metrics results/aai_rollouts_aggregate_metrics.json
```

# Deviations from Skills' AAI

1. **scicode skipped.** `overall_score` is the mean of 7 sub-scores (not 8); `code_score_livecodebench_only` reports livecodebench alone.
2. **Skills' aime24/aime25 key bug.** `nemo_skills/dataset/aai/aai_score.py:23` reads `metrics["aime24"]` but the suite declares `aime25`. Gym's `score.py` uses `aime25` (matching the actual roll-out).
3. **AAI livecodebench prompt parity.** Skills' `eval/aai/livecodebench.yaml` uses `{question}` (no such field in the prepared data), with a comment claiming starter/format are pre-baked — the actual `prepare.py` does NOT bake them. Gym's AAI prompt at `benchmarks/aai/prompts/livecodebench.yaml` uses the 3 LCB fields (matching the `aa_index.yaml` rendering shape). Diff renderings on the same row in Stage C if parity is in question.
