# BFCL v4 — web_search family

2 web-search-augmented function-calling splits from BFCL v4:

- `web_search_base` — multi-turn dialogue with the WebSearchAPI tool
  exposed; the model issues queries and consumes results.
- `web_search_no_snippet` — same but without snippet text in results
  (model has to fetch URLs itself).

This benchmark reuses `bfcl_v4_multi_turn_agent` and
`bfcl_v4_multi_turn` resource server. The agent's vendored
`_bfcl_web_search.py` (copy of Skills'
`nemo_skills/inference/eval/bfcl_web_search.py`) provides the
DuckDuckGo + optional SerpAPI backend that BFCL's WebSearchAPI calls.

To run web_search rollouts the agent venv must have either the `ddgs`
package installed (default DuckDuckGo path) or `SERPAPI_API_KEY` set
in the environment. The vendored backend fails fast at scenario load
time if neither is available.

## Example usage

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/bfcl_v4_web_search/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/bfcl_v4_web_search/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=bfcl_v4_web_search_agent \
    +input_jsonl_fpath=benchmarks/bfcl_v4_web_search/data/bfcl_v4_web_search_benchmark.jsonl \
    +output_jsonl_fpath=results/bfcl_v4_web_search_rollouts.jsonl \
    +num_repeats=4
```
