# bfcl_v4_multi_turn_agent

Multi-turn function-calling agent for the BFCL v4 multi_turn, memory,
and web_search families. Wraps the upstream `bfcl_eval` package's
per-class Python backends (`GorillaFileSystem`, `MathAPI`, `MessageAPI`,
`TwitterAPI`, `TicketAPI`, `TradingBot`, `TravelAPI`, `VehicleControlAPI`,
`MemoryAPI_*`, `WebSearchAPI`) and runs the per-turn execute-and-feed-back
loop end-to-end on the agent side — same topology as NeMo Skills'
`BFCLGenerationTask::_generate_single_data_point_multi_turn`.

## Vendored modules

`_bfcl_utils.py` — exact copy of Skills'
`nemo_skills/inference/eval/bfcl_utils.py` (the per-turn executor,
`convert_to_function_call`, `execute_multi_turn_func_call`, etc.),
patched only to use this agent's `_bfcl_web_search` for the WebSearchAPI
class mapping. Keeping a vendored copy avoids a runtime nemo_skills
dependency in the agent venv.

`_bfcl_web_search.py` — exact copy of Skills'
`nemo_skills/inference/eval/bfcl_web_search.py` (DuckDuckGo / SerpAPI
backend that BFCL's WebSearchAPI calls into).

`_bfcl_data_utils.py` — copy of Skills'
`nemo_skills/dataset/bfcl_v3/utils.py` for the `multi_turn_miss_func`
holdout path that rebuilds tools mid-conversation.

## Tool-call parsing

Same as `bfcl_v4_ast_agent`: vLLM `/v1/chat/completions` (no
`--tool-call-parser` / no `--enable-auto-tool-choice`) → strip
`<think>…</think>` → run BFCL's per-model FC handler
(`Qwen/Qwen3-8B-FC` by default) on raw content. The structured tool
calls go through both the per-turn executor and the resource server's
verifier.

## Memory family — prereq pre-pass

Memory tasks (`memory_kv`, `memory_vector`, `memory_rec_sum`) require
populating per-conversation state before the scored row runs. The agent
detects `_is_memory(test_category)` and, before executing the user's
turns, calls BFCL's `add_memory_instruction_system_prompt` to inject the
memory-specific system prompt. The prepare.py for `bfcl_v4_memory`
emits prereq + scored rows in the order BFCL expects, and BFCL's
`MemoryAPI._flush_memory_to_local_file()` hook persists state across
prereq → scored row boundary.

## Used by

- `benchmarks/bfcl_v4_multi_turn/`  (4 splits)
- `benchmarks/bfcl_v4_memory/`      (3 splits, with prereq pre-pass)
- `benchmarks/bfcl_v4_web_search/`  (2 splits, with DuckDuckGo backend)
