# sglang_model

Adapter that targets a remote [SGLang](https://github.com/sgl-project/sglang) endpoint
through Gym's Responses API. Reuses `responses_api_models.vllm_model.VLLMModel`'s
Responses↔Chat-Completions converter end-to-end and overrides only the request
shaping needed for SGLang.

## Difference vs. `vllm_model`

`SGLangModel._preprocess_chat_completion_create_params` injects
`tool_choice="auto"` when tools are present and no `tool_choice` was already
set by the caller. Mirrors `nemo_skills/inference/model/sglang.py`, where this
was added because SGLang historically refused tool calls without an explicit
body-level `tool_choice` (vLLM accepts the server flag `--enable-auto-tool-choice`
instead).

Smoke-tested against SGLang 0.5.10.post1 + Qwen2.5-1.5B-Instruct + `--tool-call-parser qwen`,
both with and without the body-level `tool_choice` produced identical clean
tool calls — newer SGLang appears to no longer require it for this path.
The adapter is kept for compatibility with older SGLang versions and other
parser/model combinations the Skills adapter was originally written against.
It is idempotent: it only sets `tool_choice` when the caller has not.

`return_token_id_information` and `is_responses_native` raise `NotImplementedError`
at startup. `return_token_id_information=True` requires per-generation-token
IDs and prompt-token IDs to be surfaced exactly; SGLang 0.5.x's OpenAI-compat
layer drops them in `_process_logprobs_tokens` (only `(token_str, logprob)` is
kept), and `/tokenize` accepts only `{model, prompt}` — chat-style payloads are
rejected. Workarounds based on server `/tokenize` of the assistant content or
client-side vocab lookup were evaluated; the latter fails on multi-byte
UTF-8 byte-fallback tokens (~7% wrong IDs on non-Latin content), the former
is implementable at the cost of two extra round trips per request + EOS-handling
edge cases. The adapter is **eval/inference-only**; use `local_vllm_model` for
RL-training rollouts, which gives exact IDs in-response. Lifting the guard
requires upstream SGLang support (filed as a follow-up).

## Usage

Launch an SGLang server (small-model example):

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --tool-call-parser qwen25 \
    --host 0.0.0.0 --port 30000
```

Set `policy_base_url=http://localhost:30000/v1`, `policy_api_key=dummy`,
`policy_model_name=Qwen/Qwen2.5-1.5B-Instruct` in `env.yaml`, then:

```bash
ng_run "+config_paths=[\
resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/sglang_model/configs/sglang_model.yaml]"
```

Smoke-test against the agent client:

```bash
python responses_api_agents/simple_agent/client.py
```

For a thinking model (e.g. `Qwen/Qwen3-1.7B`), launch SGLang with
`--reasoning-parser qwen3 --tool-call-parser qwen25`, set
`uses_reasoning_parser: true` in the config, and verify `<think>` blocks
round-trip through the converter.
