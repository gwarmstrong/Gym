# bfcl_v4_ast_agent

Single-turn function-calling agent for the BFCL v4 AST family. Mirrors
NeMo Skills' `BFCLGenerationTask` client-parsing path:

1. Apply HF chat template (`apply_chat_template(messages, tools=...)`)
   server-side via vLLM's `/v1/chat/completions` (same chat template +
   tokenizer Skills imports client-side — they match byte-for-byte
   when both ends share the HF tokenizer cache).
2. vLLM is configured WITHOUT `--tool-call-parser` /
   `--enable-auto-tool-choice` so it returns raw text content.
3. Strip `<think>…</think>` from content (analog of Skills'
   `parse_reasoning=True`).
4. Run BFCL's per-model FC handler (e.g. `Qwen/Qwen3-8B-FC`) on the
   raw content via `_parse_query_response_prompting`. This is the
   same parser Skills uses; using vLLM's tool-call parser instead
   would silently diverge from Skills.
5. Forward parsed `predicted_tool_calls` + `predicted_text` to the
   `bfcl_v4_ast` resource server's `verify()`.

Used by `benchmarks/bfcl_v4_ast/` for the 13 single-turn AST splits.
