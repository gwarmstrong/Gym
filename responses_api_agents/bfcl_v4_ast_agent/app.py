# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BFCL v4 AST single-turn agent.

Mirrors `nemo_skills/inference/eval/bfcl.py::ClientMessageParser` exactly:

  1. Pick a per-model FC handler from `bfcl_eval.local_inference_model_map`
     (e.g. `Qwen/Qwen3-8B-FC`). The handler knows that model's exact
     tool-call emission format.
  2. Send messages + tools to vLLM via /v1/chat/completions. Do NOT use
     vLLM's `--tool-call-parser` / `--enable-auto-tool-choice` — vLLM
     emits raw text content; the FC handler does the extraction.
  3. Strip `<think>…</think>` (Skills' `parse_reasoning=True` analog).
  4. Run the FC handler's `_parse_query_response_prompting` on the raw
     content to produce structured tool calls.
  5. Hand off to the resource server's verify() with the parsed calls.

This is the AST single-turn flavor — runs the loop exactly once.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional


LOG = logging.getLogger(__name__)

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import get_response_json, raise_for_status


# Strip `<think>...</think>` blocks (vllm_model wraps reasoning_content
# back into the content field as a `<think>` block when uses_reasoning_parser=True).
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class BfclV4AstAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # FC handler key in bfcl_eval.constants.model_config.local_inference_model_map.
    # Must match the resource server's model_handler.
    model_handler: str = "Qwen/Qwen3-8B-FC"


class BfclV4AstAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class BfclV4AstAgentVerifyRequest(BaseVerifyRequest):
    # Override BaseVerifyRequest.response (NeMoGymResponse) — BFCL flow
    # never uses the model's full Responses-API shape; we only carry the
    # extracted text/tool_calls forward to the resource server.
    response: Dict[str, Any] = Field(default_factory=dict)
    id: str
    test_category: str
    predicted_tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    predicted_text: str = ""


class BfclV4AstAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _strip_reasoning(text: Optional[str]) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text).lstrip("\n")


_DIRECT_HANDLER_MODULES = {
    "Qwen/Qwen3-8B-FC": ("bfcl_eval.model_handler.local_inference.qwen_fc", "QwenFCHandler"),
    "Qwen/Qwen3-4B-FC": ("bfcl_eval.model_handler.local_inference.qwen_fc", "QwenFCHandler"),
}


class _SyntheticChoice:
    """Stand-in for OpenAI text-completion Choice (has `.text`)."""

    def __init__(self, text: str) -> None:
        self.text = text


class _SyntheticUsage:
    """Stand-in for OpenAI usage block. QwenFC handler reads
    `prompt_tokens` and `completion_tokens` for return-dict bookkeeping
    only — token counts aren't used downstream by Skills' parser."""

    prompt_tokens = 0
    completion_tokens = 0


class _SyntheticResponse:
    """Stand-in for an OpenAI text-completion Response object."""

    def __init__(self, text: str) -> None:
        self.choices = [_SyntheticChoice(text)]
        self.usage = _SyntheticUsage()


def _build_response_parser(model_handler_key: str):
    """Instantiate BFCL's per-model FC handler and return its parse fn.

    Mirrors ClientMessageParser._validate_and_setup_client_parsing.

    We import the handler module DIRECTLY (not through
    bfcl_eval.constants.model_config) because the registry module eagerly
    imports every backend (Gemini, Anthropic, Cohere, Qwen API, Mistral,
    Writer, Bedrock, ...). Each backend pulls in SDKs the lean Gym
    container lacks (cryptography, soundfile, PIL, ...). Importing the
    one local-inference handler we need avoids that whole tree.

    Skills' _parse_query_response_prompting expects an
    OpenAI-text-completions response (`api_response.choices[0].text`).
    We're calling the chat-completions endpoint, so we adapt content to
    that shape with _SyntheticResponse.
    """
    if model_handler_key not in _DIRECT_HANDLER_MODULES:
        raise ValueError(
            f"BFCL handler {model_handler_key!r} not yet wired in _DIRECT_HANDLER_MODULES. "
            f"Add a (module, class) entry to bfcl_v4_ast_agent/app.py."
        )
    module_path, class_name = _DIRECT_HANDLER_MODULES[model_handler_key]
    import importlib

    module = importlib.import_module(module_path)
    handler_cls = getattr(module, class_name)
    handler = handler_cls(
        model_name=model_handler_key.replace("-FC", ""),
        temperature=0.0,  # not used during parsing
        registry_name=model_handler_key.replace("-FC", ""),
        is_fc_model=True,
    )

    def parse(raw_chat_response: Dict[str, Any]) -> Dict[str, Any]:
        # Adapt chat-completions message.content to text-completions .text.
        content = raw_chat_response["choices"][0]["message"].get("content", "") or ""
        synthetic = _SyntheticResponse(content)
        parsed = handler._parse_query_response_prompting(synthetic)
        msg = parsed.get("model_responses_message_for_chat_history") or {}
        tool_calls = msg.get("tool_calls") or []
        tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
        return {
            "content": msg.get("content", "") or "",
            "tool_calls": tool_calls,
        }

    return parse


class BfclV4AstAgent(SimpleResponsesAPIAgent):
    config: BfclV4AstAgentConfig
    _response_parser = None  # set on first request

    def model_post_init(self, __context) -> None:
        # The rollout client runs `uv sync` on startup, which strips
        # bfcl_eval (intentionally not pinned in pyproject.toml). Restore
        # it now, after the sync, before any rollout request lands.
        from responses_api_agents.bfcl_v4_ast_agent._ensure_bfcl_eval import (
            ensure_bfcl_eval_installed,
        )

        ensure_bfcl_eval_installed()
        super().model_post_init(__context)

    def _get_parser(self):
        if self._response_parser is None:
            self._response_parser = _build_response_parser(self.config.model_handler)
        return self._response_parser

    async def _call_model(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        responses_create_params: Dict[str, Any],
        cookies,
    ) -> Dict[str, Any]:
        """Invoke vllm_model.chat_completions and return the raw dict."""
        # Pass through sampling params from responses_create_params; map
        # max_output_tokens -> max_completion_tokens (chat-completions).
        # tool_choice="none" tells vLLM to apply the chat template with
        # tool definitions injected but NOT add any tool-choice constraint
        # and NOT try to parse tool_calls server-side. Output is raw text,
        # which BFCL's per-model FC handler then parses.
        # Sending tools without tool_choice (or with tool_choice="auto")
        # is rejected by vLLM unless --enable-auto-tool-choice +
        # --tool-call-parser are set — but BFCL parity requires those
        # flags off (vLLM's parser would silently disagree with BFCL's
        # per-model FC handler).
        chat_body: Dict[str, Any] = {"messages": messages}
        if tools:
            chat_body["tools"] = tools
            chat_body["tool_choice"] = "none"
        for src, dst in [
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_output_tokens", "max_completion_tokens"),
            ("stop", "stop"),
            ("seed", "seed"),
            ("metadata", "metadata"),
        ]:
            if src in responses_create_params and responses_create_params[src] is not None:
                chat_body[dst] = responses_create_params[src]

        response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/chat/completions",
            json=chat_body,
            cookies=cookies,
        )
        if response.status >= 400:
            err_body = (await response.content.read()).decode("utf-8", "replace")
            LOG.error(
                "vllm_model /v1/chat/completions returned %d. body sent (truncated 1KB): %s. response: %s",
                response.status,
                json.dumps(chat_body, default=str)[:1024],
                err_body[:2048],
            )
        await raise_for_status(response)
        return await get_response_json(response)

    async def responses(self, body=None):
        # BFCL flow goes straight from run() to vLLM /v1/chat/completions
        # — the agent's own /v1/responses endpoint is unused. Stub kept
        # only to satisfy SimpleResponsesAPIAgent's abstract interface.
        raise NotImplementedError("bfcl_v4_ast_agent does not expose /v1/responses; use /run instead.")

    async def run(
        self,
        request: Request,
        body: BfclV4AstAgentRunRequest,
    ) -> BfclV4AstAgentVerifyResponse:
        try:
            return await self._run_inner(request, body)
        except Exception:
            LOG.exception("bfcl_v4_ast_agent.run() failed")
            raise

    async def _run_inner(
        self,
        request: Request,
        body: BfclV4AstAgentRunRequest,
    ) -> BfclV4AstAgentVerifyResponse:
        cookies = request.cookies
        meta = body.verifier_metadata or {}
        test_category = meta.get("test_category", "")
        row_id = meta.get("id", "")
        # `question` is BFCL's nested message format: list of message-list
        # turns. AST is single-turn so we always have exactly one turn.
        question_turns: List[List[Dict[str, Any]]] = meta.get("question", [])
        messages = list(question_turns[0]) if question_turns else []
        tools = meta.get("tools", [])

        # Single forward pass; no execute-and-feed-back loop for AST.
        chat_completion = await self._call_model(
            messages=messages,
            tools=tools,
            responses_create_params=body.responses_create_params or {},
            cookies=cookies,
        )

        choice = chat_completion.get("choices", [{}])[0]
        message = choice.get("message", {}) or {}
        raw_content = _strip_reasoning(message.get("content", ""))
        # vllm_model wraps reasoning into <think>...</think> in content;
        # our strip leaves the post-reasoning answer. Re-attach to a
        # synthetic response shape the FC handler can read.
        synthetic_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": raw_content,
                        "tool_calls": message.get("tool_calls"),
                    }
                }
            ]
        }

        parser = self._get_parser()
        parsed = parser(synthetic_response)

        verify_request = BfclV4AstAgentVerifyRequest(
            responses_create_params=body.responses_create_params,
            response={"output_text": raw_content},
            id=row_id,
            test_category=test_category,
            predicted_tool_calls=parsed["tool_calls"],
            predicted_text=parsed["content"],
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return BfclV4AstAgentVerifyResponse.model_validate(await get_response_json(verify_response))

    async def aggregate_metrics(self, body):
        from nemo_gym.base_resources_server import AggregateMetrics

        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    BfclV4AstAgent.run_webserver()
