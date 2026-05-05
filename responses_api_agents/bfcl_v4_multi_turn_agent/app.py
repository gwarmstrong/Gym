# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BFCL v4 multi-turn agent.

Handles all three multi-turn families (multi_turn, memory, web_search)
with per-test-category branching internal to `run()`. Mirrors NeMo
Skills' `BFCLGenerationTask::_generate_single_data_point_multi_turn`:

  1. Per-turn: call vLLM /v1/chat/completions with messages + tools.
  2. Strip <think>...</think>, run BFCL FC handler to extract structured calls.
  3. Convert structured calls -> Python eval strings via convert_to_function_call.
  4. Execute against Gorilla's stateful Python class instances
     (GorillaFileSystem, MathAPI, MessageAPI, TwitterAPI, TicketAPI,
     TradingBot, TravelAPI, VehicleControlAPI, WebSearchAPI, MemoryAPI_*).
  5. Append `tool` messages with execution results.
  6. Loop until: no more tool calls, MAXIMUM_STEP_LIMIT (20), or context exhausted.
  7. After all turns done, ship the full nested response list to verify().

Memory test cases require a *prereq pre-pass*: the agent first walks
all "<id>_prereq_<n>" rows (in order) to populate memory state, then
runs the scored row against that populated state. Implemented at the
batch-input level: prepare.py emits prereq + scored rows in interleaved
order, and the agent's run() does prereq accounting via the
`memory_instance._flush_memory_to_local_file()` hook from BFCL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, ClassVar, Dict, List, Optional

from fastapi import Body, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
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
from responses_api_agents.bfcl_v4_multi_turn_agent._bfcl_utils import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    MAXIMUM_STEP_LIMIT,
    convert_to_function_call,
    execute_multi_turn_func_call,
    is_empty_execute_response,
)


LOG = logging.getLogger(__name__)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class BfclV4MultiTurnAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # FC handler key in bfcl_eval.constants.model_config.local_inference_model_map.
    model_handler: str = "Qwen/Qwen3-8B-FC"
    # Per-turn step cap inside one user-turn. Mirrors Skills' MAXIMUM_STEP_LIMIT=20.
    max_steps_per_turn: int = MAXIMUM_STEP_LIMIT


class BfclV4MultiTurnAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class BfclV4MultiTurnAgentVerifyRequest(BaseVerifyRequest):
    # Override BaseVerifyRequest.response (NeMoGymResponse) — see
    # bfcl_v4_ast_agent for rationale.
    response: Dict[str, Any] = Field(default_factory=dict)
    id: str
    test_category: str
    # Per-turn list of per-step model responses (Skills' all_model_response shape).
    generation: List[List[Any]] = Field(default_factory=list)
    error: Optional[str] = None


class BfclV4MultiTurnAgentVerifyResponse(BaseVerifyResponse):
    # Same response-field relaxation as the request side.
    response: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


def _strip_reasoning(text: Optional[str]) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text).lstrip("\n")


def _is_memory(category: str) -> bool:
    return category.startswith("memory_")


def _is_long_context(category: str) -> bool:
    return "long_context" in category or "composite" in category


_DIRECT_HANDLER_MODULES = {
    "Qwen/Qwen3-8B-FC": ("bfcl_eval.model_handler.local_inference.qwen_fc", "QwenFCHandler"),
    "Qwen/Qwen3-4B-FC": ("bfcl_eval.model_handler.local_inference.qwen_fc", "QwenFCHandler"),
}


class _SyntheticChoice:
    def __init__(self, text: str) -> None:
        self.text = text


class _SyntheticUsage:
    prompt_tokens = 0
    completion_tokens = 0


class _SyntheticResponse:
    def __init__(self, text: str) -> None:
        self.choices = [_SyntheticChoice(text)]
        self.usage = _SyntheticUsage()


def _build_response_parser(model_handler_key: str):
    """See bfcl_v4_ast_agent._build_response_parser for rationale.

    Direct handler-module import avoids bfcl_eval.constants.model_config's
    eager registry import (which pulls Gemini/Anthropic/Cohere/Qwen API
    SDK chains the Gym container lacks). chat_completions response is
    adapted to text-completions shape via _SyntheticResponse.
    """
    if model_handler_key not in _DIRECT_HANDLER_MODULES:
        raise ValueError(f"BFCL handler {model_handler_key!r} not yet wired in _DIRECT_HANDLER_MODULES.")
    module_path, class_name = _DIRECT_HANDLER_MODULES[model_handler_key]
    import importlib

    module = importlib.import_module(module_path)
    handler_cls = getattr(module, class_name)
    handler = handler_cls(
        model_name=model_handler_key.replace("-FC", ""),
        temperature=0.0,
        registry_name=model_handler_key.replace("-FC", ""),
        is_fc_model=True,
    )

    def parse(raw_chat_response: Dict[str, Any]) -> Dict[str, Any]:
        content = raw_chat_response["choices"][0]["message"].get("content", "") or ""
        synthetic = _SyntheticResponse(content)
        parsed = handler._parse_query_response_prompting(synthetic)
        msg = parsed.get("model_responses_message_for_chat_history") or {}
        tool_calls = msg.get("tool_calls") or []
        tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
        return {"content": msg.get("content", "") or "", "tool_calls": tool_calls}

    return parse


class BfclV4MultiTurnAgent(SimpleResponsesAPIAgent):
    config: BfclV4MultiTurnAgentConfig
    _response_parser = None
    # Per-(seed, scenario) async lock for memory rollouts. BFCL's MemoryAPI
    # flushes whole-state snapshots to a single
    # <model_result_dir>/agentic/memory/<backend>/memory_snapshot/<scenario>_final.json
    # file. When multiple rollouts of the same scenario run concurrently
    # they overwrite each other's flush and prereq state is lost. Skills
    # avoids this by running prereqs sequentially in load_data; we don't
    # have a load_data hook through the Gym rollout client, so serialize
    # per-(seed, scenario) here. Across (seed, scenario) tuples, the
    # filesystem paths are independent so they can run in parallel.
    # ClassVar so Pydantic v2 doesn't treat these as model fields (which
    # would make the bare class-attribute access return a
    # ModelPrivateAttr descriptor instead of the dict).
    _memory_locks: ClassVar[Dict[tuple, asyncio.Lock]] = {}
    # Per-(seed, prereq_id) async event signalling that the prereq has
    # finished flushing its MemoryAPI state to disk. Scored memory
    # rollouts wait on the events for all ids listed in their
    # `depends_on` field before starting, so they don't read an empty
    # snapshot file. Skills enforces this serialization in load_data
    # (sync for-loop over prereqs before returning non_prereqs); we do
    # it via async events because our rollout client doesn't have a
    # load_data hook.
    _prereq_done_events: ClassVar[Dict[tuple, asyncio.Event]] = {}

    def model_post_init(self, __context) -> None:
        # Re-install bfcl_eval after the rollout client's `uv sync`
        # stripped it on startup. See bfcl_v4_ast_agent for details.
        from responses_api_agents.bfcl_v4_multi_turn_agent._ensure_bfcl_eval import (
            ensure_bfcl_eval_installed,
        )

        ensure_bfcl_eval_installed()
        super().model_post_init(__context)

    @classmethod
    def _get_memory_lock(cls, key: tuple) -> asyncio.Lock:
        if key not in cls._memory_locks:
            cls._memory_locks[key] = asyncio.Lock()
        return cls._memory_locks[key]

    @classmethod
    def _get_prereq_event(cls, seed: int, prereq_id: str) -> asyncio.Event:
        key = (seed, prereq_id)
        if key not in cls._prereq_done_events:
            cls._prereq_done_events[key] = asyncio.Event()
        return cls._prereq_done_events[key]

    def _get_parser(self):
        if self._response_parser is None:
            self._response_parser = _build_response_parser(self.config.model_handler)
        return self._response_parser

    async def _call_model(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        responses_create_params: Any,
        cookies,
    ) -> Optional[Dict[str, Any]]:
        # responses_create_params is a Pydantic v2 model
        # (NeMoGymResponseCreateParamsNonStreaming); v2 BaseModel does NOT
        # support `in`/`[]` lookups, so the prior dict-style code silently
        # fell through to vLLM defaults for every override. Use attribute
        # access with a dict fallback (callers that pass `or {}` for a
        # missing field still work).
        def _get(field: str):
            if isinstance(responses_create_params, dict):
                return responses_create_params.get(field)
            return getattr(responses_create_params, field, None)

        # See bfcl_v4_ast_agent for why we use tool_choice="none".
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
            val = _get(src)
            if val is not None:
                chat_body[dst] = val
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
        try:
            await raise_for_status(response)
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            if "context length" in err or "max_tokens" in err:
                LOG.warning("BFCL multi-turn ran out of context: %s", err)
                return None
            raise
        return await get_response_json(response)

    @staticmethod
    def _format_function_call_dict_list(
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Convert structured tool-call dicts to Skills' BFCL `generation` shape:
        a list of single-key {"<name>": "<args_json>"} dicts."""
        out: List[Dict[str, str]] = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            args_str = args if isinstance(args, str) else json.dumps(args)
            out.append({name: args_str})
        return out

    async def responses(self, body=None):
        # BFCL flow drives vLLM /v1/chat/completions directly from run();
        # the agent's own /v1/responses is unused. Stub satisfies the
        # SimpleResponsesAPIAgent abstract interface.
        raise NotImplementedError("bfcl_v4_multi_turn_agent does not expose /v1/responses; use /run.")

    async def run(
        self,
        request: Request,
        body: BfclV4MultiTurnAgentRunRequest,
    ) -> BfclV4MultiTurnAgentVerifyResponse:
        # For memory rollouts, acquire a per-(seed, scenario) lock so the
        # whole-state snapshot in customer_final.json (or finance_final,
        # etc.) doesn't get clobbered by concurrent rollouts of the same
        # scenario within the same seed. See _memory_locks docstring.
        meta = body.verifier_metadata or {}
        category = meta.get("test_category", "")
        if _is_memory(category):
            scenario = meta.get("scenario", "")
            rollout_index = self._extract_rollout_index(body)
            row_id = meta.get("id", "")
            # Memory rollouts (both prereq AND scored) wait on
            # depends_on prereq events before acquiring the per-(seed,
            # scenario) lock — prereq-chain dependencies (prereq_N waits
            # for prereq_N-1) AND scored dependencies (scored waits for
            # all prereqs in scenario). Wait *outside* the lock — the
            # depended-on prereqs need the same lock to flush, so
            # waiting inside would deadlock. The per-(seed, scenario)
            # lock alone provides mutual exclusion but no FIFO ordering
            # of unrelated waiters, so we'd see e.g. prereq_23 hold the
            # lock before prereq_22 ran.
            depends_on = meta.get("depends_on") or []
            for prereq_id in depends_on:
                await self._get_prereq_event(rollout_index, prereq_id).wait()
            lock = self._get_memory_lock((rollout_index, scenario))
            async with lock:
                try:
                    return await self._run_inner(request, body)
                except Exception:
                    LOG.exception(
                        "bfcl_v4_multi_turn_agent.run() failed for id=%s test_category=%s",
                        meta.get("id", ""),
                        category,
                    )
                    raise
        try:
            return await self._run_inner(request, body)
        except Exception:
            LOG.exception(
                "bfcl_v4_multi_turn_agent.run() failed for id=%s test_category=%s",
                meta.get("id", ""),
                category,
            )
            raise

    @staticmethod
    def _extract_rollout_index(body) -> int:
        try:
            rcp = body.responses_create_params
            md = getattr(rcp, "metadata", None) if rcp is not None else None
            if md is None:
                return 0
            eb_raw = md.get("extra_body") if isinstance(md, dict) else getattr(md, "extra_body", None)
            if isinstance(eb_raw, str):
                eb = json.loads(eb_raw)
            elif isinstance(eb_raw, dict):
                eb = eb_raw
            else:
                eb = {}
            if isinstance(eb.get("seed"), int):
                return eb["seed"]
        except Exception:  # noqa: BLE001
            pass
        return 0

    async def _run_inner(
        self,
        request: Request,
        body: BfclV4MultiTurnAgentRunRequest,
    ) -> BfclV4MultiTurnAgentVerifyResponse:
        cookies = request.cookies
        meta = body.verifier_metadata or {}
        row_id = meta.get("id", "")
        test_category = meta.get("test_category", "")
        # `question` is the BFCL multi-turn shape: list of message-list turns.
        all_multi_turn_messages: List[List[Dict[str, Any]]] = meta.get("question", [])
        tools = list(meta.get("tools", []))
        initial_config = meta.get("initial_config", {})
        involved_classes = list(meta.get("involved_classes", []))
        # Multi_turn_miss_func splits introduce extra functions partway through.
        holdout_function: Dict[str, list] = meta.get("missed_function", {})
        # +num_repeats=N +num_repeats_add_seed=true makes the rollout client
        # set responses_create_params.metadata.extra_body = {"seed": k} for
        # k in 0..N-1. We pass `k` to execute_multi_turn_func_call as
        # rollout_index so each seed gets an isolated GorillaFileSystem /
        # MathAPI / etc instance — without this, all N seeds of the same
        # task share one stateful instance via globals(), and seed-1
        # inherits seed-0's leftover filesystem state. Skills doesn't hit
        # this because it runs each seed in a separate process.
        rollout_index = self._extract_rollout_index(body)

        # memory_vector imports SentenceTransformer("all-MiniLM-L6-v2") at
        # module load time, which hits HF Hub. With HF_HUB_OFFLINE=1 (set
        # for the agent venv to keep cluster runs hermetic) this fails
        # before the rollout can begin. Skills' baseline shows the same
        # gap — bfcl_v4.memory_vector/ has no output-rs*.jsonl and only
        # an empty summarized-results dir. Cleanly skip the category so
        # both pipelines align on the same coverage gap. Pre-caching
        # all-MiniLM-L6-v2 in $HF_HOME would lift this restriction on
        # both sides; tracked as follow-up work.
        if test_category == "memory_vector":
            skip_request = BfclV4MultiTurnAgentVerifyRequest(
                responses_create_params=body.responses_create_params,
                response={"output_text": ""},
                id=row_id,
                test_category=test_category,
                generation=[],
                error="memory_vector skipped (all-MiniLM-L6-v2 not cached; Skills baseline also missed this category)",
            )
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=skip_request.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            return BfclV4MultiTurnAgentVerifyResponse.model_validate(await get_response_json(verify_response))

        # Memory category: rewrite `initial_config[MemoryAPI_*]["model_result_dir"]`
        # before materializing the instance. The path baked in during
        # prepare.py points at the temp dir bfcl_eval was cloned into
        # (e.g. /tmp/tmp4cygoma_/.../bfcl_eval/data) which doesn't exist
        # on the cluster — MemoryAPI._flush_memory_to_local_file then
        # silently fails to persist prereq state, and every scored
        # rollout starts with an empty memory. Skills' BFCLGenerationTask.load_data
        # does the equivalent rewrite (with our Path.parent fix). Use a
        # stable per-seed dir so prereq rollouts and the scored rollout
        # share state, but seeds don't race each other.
        if _is_memory(test_category):
            memory_state_dir = f"/tmp/bfcl_v4_memory_state_seed{rollout_index}"
            initial_config = {
                k: ({**v, "model_result_dir": memory_state_dir} if k.startswith("MemoryAPI") else v)
                for k, v in initial_config.items()
            }

        # Memory category: inject memory instruction system prompt now that
        # the memory instance state is known. Skills runs an empty
        # execute_multi_turn_func_call to materialize the instance, then
        # calls bfcl_eval.model_handler.utils.add_memory_instruction_system_prompt.
        if _is_memory(test_category):
            from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_api_metaclass import (  # noqa
                MemoryAPI,
            )
            from bfcl_eval.model_handler.utils import add_memory_instruction_system_prompt

            _, involved_instances = execute_multi_turn_func_call(
                [],
                initial_config,
                involved_classes,
                test_entry_id=row_id,
                long_context=_is_long_context(test_category),
                rollout_index=rollout_index,
            )
            assert len(involved_instances) == 1
            memory_instance: MemoryAPI = list(involved_instances.values())[0]
            all_multi_turn_messages = add_memory_instruction_system_prompt(
                all_multi_turn_messages,
                test_category,
                meta.get("scenario", ""),
                memory_instance,
            )

        chat_messages: List[Dict[str, Any]] = []
        all_model_response: List[List[Any]] = []
        force_quit = False
        out_of_context = False
        parser = self._get_parser()

        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_response: List[Any] = []
            count = 0

            # multi_turn_miss_func: holdout function appears at this turn.
            if str(turn_idx) in holdout_function:
                # Append new functions and rebuild tools.
                func_list = list(meta.get("function", []))
                func_list.extend(holdout_function[str(turn_idx)])
                # Rebuild tools using BFCL's preprocessing.
                from responses_api_agents.bfcl_v4_multi_turn_agent._bfcl_data_utils import (
                    convert_to_tool,
                    func_doc_language_specific_pre_processing,
                )

                func_list = func_doc_language_specific_pre_processing(func_list, test_category)
                tools = convert_to_tool(func_list)
                meta["function"] = func_list
                assert len(current_turn_message) == 0, "Holdout turn should have no user message"
                current_turn_message = [{"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]

            chat_messages.extend(current_turn_message)

            while True:
                model_response = await self._call_model(
                    messages=chat_messages,
                    tools=tools,
                    responses_create_params=body.responses_create_params or {},
                    cookies=cookies,
                )
                if model_response is None:
                    out_of_context = True
                    break

                choice = model_response.get("choices", [{}])[0]
                message = choice.get("message", {}) or {}
                raw_content = _strip_reasoning(message.get("content", ""))
                synthetic = {
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
                parsed = parser(synthetic)

                # Build the assistant message for chat history. vllm_model's
                # pydantic schema (NeMoGymChatCompletionAssistantMessageParam)
                # requires each tool_call to be the OpenAI-shape
                #   {id, type: "function", function: {name, arguments(str)}}
                # — BFCL's FC handler emits the flat {name, arguments(dict)}
                # form, so we wrap here. Generate per-call IDs and reuse
                # them on the matching tool messages so vllm_model's schema
                # validation passes (tool_call_id is also required).
                import uuid

                openai_tool_calls = []
                tool_call_ids = []
                for tc in parsed["tool_calls"]:
                    call_id = tc.get("id") or tc.get("call_id") or f"call_{uuid.uuid4().hex[:8]}"
                    args_val = tc.get("arguments", {})
                    args_str = args_val if isinstance(args_val, str) else json.dumps(args_val)
                    openai_tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": args_str,
                            },
                        }
                    )
                    tool_call_ids.append(call_id)

                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": parsed["content"] or "",
                }
                if openai_tool_calls:
                    assistant_msg["tool_calls"] = openai_tool_calls
                chat_messages.append(assistant_msg)

                # Skills' `generation` shape: list of {name: args_json} dicts
                # if there were calls, else the raw string content.
                formatted_calls: List[Dict[str, str]] = []
                if parsed["tool_calls"]:
                    formatted_calls = self._format_function_call_dict_list(parsed["tool_calls"])
                    current_turn_response.append(formatted_calls)
                else:
                    current_turn_response.append(parsed["content"])

                # Decode → Python eval strings.
                try:
                    decoded_calls = convert_to_function_call(formatted_calls) if parsed["tool_calls"] else []
                    if not decoded_calls or is_empty_execute_response(decoded_calls):
                        break
                except Exception:  # noqa: BLE001
                    break

                # Execute against Gorilla's class instances.
                exec_results, _ = execute_multi_turn_func_call(
                    decoded_calls,
                    initial_config,
                    involved_classes,
                    test_entry_id=row_id,
                    long_context=_is_long_context(test_category),
                    rollout_index=rollout_index,
                )
                # Feed results back as `tool` messages with matched IDs.
                for exec_result, call_id in zip(exec_results, tool_call_ids):
                    chat_messages.append(
                        {
                            "role": "tool",
                            "content": exec_result,
                            "tool_call_id": call_id,
                        }
                    )

                count += 1
                if count > self.config.max_steps_per_turn:
                    force_quit = True
                    break

            all_model_response.append(current_turn_response)
            if force_quit or out_of_context:
                break

        # Memory prereq rollouts must flush their MemoryAPI instance state
        # to <model_result_dir>/agentic/memory/<backend>/memory_snapshot/<scenario>_final.json
        # so that subsequent scored rollouts can read the populated state.
        # BFCL's own base_handler.py does this in two places (lines 376
        # and 667 of bfcl_eval/model_handler/base_handler.py); since our
        # agent doesn't use base_handler, we have to call it explicitly.
        # Without this, scored memory rollouts always start with empty
        # memory and the model can't answer questions about prior turns.
        if _is_memory(test_category) and "_prereq_" in row_id:
            try:
                memory_instance._flush_memory_to_local_file()
            except Exception:  # noqa: BLE001
                LOG.exception("memory flush failed for prereq id=%s", row_id)
            # Signal scored rollouts that depend on this prereq.
            self._get_prereq_event(rollout_index, row_id).set()

        verify_request = BfclV4MultiTurnAgentVerifyRequest(
            responses_create_params=body.responses_create_params,
            response={"output_text": json.dumps(all_model_response)},
            id=row_id,
            test_category=test_category,
            generation=all_model_response,
            error="_ran_out_of_context_" if out_of_context else None,
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return BfclV4MultiTurnAgentVerifyResponse.model_validate(await get_response_json(verify_response))

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    BfclV4MultiTurnAgent.run_webserver()
