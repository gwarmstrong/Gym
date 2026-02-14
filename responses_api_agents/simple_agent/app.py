# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import time
from collections import defaultdict
from typing import List

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status

# ========== DIAGNOSTIC: Request tracking ==========
_ACTIVE_REQUESTS = {}  # rid -> {start_time, phase, step, detail}
_REQUEST_COUNTER = 0
_DIAG_INTERVAL = 30  # seconds
_TOOL_CALL_TIMEOUTS = 0  # count of timeouts hit


async def _diag_async_monitor():
    """Periodic background task that logs active request state and asyncio health."""
    print(f"[DIAG-AGENT] Monitor started (interval={_DIAG_INTERVAL}s)", flush=True)
    while True:
        await asyncio.sleep(_DIAG_INTERVAL)
        try:
            now = time.time()
            active = len(_ACTIVE_REQUESTS)

            if active == 0:
                all_tasks = asyncio.all_tasks()
                print(f"[DIAG-AGENT] t={now:.0f} | No active requests | {len(all_tasks)} asyncio tasks | timeouts_hit={_TOOL_CALL_TIMEOUTS}", flush=True)
                continue

            # Group by phase
            by_phase = defaultdict(list)
            longest_elapsed = 0
            for rid, info in _ACTIVE_REQUESTS.items():
                elapsed = now - info['start_time']
                longest_elapsed = max(longest_elapsed, elapsed)
                phase = info.get('phase', 'unknown')
                detail = info.get('detail', '')
                step = info.get('step', 0)
                label = f"r{rid}(s{step},{elapsed:.0f}s)"
                if detail:
                    label = f"r{rid}(s{step},{elapsed:.0f}s,{detail})"
                by_phase[phase].append(label)

            all_tasks = asyncio.all_tasks()
            print(f"[DIAG-AGENT] t={now:.0f} | {active} active reqs | longest={longest_elapsed:.0f}s | {len(all_tasks)} asyncio tasks | timeouts_hit={_TOOL_CALL_TIMEOUTS}", flush=True)
            for phase, reqs in sorted(by_phase.items(), key=lambda x: -len(x[1])):
                sample = ', '.join(reqs[:5])
                suffix = f'... +{len(reqs)-5} more' if len(reqs) > 5 else ''
                print(f"  {phase}: {len(reqs)} [{sample}{suffix}]", flush=True)

        except Exception as e:
            print(f"[DIAG-AGENT] monitor error: {e}", flush=True)


# ========== END DIAGNOSTIC ==========


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None


class SimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SimpleAgent(SimpleResponsesAPIAgent):
    config: SimpleAgentConfig
    _tool_call_timeout: float = None

    def setup_webserver(self):
        app = super().setup_webserver()

        # Read max_execution_time from the resources server config and set tool call timeout
        try:
            rs_config = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.resources_server.name,
            )
            max_exec_time = rs_config.get("max_execution_time", None)
            if max_exec_time is not None:
                self._tool_call_timeout = float(max_exec_time) + 5.0
                print(f"[DIAG-AGENT] Tool call timeout set to {self._tool_call_timeout}s (max_execution_time={max_exec_time} + 5)", flush=True)
            else:
                print("[DIAG-AGENT] No max_execution_time in resources server config, tool call timeout disabled", flush=True)
        except Exception as e:
            print(f"[DIAG-AGENT] Could not read max_execution_time from config: {e}", flush=True)

        @app.on_event("startup")
        async def start_diag_monitor():
            asyncio.create_task(_diag_async_monitor())

        return app

    async def _tool_call_with_timeout(self, output_function_call, resources_server_cookies):
        """Execute a tool call with optional timeout to prevent zombie connection hangs."""
        coro = self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=f"/{output_function_call.name}",
            json=json.loads(output_function_call.arguments),
            cookies=resources_server_cookies,
        )
        if self._tool_call_timeout is not None:
            return await asyncio.wait_for(coro, timeout=self._tool_call_timeout)
        return await coro

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        global _REQUEST_COUNTER, _TOOL_CALL_TIMEOUTS
        _REQUEST_COUNTER += 1
        rid = _REQUEST_COUNTER
        _ACTIVE_REQUESTS[rid] = {'start_time': time.time(), 'phase': 'init', 'step': 0, 'detail': ''}

        try:
            body = body.model_copy(deep=True)

            if isinstance(body.input, str):
                body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

            new_outputs = []
            step = 0
            model_server_cookies = None  # update the cookies on every model response
            resources_server_cookies = request.cookies  # update the cookies on every resources server response

            while True:
                step += 1
                _ACTIVE_REQUESTS[rid]['step'] = step
                _ACTIVE_REQUESTS[rid]['phase'] = 'model_call'
                _ACTIVE_REQUESTS[rid]['detail'] = ''

                new_body = body.model_copy(update={"input": body.input + new_outputs})

                model_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/v1/responses",
                    json=new_body,
                    cookies=model_server_cookies,
                )
                # We raise for status here since we expect model calls to always work.
                await raise_for_status(model_response)
                _ACTIVE_REQUESTS[rid]['phase'] = 'parse_model_resp'
                model_response_json = await get_response_json(model_response)
                model_server_cookies = model_response.cookies
                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as e:
                    raise RuntimeError(
                        f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                    ) from e

                output = model_response.output
                new_outputs.extend(output)

                if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                    break

                all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
                all_output_messages: List[NeMoGymResponseOutputMessage] = [
                    o for o in output if o.type == "message" and o.role == "assistant"
                ]
                if not all_fn_calls and all_output_messages:
                    break

                for i, output_function_call in enumerate(all_fn_calls):
                    _ACTIVE_REQUESTS[rid]['phase'] = 'tool_call'
                    _ACTIVE_REQUESTS[rid]['detail'] = f'{output_function_call.name}[{i+1}/{len(all_fn_calls)}]'

                    try:
                        api_response = await self._tool_call_with_timeout(output_function_call, resources_server_cookies)
                    except asyncio.TimeoutError:
                        _TOOL_CALL_TIMEOUTS += 1
                        print(f"[DIAG-AGENT] TIMEOUT r{rid} step={step} tool={output_function_call.name} after {self._tool_call_timeout}s (total timeouts: {_TOOL_CALL_TIMEOUTS})", flush=True)
                        tool_response = NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=output_function_call.call_id,
                            output=json.dumps({"success": False, "error_message": f"Tool call timed out after {self._tool_call_timeout}s"}),
                        )
                        new_outputs.append(tool_response)
                        continue

                    # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                    resources_server_cookies = api_response.cookies

                    tool_response = NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=output_function_call.call_id,
                        output=(await api_response.content.read()).decode(),
                    )
                    new_outputs.append(tool_response)

                # Check if max steps is not None and if we have exhausted it.
                if self.config.max_steps and step >= self.config.max_steps:
                    break

            # Propogate any extra cookies necessary for downstream verification
            for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
                response.set_cookie(k, v)

            model_response.output = new_outputs
            return model_response

        finally:
            _ACTIVE_REQUESTS.pop(rid, None)

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        global _REQUEST_COUNTER
        _REQUEST_COUNTER += 1
        rid = _REQUEST_COUNTER
        _ACTIVE_REQUESTS[rid] = {'start_time': time.time(), 'phase': 'run:seed_session', 'step': 0, 'detail': ''}

        try:
            cookies = request.cookies

            _ACTIVE_REQUESTS[rid]['phase'] = 'run:seed_session'
            seed_session_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_session_response)
            cookies = seed_session_response.cookies

            _ACTIVE_REQUESTS[rid]['phase'] = 'run:responses'
            response = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(response)
            cookies = response.cookies

            _ACTIVE_REQUESTS[rid]['phase'] = 'run:verify'
            verify_request = SimpleAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": await get_response_json(response)}
            )

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            return SimpleAgentVerifyResponse.model_validate(await get_response_json(verify_response))

        finally:
            _ACTIVE_REQUESTS.pop(rid, None)


if __name__ == "__main__":
    SimpleAgent.run_webserver()
