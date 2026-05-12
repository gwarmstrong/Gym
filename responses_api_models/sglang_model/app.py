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
from typing import Any, Dict

from fastapi import Request

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class SGLangModelConfig(VLLMModelConfig):
    """Config for SGLang. Identical surface to VLLMModelConfig today; kept as a
    distinct type so tests/configs that check ``isinstance`` resolve unambiguously,
    and so future SGLang-only knobs can land here without leaking back into vLLM."""


class SGLangModel(VLLMModel):
    """SGLang model adapter that reuses VLLMModel's Responses<->Chat-Completions
    converter and only adjusts request shaping.

    Mirrors ``nemo_skills/inference/model/sglang.py``: SGLang requires
    ``tool_choice`` in the request body when tools are provided (vLLM accepts
    server CLI flag ``--enable-auto-tool-choice`` instead). Without this,
    SGLang returns a 400 / refuses to call the tool.
    """

    config: SGLangModelConfig

    def model_post_init(self, context):
        if self.config.return_token_id_information:
            # SGLang 0.5.x's OpenAI-compat /v1/chat/completions does not surface
            # per-token IDs in the logprobs response (return_tokens_as_token_ids
            # is silently ignored), and /tokenize accepts only {model, prompt} —
            # so the chat-style payload VLLMModel uses for prompt-token retrieval
            # is rejected. Workarounds (server /tokenize on the assistant content
            # + client-side chat-template apply) are implementable but carry
            # extra round trips and edge cases (EOS handling, BPE byte-fallback
            # collisions on multi-byte UTF-8 generation tokens). RL training
            # rollouts use local_vllm_model today, which gives exact IDs in-band.
            raise NotImplementedError(
                "SGLangModel does not support return_token_id_information=True. "
                "SGLang 0.5.x's OpenAI-compat layer does not surface per-token IDs "
                "and lifting the guard requires upstream support (filed as a "
                "follow-up). Use local_vllm_model for RL-training rollouts; "
                "SGLang is eval/inference-only today."
            )
        if self.config.is_responses_native:
            raise NotImplementedError(
                "SGLangModel does not yet support is_responses_native=True. Use the chat-completions path."
            )
        return super().model_post_init(context)

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)
        # SGLang requires tool_choice in the body when tools are provided.
        # Don't clobber an explicit caller value (e.g. "none", "required",
        # or a forced function selection).
        if body_dict.get("tools") and not body_dict.get("tool_choice"):
            body_dict["tool_choice"] = "auto"
        return body_dict


if __name__ == "__main__":
    SGLangModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SGLangModel.run_webserver()  # noqa: F401
