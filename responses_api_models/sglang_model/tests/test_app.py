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
"""Tests for ``responses_api_models.sglang_model.app``.

Scope: only the behavioral delta vs ``vllm_model`` — the ``tool_choice``
auto-injection in the chat-completions request body. Parent-class tests in
``responses_api_models/vllm_model/tests/test_app.py`` cover the rest of the
Responses↔Chat-Completions plumbing, which we inherit unchanged.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import Request

from nemo_gym.server_utils import ServerClient
from responses_api_models.sglang_model.app import SGLangModel, SGLangModelConfig


def _make_model(**config_overrides) -> SGLangModel:
    cfg_kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="sglang_model",
        base_url="http://localhost:30000/v1",
        api_key="dummy",  # pragma: allowlist secret
        model="dummy-model",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
    )
    cfg_kwargs.update(config_overrides)
    config = SGLangModelConfig(**cfg_kwargs)
    return SGLangModel(config=config, server_client=MagicMock(spec=ServerClient))


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]


class TestToolChoiceAutoInjection:
    def test_injected_when_tools_present_and_choice_unset(self) -> None:
        model = _make_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "what's it like in sf?"}],
            "tools": _TOOLS,
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(spec=Request), body)
        assert result["tool_choice"] == "auto"

    def test_not_injected_when_no_tools(self) -> None:
        model = _make_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(spec=Request), body)
        assert "tool_choice" not in result

    def test_not_injected_when_tools_empty_list(self) -> None:
        model = _make_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(spec=Request), body)
        assert "tool_choice" not in result

    @pytest.mark.parametrize(
        "explicit_choice",
        [
            "none",
            "required",
            {"type": "function", "function": {"name": "get_weather"}},
        ],
    )
    def test_explicit_caller_choice_is_preserved(self, explicit_choice) -> None:
        model = _make_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": explicit_choice,
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(spec=Request), body)
        assert result["tool_choice"] == explicit_choice


class TestUnsupportedConfigsRaise:
    def test_return_token_id_information_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="return_token_id_information"):
            _make_model(return_token_id_information=True)

    def test_is_responses_native_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="is_responses_native"):
            _make_model(is_responses_native=True)
