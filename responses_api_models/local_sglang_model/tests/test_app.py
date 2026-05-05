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
"""Unit tests for ``responses_api_models.local_sglang_model.app``.

Scope: pure functions that translate the Hydra config into the SGLang CLI.
The Ray-actor-driven server spinup is exercised by the e2e smoke test, not here.
"""

from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.server_utils import ServerClient
from responses_api_models.local_sglang_model.app import (
    LocalSGLangModel,
    LocalSGLangModelConfig,
    _extract_required_size,
    _kwargs_to_cli_args,
)


class TestKwargsToCliArgs:
    def test_underscore_to_dash(self) -> None:
        assert _kwargs_to_cli_args({"tp_size": 8}) == ["--tp-size", "8"]

    def test_bool_true_emits_flag(self) -> None:
        assert _kwargs_to_cli_args({"trust_remote_code": True}) == ["--trust-remote-code"]

    def test_bool_false_omitted(self) -> None:
        assert _kwargs_to_cli_args({"trust_remote_code": False}) == []

    def test_none_omitted(self) -> None:
        assert _kwargs_to_cli_args({"chat_template": None}) == []

    def test_list_value_repeated(self) -> None:
        assert _kwargs_to_cli_args({"foo": ["a", "b"]}) == ["--foo", "a", "--foo", "b"]

    def test_mixed(self) -> None:
        out = _kwargs_to_cli_args(
            {
                "tp_size": 8,
                "tool_call_parser": "qwen",
                "trust_remote_code": True,
                "chat_template": None,
                "mem_fraction_static": 0.7,
            }
        )
        # Order follows insertion order (Python 3.7+ dict).
        assert out == [
            "--tp-size",
            "8",
            "--tool-call-parser",
            "qwen",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.7",
        ]


class TestExtractRequiredSize:
    def test_underscore_key(self) -> None:
        assert _extract_required_size({"tp_size": 8}, "tp_size") == 8

    def test_dash_key(self) -> None:
        assert _extract_required_size({"tp-size": 4}, "tp_size") == 4

    def test_default_when_missing(self) -> None:
        assert _extract_required_size({}, "tp_size", default=1) == 1

    def test_none_falls_back_to_default(self) -> None:
        assert _extract_required_size({"tp_size": None}, "tp_size", default=1) == 1


def _make_config(**overrides) -> LocalSGLangModelConfig:
    cfg_kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="local_sglang_model",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        sglang_serve_kwargs={"tp_size": 1, "pp_size": 1, "dp_size": 1},
        sglang_serve_env_vars={},
    )
    cfg_kwargs.update(overrides)
    return LocalSGLangModelConfig(**cfg_kwargs)


class TestConfigureSGLangServe:
    @patch("responses_api_models.local_sglang_model.app.find_open_port", return_value=30000)
    @patch(
        "responses_api_models.local_sglang_model.app.get_global_config_dict",
        return_value={"disallowed_ports": []},
    )
    @patch("responses_api_models.local_sglang_model.app.get_hf_token", return_value=None)
    def test_basic(self, _hf_tok, _gcd, _port) -> None:
        config = _make_config(
            sglang_serve_kwargs={
                "tp_size": 1,
                "pp_size": 1,
                "dp_size": 1,
                "tool_call_parser": "qwen",
            },
            sglang_serve_env_vars={"SGLANG_FOO": "1"},
        )
        model = LocalSGLangModel(config=config, server_client=MagicMock(spec=ServerClient))
        cli_args, env_vars, port, num_gpus = model._configure_sglang_serve()

        assert port == 30000
        assert num_gpus == 1
        # Required overrides win over user-provided.
        assert "--model-path" in cli_args
        assert cli_args[cli_args.index("--model-path") + 1] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert "--host" in cli_args and cli_args[cli_args.index("--host") + 1] == "0.0.0.0"
        assert "--port" in cli_args and cli_args[cli_args.index("--port") + 1] == "30000"
        assert "--tool-call-parser" in cli_args
        assert env_vars["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
        assert env_vars["SGLANG_FOO"] == "1"
        assert env_vars["HF_HOME"]  # default cwd-relative path is set

    @patch("responses_api_models.local_sglang_model.app.find_open_port", return_value=30000)
    @patch(
        "responses_api_models.local_sglang_model.app.get_global_config_dict",
        return_value={"disallowed_ports": []},
    )
    @patch("responses_api_models.local_sglang_model.app.get_hf_token", return_value=None)
    def test_tp_pp_yields_correct_num_gpus(self, _hf_tok, _gcd, _port) -> None:
        config = _make_config(
            sglang_serve_kwargs={"tp_size": 4, "pp_size": 2, "dp_size": 1},
        )
        model = LocalSGLangModel(config=config, server_client=MagicMock(spec=ServerClient))
        _, _, _, num_gpus = model._configure_sglang_serve()
        assert num_gpus == 8

    @patch("responses_api_models.local_sglang_model.app.find_open_port", return_value=30000)
    @patch(
        "responses_api_models.local_sglang_model.app.get_global_config_dict",
        return_value={"disallowed_ports": []},
    )
    @patch("responses_api_models.local_sglang_model.app.get_hf_token", return_value=None)
    def test_dp_size_gt_one_raises(self, _hf_tok, _gcd, _port) -> None:
        config = _make_config(sglang_serve_kwargs={"tp_size": 1, "pp_size": 1, "dp_size": 2})
        model = LocalSGLangModel(config=config, server_client=MagicMock(spec=ServerClient))
        with pytest.raises(NotImplementedError, match="dp_size=1 only"):
            model._configure_sglang_serve()


class TestUnsupportedConfigsRaiseEarly:
    """Inherited from SGLangModel — re-asserted here because LocalSGLangModelConfig is its own type."""

    def test_return_token_id_information_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="return_token_id_information"):
            LocalSGLangModel(
                config=_make_config(return_token_id_information=True),
                server_client=MagicMock(spec=ServerClient),
            )

    def test_is_responses_native_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="is_responses_native"):
            LocalSGLangModel(
                config=_make_config(is_responses_native=True),
                server_client=MagicMock(spec=ServerClient),
            )
