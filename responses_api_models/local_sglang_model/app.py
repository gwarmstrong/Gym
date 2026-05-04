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
"""LocalSGLangModel — spin up an SGLang OpenAI-compatible server inside the
Gym head process via a Ray actor and route the ``SGLangModel`` adapter at it.

Compared to ``local_vllm_model``, the launcher is intentionally simpler:

* Subprocess launch (``python -m sglang.launch_server``) — SGLang already runs
  its scheduler / tokenizer workers as subprocesses, so isolating the launcher
  itself avoids the signal-handler, uvicorn-logger and DP-placement-group
  monkeypatches that ``local_vllm_model`` carries.
* Standard Ray GPU accounting: the actor requests ``num_gpus = tp_size * pp_size``,
  Ray sets ``CUDA_VISIBLE_DEVICES`` and the subprocess inherits it.

Single-instance, single-node only in this first cut. Multi-instance / multi-node
SGLang DP needs a placement-group strategy similar to ``local_vllm_model`` and
is deferred.
"""

import os
import subprocess
import sys
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Union

import ray
import requests
from pydantic import Field
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from requests.exceptions import ConnectionError

from nemo_gym.global_config import (
    DISALLOWED_PORTS_KEY_NAME,
    find_open_port,
    get_global_config_dict,
    get_hf_token,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.sglang_model.app import SGLangModel, SGLangModelConfig


class LocalSGLangModelConfig(SGLangModelConfig):
    # Inherited from SGLangModelConfig but optional here — populated after server
    # spinup. Keeping the type the same as the parent (Union[str, List[str]])
    # so VLLMModel._post_init's iteration over base_url keeps working.
    base_url: Union[str, List[str]] = Field(default_factory=list)
    # Not used on local deployments
    api_key: str = "dummy"  # pragma: allowlist secret

    hf_home: Optional[str] = None
    sglang_serve_kwargs: Dict[str, Any]
    sglang_serve_env_vars: Dict[str, str] = Field(default_factory=dict)

    ray_worker_py_executable: str = sys.executable

    debug: bool = False

    def model_post_init(self, context):
        # Default to .cache/huggingface in the cwd, mirroring local_vllm_model.
        if not self.hf_home:
            self.hf_home = str(Path.cwd() / ".cache" / "huggingface")
        return super().model_post_init(context)


def _kwargs_to_cli_args(kwargs: Dict[str, Any]) -> List[str]:
    """Translate a Hydra-friendly kwargs dict into SGLang CLI args.

    * ``{"tp_size": 8}`` → ``["--tp-size", "8"]``
    * ``{"trust_remote_code": True}`` → ``["--trust-remote-code"]``
    * ``{"trust_remote_code": False}`` → ``[]`` (omitted; SGLang's CLI flags
      are off-by-default booleans, so explicitly ``False`` is just "don't pass it")
    * ``{"chat_template": None}`` → ``[]`` (skip, SGLang resolves defaults)
    """
    cli: List[str] = []
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            # False booleans are omitted; SGLang's flags are off-by-default.
            continue
        if isinstance(value, (list, tuple)):
            # SGLang accepts ``--foo a --foo b`` for repeated flags; passthrough.
            for item in value:
                cli.extend([flag, str(item)])
            continue
        cli.extend([flag, str(value)])
    return cli


def _extract_required_size(kwargs: Dict[str, Any], key_underscore: str, default: int = 1) -> int:
    """Read parallelism dimensions out of the kwargs dict, accepting either
    underscore (Hydra-friendly) or dash (CLI-style) keys."""
    if key_underscore in kwargs and kwargs[key_underscore] is not None:
        return int(kwargs[key_underscore])
    dash_key = key_underscore.replace("_", "-")
    if dash_key in kwargs and kwargs[dash_key] is not None:
        return int(kwargs[dash_key])
    return default


@ray.remote
class LocalSGLangModelActor:
    """Ray actor that supervises a single ``sglang.launch_server`` subprocess.

    The actor reserves ``num_gpus`` via the parent placement group; Ray sets
    ``CUDA_VISIBLE_DEVICES`` accordingly and the subprocess inherits it.
    """

    def __init__(
        self,
        cli_args: List[str],
        env_vars: Dict[str, str],
        port: int,
        server_name: str,
        debug: bool,
    ) -> None:
        self.server_name = server_name
        self.debug = debug

        node_ip = ray._private.services.get_node_ip_address()
        self._base_url = f"http://{node_ip}:{port}/v1"
        print(f"[{server_name}] Spinning up local SGLang server at {self._base_url}", file=sys.stderr)

        env = os.environ.copy()
        for k, v in env_vars.items():
            env[k] = v

        cmd = [sys.executable, "-m", "sglang.launch_server", *cli_args]
        if self.debug:
            redacted = dict(env_vars)
            if "HF_TOKEN" in redacted:
                redacted["HF_TOKEN"] = "****"
            print(f"[{server_name}] cmd: {' '.join(cmd)}", file=sys.stderr)
            print(f"[{server_name}] env overrides: {redacted}", file=sys.stderr)

        self.proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)

    def base_url(self) -> str:
        return self._base_url

    def is_alive(self) -> bool:
        return self.proc.poll() is None


class LocalSGLangModel(SGLangModel):
    config: LocalSGLangModelConfig

    # Mirrors local_vllm_model's declaration: leading-underscore attribute,
    # which Pydantic v2 treats as a PrivateAttr settable on instances.
    _local_sglang_model_actor: Any

    def setup_webserver(self):
        print("Starting SGLang server. This will take a few minutes...")
        self.start_sglang_server()
        return super().setup_webserver()

    def get_cache_dir(self) -> str:
        # HF cache layout: HF_HOME/hub/...
        return str(Path(self.config.hf_home) / "hub")

    def _build_cli_and_env(self) -> tuple[List[str], Dict[str, str], int, int]:
        kwargs = dict(self.config.sglang_serve_kwargs)

        tp_size = _extract_required_size(kwargs, "tp_size", default=1)
        pp_size = _extract_required_size(kwargs, "pp_size", default=1)
        dp_size = _extract_required_size(kwargs, "dp_size", default=1)
        if dp_size != 1:
            raise NotImplementedError(
                "LocalSGLangModel currently supports dp_size=1 only. "
                "Multi-instance / multi-node SGLang DP is deferred."
            )
        num_gpus = tp_size * pp_size

        port = find_open_port(disallowed_ports=get_global_config_dict()[DISALLOWED_PORTS_KEY_NAME])

        # Required overrides — set after the user-provided dict so they win.
        kwargs.update(
            {
                "model_path": self.config.model,
                "host": "0.0.0.0",  # cross-node addressable
                "port": port,
                "download_dir": self.get_cache_dir(),
            }
        )

        cli_args = _kwargs_to_cli_args(kwargs)

        env_vars: Dict[str, str] = {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": self.config.hf_home,
        }
        if hf_token := get_hf_token():
            env_vars["HF_TOKEN"] = hf_token
        env_vars.update(self.config.sglang_serve_env_vars)

        return cli_args, env_vars, port, num_gpus

    def _reserve_placement_group(self, num_gpus: int) -> PlacementGroup:
        # Single fat bundle so the actor's {GPU: num_gpus, CPU: 1} request
        # fits without crossing bundles. Ray rejects placement groups where
        # the actor resource request can't be satisfied by a single bundle,
        # even with STRICT_PACK across the group.
        bundles = [{"GPU": float(num_gpus), "CPU": 1.0}]
        pg = ray.util.placement_group(
            name=f"{self.config.name}_dp_rank_0",
            strategy="STRICT_PACK",
            bundles=bundles,
        )
        ray.get(pg.ready())
        return pg

    def start_sglang_server(self) -> None:
        cli_args, env_vars, port, num_gpus = self._build_cli_and_env()

        if self.config.debug:
            print(f"Final SGLang CLI args: {cli_args}")

        pg = self._reserve_placement_group(num_gpus)

        self._local_sglang_model_actor = LocalSGLangModelActor.options(
            num_gpus=num_gpus,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
            runtime_env=dict(
                py_executable=self.config.ray_worker_py_executable,
                env_vars=env_vars,
            ),
        ).remote(
            cli_args=cli_args,
            env_vars=env_vars,
            port=port,
            server_name=self.config.name,
            debug=self.config.debug,
        )

        self.config.base_url = [ray.get(self._local_sglang_model_actor.base_url.remote())]

        # Reset clients to point at the local URL.
        self._post_init()

        self.await_server_ready()

    def await_server_ready(self) -> None:
        poll_count = 0
        while True:
            is_alive = ray.get(self._local_sglang_model_actor.is_alive.remote())
            assert is_alive, f"{self.config.name} LocalSGLangModel server spinup failed; see the actor's stderr above."

            try:
                requests.get(url=f"{self.config.base_url[0]}/models", timeout=5)
                return
            except ConnectionError:
                if poll_count % 10 == 0:  # every 30s
                    print(f"Waiting for {self.config.name} LocalSGLangModel server to spin up...")
                poll_count += 1
                sleep(3)


if __name__ == "__main__":
    LocalSGLangModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = LocalSGLangModel.run_webserver()  # noqa: F401
