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
import setuptools


dependencies = [
    "nemo-gym[dev]",
    # The sibling sglang_model package contains the SGLangModel adapter we
    # subclass. local_sglang_model is the launcher; sglang_model is the
    # routing/adapter layer.
    "sglang-model",
    # SGLang itself. Pinned to the version smoke-tested on aws-iad
    # (nemo-skills-sglang-latest.sqsh, 2026-04-28).
    # License: Apache 2.0 https://github.com/sgl-project/sglang/blob/main/LICENSE
    "sglang==0.5.10.post1",
    # hf_transfer for faster model download from HuggingFace
    # License: Apache 2.0 https://github.com/huggingface/hf_transfer
    "hf_transfer",
    # uvicorn — Gym's server lifecycle expects this exact pin (matches
    # local_vllm_model).
    "uvicorn==0.40.0",
]


setuptools.setup(install_requires=dependencies)
