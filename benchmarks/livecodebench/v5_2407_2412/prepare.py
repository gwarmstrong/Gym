# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""LiveCodeBench v5 (Jul 2024 – Dec 2024).

Matches Skills' `test_v5_2407_2412` split used by the AAI benchmark suite.
"""

from pathlib import Path

from benchmarks.livecodebench.prepare_utils import prepare_from_hf_raw


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FPATH = DATA_DIR / "livecodebench_v5_2407_2412_validation.jsonl"


def prepare() -> Path:
    return prepare_from_hf_raw(
        OUTPUT_FPATH, release_version="release_v5", date_from="2024-07-01", date_to="2025-01-01"
    )


if __name__ == "__main__":
    prepare()
