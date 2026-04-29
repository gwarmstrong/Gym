# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for bfcl_v4_ast_agent.

Heavy lifting (BFCL FC handler instantiation, vLLM round-trip) is
covered by the on-cluster B5 probe. Here we test pure-Python helpers.
"""

from responses_api_agents.bfcl_v4_ast_agent.app import _strip_reasoning


def test_strip_reasoning_removes_think_block():
    text = "<think>scratch work\nmore work</think>\n[func()]"
    assert _strip_reasoning(text) == "[func()]"


def test_strip_reasoning_no_block_passes_through():
    text = "[func(a=1)]"
    assert _strip_reasoning(text) == "[func(a=1)]"


def test_strip_reasoning_handles_none():
    assert _strip_reasoning(None) == ""


def test_strip_reasoning_handles_empty():
    assert _strip_reasoning("") == ""


def test_strip_reasoning_multiple_blocks():
    text = "<think>a</think>between<think>b</think>final"
    # Multi-block: strip both. Implementation removes each block
    # plus trailing whitespace.
    assert _strip_reasoning(text) == "betweenfinal"
