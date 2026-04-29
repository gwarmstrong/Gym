# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-turn agent helpers.

Full per-turn loop is exercised by the on-cluster B5 probe (requires
bfcl_eval + Gorilla backends in the agent venv).
"""

from responses_api_agents.bfcl_v4_multi_turn_agent.app import (
    BfclV4MultiTurnAgent,
    _is_long_context,
    _is_memory,
    _strip_reasoning,
)


def test_is_memory_categories():
    assert _is_memory("memory_kv")
    assert _is_memory("memory_vector")
    assert _is_memory("memory_rec_sum")
    assert not _is_memory("multi_turn_base")


def test_is_long_context():
    assert _is_long_context("multi_turn_long_context")
    assert _is_long_context("multi_turn_composite")
    assert not _is_long_context("multi_turn_base")


def test_strip_reasoning_strips_think_block():
    assert _strip_reasoning("<think>scratch</think>\n[func()]") == "[func()]"
    assert _strip_reasoning("[func()]") == "[func()]"
    assert _strip_reasoning(None) == ""


def test_format_function_call_dict_list_dict_args():
    out = BfclV4MultiTurnAgent._format_function_call_dict_list([{"name": "foo", "arguments": {"x": 1}}])
    assert out == [{"foo": '{"x": 1}'}]


def test_format_function_call_dict_list_string_args_passthrough():
    out = BfclV4MultiTurnAgent._format_function_call_dict_list([{"name": "foo", "arguments": '{"x": 1}'}])
    assert out == [{"foo": '{"x": 1}'}]


def test_format_function_call_dict_list_multiple():
    out = BfclV4MultiTurnAgent._format_function_call_dict_list(
        [
            {"name": "a", "arguments": {"k": 1}},
            {"name": "b", "arguments": {}},
        ]
    )
    assert out == [{"a": '{"k": 1}'}, {"b": "{}"}]
