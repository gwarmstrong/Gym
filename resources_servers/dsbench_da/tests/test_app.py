# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the dsbench_da resources server.

Mirrors the verification semantics of NeMo Skills'
``DSBenchEvaluator`` (`nemo_skills/evaluation/evaluator/dsbench.py`):

  * ``extract_dsbench_answer`` — regex (last match) preferred, ``\\boxed{...}``
    fallback. Regex is the DSBench-specific
    ``r"(?:The final answer is |\\boxed=)(.+)$"``.
  * ``relaxed_equal`` — JSON-parse, recurse on dict/list, MCQ, math_equal.
  * ``DSBenchDAResourcesServer.verify`` — runs math_equal first, falls back
    to relaxed_equal; populates ``extracted_answer`` so the metrics layer
    can compute ``no_answer`` correctly.
  * ``compute_metrics`` / ``get_key_metrics`` — Tier 1 pass@k pipeline.
"""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from pytest import approx, fixture

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.dsbench_da.app import (
    DSBenchDAResourcesServer,
    DSBenchDAResourcesServerConfig,
    DSBenchDAVerifyRequest,
    extract_dsbench_answer,
    relaxed_equal,
)


DSBENCH_REGEX = r"(?:The final answer is |\\boxed=)(.+)$"


def _make_response(text: str) -> Dict[str, Any]:
    return NeMoGymResponse(
        id="r1",
        created_at=1.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg1",
                role="assistant",
                status="completed",
                type="message",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump()


class TestExtractDSBenchAnswer:
    def test_boxed_only(self):
        assert extract_dsbench_answer("therefore \\boxed{42}", DSBENCH_REGEX) == "42"

    def test_boxed_with_braces_inside(self):
        # Nested braces (e.g. dict answer) — boxed extractor must respect brace balance.
        assert extract_dsbench_answer('result \\boxed{{"x": 10, "y": 20}}', DSBENCH_REGEX) == '{"x": 10, "y": 20}'

    def test_regex_the_final_answer_is(self):
        # Regex match wins over boxed when both present (relaxed=True semantics).
        out = extract_dsbench_answer("blah \\boxed{12}\nThe final answer is 42", DSBENCH_REGEX)
        assert out == "42"

    def test_regex_boxed_equals_form(self):
        assert extract_dsbench_answer("answer: \\boxed=42", DSBENCH_REGEX) == "42"

    def test_regex_takes_last_match(self):
        text = "The final answer is 1\nThe final answer is 2\nThe final answer is 3"
        assert extract_dsbench_answer(text, DSBENCH_REGEX) == "3"

    def test_no_match_returns_none(self):
        assert extract_dsbench_answer("totally unrelated text", DSBENCH_REGEX) is None

    def test_unclosed_boxed_returns_none(self):
        assert extract_dsbench_answer("\\boxed{42 with no close", DSBENCH_REGEX) is None


class TestRelaxedEqual:
    def test_predicted_none_returns_false_when_gt_present(self):
        assert relaxed_equal("42", None) is False

    def test_predicted_none_returns_true_when_gt_none(self):
        assert relaxed_equal(None, None) is True

    def test_dict_subset_match(self):
        # GT is subset; predicted may have extra keys.
        assert relaxed_equal('{"x": 10}', '{"x": 10, "y": 20}') is True

    def test_dict_subset_value_mismatch(self):
        assert relaxed_equal('{"x": 10}', '{"x": 11}') is False

    def test_dict_predicted_against_scalar_gt(self):
        # When predicted is dict and gt is scalar, succeed if any value matches.
        assert relaxed_equal("10", '{"x": 10, "y": 99}') is True
        assert relaxed_equal("99", '{"x": 10, "y": 99}') is True
        assert relaxed_equal("0", '{"x": 10, "y": 99}') is False

    def test_list_exact_length_and_elementwise(self):
        assert relaxed_equal("[1, 2, 3]", "[1, 2, 3]") is True
        assert relaxed_equal("[1, 2, 3]", "[1, 2, 4]") is False
        # Length mismatch fails.
        assert relaxed_equal("[1, 2, 3]", "[1, 2, 3, 4]") is False

    def test_list_predicted_against_scalar_gt(self):
        assert relaxed_equal("3", "[1, 2, 3]") is True
        assert relaxed_equal("99", "[1, 2, 3]") is False

    def test_mcq_case_insensitive(self):
        assert relaxed_equal("A", "a") is True
        assert relaxed_equal("B", "B") is True
        assert relaxed_equal("A", "B") is False

    def test_mcq_with_surrounding_whitespace(self):
        assert relaxed_equal("A", "  a  ") is True

    def test_math_equal_fallback(self):
        # Numeric equivalence via Skills' math_equal — handles leading zeros, etc.
        assert relaxed_equal("16", "016") is True

    def test_dict_value_uses_relaxed_recursion(self):
        # Inner dict values get relaxed-equal treatment, including math.
        assert relaxed_equal('{"answer": "16"}', '{"answer": "016"}') is True


class TestVerify:
    @fixture
    def config(self) -> DSBenchDAResourcesServerConfig:
        return DSBenchDAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")

    @fixture
    def server(self, config) -> DSBenchDAResourcesServer:
        srv = DSBenchDAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        return srv

    def _make_request(self, text: str, expected_answer: str) -> DSBenchDAVerifyRequest:
        return DSBenchDAVerifyRequest.model_validate(
            {
                "responses_create_params": {"input": []},
                "response": _make_response(text),
                "question": "irrelevant for verify",
                "expected_answer": expected_answer,
            }
        )

    @pytest.mark.asyncio
    async def test_correct_int_via_boxed(self, server):
        body = self._make_request("Working...\n\\boxed{42}", "42")
        resp = await server.verify(body)
        assert resp.reward == 1.0
        assert resp.extracted_answer == "42"
        assert resp.symbolic_correct is True
        assert resp.relaxed_correct is False

    @pytest.mark.asyncio
    async def test_correct_dict_via_relaxed_only(self, server):
        # Predicted is a SUPERSET of expected (extra y key) — math_equal can't
        # see them as equal, but relaxed_equal recognises the GT as a subset.
        body = self._make_request('Final: \\boxed{{"x": 10, "y": 20, "z": 99}}', '{"x": 10, "y": 20}')
        resp = await server.verify(body)
        assert resp.reward == 1.0
        assert resp.symbolic_correct is False
        assert resp.relaxed_correct is True

    @pytest.mark.asyncio
    async def test_correct_mcq_case_insensitive(self, server):
        body = self._make_request("therefore \\boxed{a}", "A")
        resp = await server.verify(body)
        assert resp.reward == 1.0

    @pytest.mark.asyncio
    async def test_incorrect(self, server):
        body = self._make_request("\\boxed{99}", "42")
        resp = await server.verify(body)
        assert resp.reward == 0.0
        assert resp.extracted_answer == "99"
        assert resp.symbolic_correct is False
        assert resp.relaxed_correct is False

    @pytest.mark.asyncio
    async def test_no_answer_extractable(self, server):
        body = self._make_request("I have no idea", "42")
        resp = await server.verify(body)
        assert resp.reward == 0.0
        assert resp.extracted_answer is None

    @pytest.mark.asyncio
    async def test_regex_fallback_form(self, server):
        body = self._make_request("after work, The final answer is 42", "42")
        resp = await server.verify(body)
        assert resp.reward == 1.0
        assert resp.extracted_answer == "42"


class TestMetrics:
    @fixture
    def server(self) -> DSBenchDAResourcesServer:
        config = DSBenchDAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        return DSBenchDAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_compute_metrics_pass_at_k(self, server):
        tasks = [
            [
                {"reward": 1.0, "extracted_answer": "42"},
                {"reward": 0.0, "extracted_answer": "41"},
            ],
            [
                {"reward": 1.0, "extracted_answer": "A"},
                {"reward": 1.0, "extracted_answer": "A"},
            ],
        ]
        metrics = server.compute_metrics(tasks)
        # pass@1[avg-of-1]/accuracy = (1.0 + 1.0) / 2 = 1.0 → 100%
        assert metrics["pass@1[avg-of-1]/accuracy"] == approx(100.0)
        # pass@1[avg-of-2]/accuracy = (0.5 + 1.0) / 2 = 0.75 → 75%
        assert metrics["pass@1[avg-of-2]/accuracy"] == approx(75.0)
        # pass@2/accuracy: task1 passes (1 of 2), task2 passes (2 of 2) → 100%
        assert metrics["pass@2/accuracy"] == approx(100.0)
        # majority@2 with answer_key="extracted_answer"
        assert "majority@2/accuracy" in metrics

    def test_compute_metrics_handles_no_answer(self, server):
        tasks = [
            [
                {"reward": 0.0, "extracted_answer": None},
                {"reward": 1.0, "extracted_answer": "42"},
            ]
        ]
        metrics = server.compute_metrics(tasks)
        # pass@k uses "any rollout has no_answer=1" → at k=1: 1/2 of first rollouts
        # have no_answer → 50%; at k=2: at least one of the two does → 100%.
        assert metrics["pass@1/no_answer"] == approx(50.0)
        assert metrics["pass@2/no_answer"] == approx(100.0)
        # avg-of-2 across rollouts: (1 + 0) / 2 = 0.5 → 50%.
        assert metrics["pass@1[avg-of-2]/no_answer"] == approx(50.0)

    def test_get_key_metrics_picks_highest_k(self, server):
        agent_metrics = {
            "pass@1[avg-of-1]/accuracy": 50.0,
            "pass@1[avg-of-2]/accuracy": 60.0,
            "pass@1[avg-of-4]/accuracy": 70.0,
            "pass@1/accuracy": 50.0,
            "pass@2/accuracy": 60.0,
            "pass@4/accuracy": 80.0,
            "majority@1/accuracy": 50.0,
            "majority@4/accuracy": 70.0,
            "mean/input_tokens": 100,
            "mean/output_tokens": 200,
        }
        key = server.get_key_metrics(agent_metrics)
        # Should select highest-k entries.
        assert key.get("pass@1[avg-of-4]/accuracy") == 70.0
        assert key.get("pass@4/accuracy") == 80.0
        assert key.get("majority@4/accuracy") == 70.0
        assert key.get("mean/input_tokens") == 100
        assert key.get("mean/output_tokens") == 200
