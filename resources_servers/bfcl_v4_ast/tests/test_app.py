# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for bfcl_v4_ast resource server.

These tests exercise the row-formatting and metric-aggregation paths
without invoking the bfcl_eval grader subprocess (which requires the
upstream package to be installed and is covered by an integration probe).
"""

import json
from unittest.mock import patch

import pytest

from resources_servers.bfcl_v4_ast.app import (
    BfclV4AstResourcesServer,
    BfclV4AstResourcesServerConfig,
    BfclV4AstVerifyRequest,
)


@pytest.fixture
def server():
    return BfclV4AstResourcesServer(config=BfclV4AstResourcesServerConfig())


@pytest.mark.asyncio
async def test_verify_returns_placeholder_reward(server):
    body = BfclV4AstVerifyRequest(
        responses_create_params={"input": []},
        response={"output_text": ""},
        id="simple_python_0",
        test_category="simple_python",
        predicted_tool_calls=[{"name": "foo", "arguments": {"x": 1}}],
        predicted_text="",
    )
    response = await server.verify(body)
    assert response.reward == 0.0
    assert response.id == "simple_python_0"
    assert response.test_category == "simple_python"
    assert response.predicted_tool_calls == [{"name": "foo", "arguments": {"x": 1}}]


def test_to_bfcl_result_row_with_calls():
    rollout = {
        "id": "simple_python_3",
        "predicted_tool_calls": [
            {"name": "math_add", "arguments": {"a": 1, "b": 2}},
            {"name": "math_mul", "arguments": '{"a": 3, "b": 4}'},
        ],
        "predicted_text": "",
    }
    row = BfclV4AstResourcesServer._to_bfcl_result_row(rollout)
    assert row["id"] == "simple_python_3"
    assert isinstance(row["result"], list)
    assert row["result"][0] == {"math_add": json.dumps({"a": 1, "b": 2})}
    # String args are passed through verbatim.
    assert row["result"][1] == {"math_mul": '{"a": 3, "b": 4}'}


def test_to_bfcl_result_row_no_calls_passes_text():
    rollout = {
        "id": "irrelevance_0",
        "predicted_tool_calls": [],
        "predicted_text": "I cannot help with that.",
    }
    row = BfclV4AstResourcesServer._to_bfcl_result_row(rollout)
    assert row["result"] == "I cannot help with that."


def test_score_fn_maps_is_correct_to_accuracy():
    assert BfclV4AstResourcesServer._score_fn({"is_correct": True}) == {"accuracy": 1.0}
    assert BfclV4AstResourcesServer._score_fn({"is_correct": False}) == {"accuracy": 0.0}
    # Missing field defaults to wrong, mirroring "no grader output -> wrong".
    assert BfclV4AstResourcesServer._score_fn({}) == {"accuracy": 0.0}


def test_compute_metrics_groups_by_category(server, tmp_path):
    """compute_metrics must call the bfcl_eval subprocess once per
    test_category and stamp is_correct on each rollout."""
    tasks = [
        [
            {
                "id": "simple_python_0",
                "test_category": "simple_python",
                "predicted_tool_calls": [],
                "predicted_text": "",
            },
            {
                "id": "simple_python_0",
                "test_category": "simple_python",
                "predicted_tool_calls": [],
                "predicted_text": "",
            },
        ],
        [
            {
                "id": "irrelevance_5",
                "test_category": "irrelevance",
                "predicted_tool_calls": [],
                "predicted_text": "no",
            },
        ],
    ]

    fake_calls = []

    def fake_run(self_, category, rows, work_dir):
        fake_calls.append(category)
        # Mark first row of each category as wrong.
        return {rows[0]["id"]} if rows else set()

    with patch.object(BfclV4AstResourcesServer, "_run_bfcl_eval_for_category", fake_run):
        metrics = server.compute_metrics(tasks)

    assert sorted(fake_calls) == ["irrelevance", "simple_python"]
    # First simple_python row was marked wrong; second was marked right.
    assert tasks[0][0]["is_correct"] is False
    assert tasks[0][1]["is_correct"] is True
    # Irrelevance row marked wrong.
    assert tasks[1][0]["is_correct"] is False

    # Top-level pass@1 keys exist.
    pass1_keys = [k for k in metrics if k.startswith("pass@1")]
    assert pass1_keys, f"Expected pass@1 keys, got {list(metrics.keys())}"

    # Per-category subset metrics present.
    assert any(k.startswith("simple_python/") for k in metrics)
    assert any(k.startswith("irrelevance/") for k in metrics)
