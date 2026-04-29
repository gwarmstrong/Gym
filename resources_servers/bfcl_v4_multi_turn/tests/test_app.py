# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bfcl_v4_multi_turn resource server."""

from unittest.mock import patch

import pytest

from resources_servers.bfcl_v4_multi_turn.app import (
    BfclV4MultiTurnResourcesServer,
    BfclV4MultiTurnResourcesServerConfig,
    BfclV4MultiTurnVerifyRequest,
)


@pytest.fixture
def server():
    return BfclV4MultiTurnResourcesServer(config=BfclV4MultiTurnResourcesServerConfig())


@pytest.mark.asyncio
async def test_verify_returns_placeholder(server):
    body = BfclV4MultiTurnVerifyRequest(
        responses_create_params={"input": []},
        response={"output_text": ""},
        id="multi_turn_base_0",
        test_category="multi_turn_base",
        generation=[[[{"foo": '{"x":1}'}]]],
    )
    response = await server.verify(body)
    assert response.reward == 0.0
    assert response.id == "multi_turn_base_0"
    assert response.test_category == "multi_turn_base"


def test_to_bfcl_result_row_passes_generation_through():
    rollout = {
        "id": "memory_kv_0",
        "generation": [[[{"foo": "{}"}], [{"bar": "{}"}]], [[{"baz": "{}"}]]],
    }
    row = BfclV4MultiTurnResourcesServer._to_bfcl_result_row(rollout)
    assert row["id"] == "memory_kv_0"
    assert row["result"] == rollout["generation"]


def test_score_fn():
    fn = BfclV4MultiTurnResourcesServer._score_fn
    assert fn({"is_correct": True}) == {"accuracy": 1.0}
    assert fn({"is_correct": False}) == {"accuracy": 0.0}
    assert fn({}) == {"accuracy": 0.0}


def test_compute_metrics_groups_by_category(server):
    tasks = [
        [
            {"id": "multi_turn_base_0", "test_category": "multi_turn_base", "generation": []},
        ],
        [
            {"id": "memory_kv_0", "test_category": "memory_kv", "generation": []},
        ],
    ]

    seen = []

    def fake_run(self_, category, rows, work_dir):
        seen.append(category)
        return set()  # everything correct

    with patch.object(BfclV4MultiTurnResourcesServer, "_run_bfcl_eval_for_category", fake_run):
        metrics = server.compute_metrics(tasks)

    assert sorted(seen) == ["memory_kv", "multi_turn_base"]
    assert tasks[0][0]["is_correct"] is True
    assert tasks[1][0]["is_correct"] is True
    assert any(k.startswith("multi_turn_base/") for k in metrics)
    assert any(k.startswith("memory_kv/") for k in metrics)
