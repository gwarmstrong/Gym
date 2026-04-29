# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BFCL v4 multi-turn resource server.

Single resource server shared by the multi_turn, memory, and web_search
benchmarks. Same batched-grading pattern as bfcl_v4_ast: per-rollout
verify() returns placeholder reward, compute_metrics() shells out to
`python -m bfcl_eval evaluate` per test_category.

The agent (bfcl_v4_multi_turn_agent) runs the per-turn execute-and-feed-back
loop and stores the per-turn `generation` shape that BFCL's multi-turn
checker consumes. This server is just the post-loop grader.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)


LOG = logging.getLogger(__name__)
DEFAULT_MODEL_HANDLER = "Qwen/Qwen3-8B-FC"


class BfclV4MultiTurnResourcesServerConfig(BaseResourcesServerConfig):
    model_handler: str = DEFAULT_MODEL_HANDLER
    eval_timeout: int = 1800  # multi-turn graders are slower than AST


class BfclV4MultiTurnVerifyRequest(BaseVerifyRequest):
    # Override BaseVerifyRequest.response (NeMoGymResponse) — see
    # bfcl_v4_ast resource server for rationale.
    response: Dict[str, Any] = Field(default_factory=dict)
    id: str
    test_category: str
    # The per-turn shape Skills' multi-turn flow stores: list[turn] of
    # list[step] of either str (no tool calls that step) or list of
    # {name: args_json} dicts.
    generation: List[List[Any]] = Field(default_factory=list)
    error: Optional[str] = None


class BfclV4MultiTurnVerifyResponse(BaseVerifyResponse):
    id: str
    test_category: str
    generation: List[List[Any]] = Field(default_factory=list)
    error: Optional[str] = None
    is_correct: Optional[bool] = None


class BfclV4MultiTurnResourcesServer(SimpleResourcesServer):
    config: BfclV4MultiTurnResourcesServerConfig

    def model_post_init(self, __context) -> None:
        # Re-install bfcl_eval after `uv sync` strips it. Same pattern as
        # bfcl_v4_ast resource server.
        from resources_servers.bfcl_v4_multi_turn._ensure_bfcl_eval import (
            ensure_bfcl_eval_installed,
        )

        ensure_bfcl_eval_installed()
        super().model_post_init(__context)

    async def verify(self, body: BfclV4MultiTurnVerifyRequest) -> BfclV4MultiTurnVerifyResponse:
        return BfclV4MultiTurnVerifyResponse(**body.model_dump(), reward=0.0)

    @staticmethod
    def _to_bfcl_result_row(rollout: Dict[str, Any]) -> Dict[str, Any]:
        # Multi-turn BFCL expects {"id", "result"} where `result` is the
        # per-turn list — same structure as `generation` in our verify()
        # request. Pass through.
        return {
            "id": rollout["id"],
            "result": rollout.get("generation", []),
        }

    def _run_bfcl_eval_for_category(self, category: str, rows: List[Dict[str, Any]], work_dir: Path) -> set[str]:
        from bfcl_eval.utils import get_directory_structure_by_category

        model_name = self.config.model_handler.replace("/", "_")
        result_dir = work_dir / "result" / model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"BFCL_v4_{category}_result.json"
        with result_file.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        env = os.environ.copy()
        env.setdefault("OPENAI_API_KEY", "dummy")
        env["BFCL_PROJECT_ROOT"] = str(work_dir)

        cmd = [
            sys.executable,
            "-m",
            "bfcl_eval",
            "evaluate",
            "--model",
            self.config.model_handler,
            "--test-category",
            category,
        ]
        LOG.info("Running BFCL eval: %s (cwd=%s)", " ".join(cmd), work_dir)
        subprocess.run(
            cmd,
            check=True,
            timeout=self.config.eval_timeout,
            env=env,
            cwd=str(work_dir),
        )

        score_file = (
            work_dir
            / "score"
            / model_name
            / get_directory_structure_by_category(category)
            / f"BFCL_v4_{category}_score.json"
        )
        wrong_ids: set[str] = set()
        with score_file.open("r") as fh:
            first = True
            for line in fh:
                if first:
                    first = False
                    continue
                wrong_ids.add(json.loads(line)["id"])
        return wrong_ids

    @staticmethod
    def _score_fn(rollout: Dict[str, Any]) -> Dict[str, float]:
        return {"accuracy": float(bool(rollout.get("is_correct")))}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for task_rollouts in tasks:
            for rollout in task_rollouts:
                cat = rollout.get("test_category")
                if not cat:
                    continue
                by_category.setdefault(cat, []).append(rollout)

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            for category, rollouts in by_category.items():
                bfcl_rows = [self._to_bfcl_result_row(r) for r in rollouts]
                try:
                    wrong_ids = self._run_bfcl_eval_for_category(category, bfcl_rows, work_dir)
                except subprocess.CalledProcessError as exc:
                    LOG.error("bfcl_eval failed for %s: %s", category, exc)
                    for r in rollouts:
                        r["is_correct"] = False
                    continue
                for r, bfcl_row in zip(rollouts, bfcl_rows):
                    r["is_correct"] = bfcl_row["id"] not in wrong_ids

        metrics, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn, answer_key=None)
        subset = compute_subset_metrics(tasks, field="test_category", score_fn=self._score_fn, answer_key=None)
        metrics.update(subset)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        return key


if __name__ == "__main__":
    BfclV4MultiTurnResourcesServer.run_webserver()
