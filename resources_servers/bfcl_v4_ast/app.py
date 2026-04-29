# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BFCL v4 AST family resource server.

Wraps the `bfcl_eval` package's grader (same package Skills uses). Each
rollout's `verify()` records the parsed tool calls + per-row metadata.
Actual AST grading runs in batch in `compute_metrics()` — one bfcl_eval
subprocess invocation per test_category over all rollouts in that
category — mirroring `nemo_skills/evaluation/evaluator/bfcl.py::eval_bfcl`.
This avoids paying BFCL's startup cost per-rollout.
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
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


LOG = logging.getLogger(__name__)

# Same handler Skills CI pairs with Qwen3-4B / Qwen3-8B; lives in
# bfcl_eval.constants.model_config.local_inference_model_map.
DEFAULT_MODEL_HANDLER = "Qwen/Qwen3-8B-FC"


class BfclV4AstResourcesServerConfig(BaseResourcesServerConfig):
    # Handler key in bfcl_eval.local_inference_model_map. Must match the
    # one the agent uses for parsing — otherwise the parsed tool calls
    # the agent emits and the format the grader expects will diverge.
    model_handler: str = DEFAULT_MODEL_HANDLER
    # Subprocess timeout (seconds) for `python -m bfcl_eval evaluate` per category.
    eval_timeout: int = 600


class BfclV4AstVerifyRequest(BaseVerifyRequest):
    # Override BaseVerifyRequest.response (NeMoGymResponse) — BFCL flow
    # carries the parsed predicted_tool_calls/predicted_text forward;
    # the full Responses-API response object is unused.
    response: Dict[str, Any] = Field(default_factory=dict)
    id: str
    test_category: str
    # Parsed structured tool calls produced by the agent's BFCL FC handler.
    # Each entry: {"name": str, "arguments": dict|str (JSON)}.
    predicted_tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    # The textual content the model emitted (after reasoning strip),
    # included for debugging and so the grader's relevance/irrelevance
    # checks can see whether the model abstained vs answered.
    predicted_text: str = ""


class BfclV4AstVerifyResponse(BaseVerifyResponse):
    id: str
    test_category: str
    predicted_tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    predicted_text: str = ""
    # Filled in compute_metrics(); per-rollout verify() returns 0.0 as a
    # placeholder. Downstream consumers that need per-rollout correctness
    # should read the metrics JSON or re-run the aggregator script.
    is_correct: Optional[bool] = None


class BfclV4AstResourcesServer(SimpleResourcesServer):
    config: BfclV4AstResourcesServerConfig

    def model_post_init(self, __context) -> None:
        # Re-install bfcl_eval after the rollout client's `uv sync`
        # stripped it on startup. compute_metrics() shells out to the
        # bfcl_eval CLI, and the agent's parser also imports the package.
        from resources_servers.bfcl_v4_ast._ensure_bfcl_eval import (
            ensure_bfcl_eval_installed,
        )

        ensure_bfcl_eval_installed()
        super().model_post_init(__context)

    async def verify(self, body: BfclV4AstVerifyRequest) -> BfclV4AstVerifyResponse:
        # Per-rollout reward is a placeholder — actual grading is batched
        # in compute_metrics(). We still return everything needed for that
        # batch step.
        return BfclV4AstVerifyResponse(
            **body.model_dump(),
            reward=0.0,
        )

    @staticmethod
    def _to_bfcl_result_row(rollout: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one Gym rollout dict to BFCL's expected per-row format.

        BFCL evaluate reads JSON lines like:
          {"id": "<id>", "result": [{"<name>": "<args_json>"}, ...]}
        where `result` is a list of single-key dicts mapping function name
        to JSON-encoded arguments string.
        """
        result_calls = []
        for fc in rollout.get("predicted_tool_calls", []) or []:
            name = fc.get("name", "")
            args = fc.get("arguments", {})
            args_str = args if isinstance(args, str) else json.dumps(args)
            result_calls.append({name: args_str})

        # When the model abstained (no tool calls), pass through raw text
        # so the irrelevance/relevance checkers see the textual answer.
        if not result_calls:
            result_calls = rollout.get("predicted_text", "") or ""

        return {"id": rollout["id"], "result": result_calls}

    def _run_bfcl_eval_for_category(
        self,
        category: str,
        rows: List[Dict[str, Any]],
        work_dir: Path,
    ) -> set[str]:
        """Run `python -m bfcl_eval evaluate` for one category. Returns wrong-id set."""
        from bfcl_eval.utils import get_directory_structure_by_category

        model_name = self.config.model_handler.replace("/", "_")
        result_dir = work_dir / "result" / model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"BFCL_v4_{category}_result.json"
        with result_file.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        env = os.environ.copy()
        # bfcl_eval requires SOME OpenAI key for handler init even though
        # we only need its parser/checker — same trick Skills uses.
        env.setdefault("OPENAI_API_KEY", "dummy")
        # Tell bfcl_eval where to read/write so it doesn't touch /opt/gorilla.
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
        # Group all rollouts by test_category, run bfcl_eval once per
        # category, fill in is_correct on each rollout in place.
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

        # Now compute pass@k / pass@1[avg-of-k] / majority@k from is_correct.
        metrics, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key=None,  # AST has no single "answer string" to majority-vote on
        )

        # Per-category subset metrics so we can roll up to the v4 leaderboard
        # weighted score in the recipe-side aggregator.
        from nemo_gym.reward_profile import compute_subset_metrics

        subset = compute_subset_metrics(
            tasks,
            field="test_category",
            score_fn=self._score_fn,
            answer_key=None,
        )
        metrics.update(subset)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        return key


if __name__ == "__main__":
    BfclV4AstResourcesServer.run_webserver()
