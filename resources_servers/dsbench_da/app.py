# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""DSBench-DA resources server.

Reproduces NeMo Skills' DSBench-DA evaluator
(``nemo_skills/evaluation/evaluator/dsbench.py``):

  1. ``extract_answer(generation, relaxed=True)`` with the DSBench-specific
     ``extract_regex = r"(?:The final answer is |\\boxed=)(.+)$"``. ``relaxed=True``
     tries the regex first, falling back to ``\boxed{...}``.
  2. ``math_equal(expected, predicted)`` — Skills' MCQ-aware,
     latex-normalizing, math_verify-backed comparison.
  3. If ``math_equal`` returns False, fall back to ``relaxed_equal``:
     JSON-parse both sides and recurse on dict/list, then case-insensitive
     single-letter MCQ matching, then ``math_equal`` of stringified args.

Verification only — no python-tool execution. Tool execution is delegated
to the ``ns_tools`` server, which dispatches verify requests here via its
``verifiers`` indirection. Set ``verifier_type=dsbench_da`` per row in the
JSONL or wire ``ns_tools.default_verifier=dsbench_da`` in the config.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


logger = logging.getLogger(__name__)


# Vendored from nemo_skills.evaluation.math_grader so this server doesn't
# need a runtime dependency on the full nemo-skills install. Keep these in
# sync with Skills if its grader changes meaningfully.
_MCQ_OPTIONS = "ABCDEFGHIJ"


def _additional_normalization(expr: str) -> str:
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match = re.fullmatch(percentage_pattern, expr)
    if match:
        expr = match.group(1)
    return expr.rstrip(".\\")


def _math_equal(gt_answer, predicted_answer) -> bool:
    """Skills' math_equal, vendored verbatim (with take_modulo=None)."""
    if predicted_answer is None:
        return False

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)

    norm_gt_mcq = gt_answer.strip()
    is_mcq = re.fullmatch("|".join(_MCQ_OPTIONS), norm_gt_mcq)
    parsed_gt = parse(gt_answer, [StringExtractionConfig(strings=tuple(_MCQ_OPTIONS))])
    parsed_pred = parse(predicted_answer, [StringExtractionConfig(strings=tuple(_MCQ_OPTIONS))])
    if is_mcq and verify(parsed_gt, parsed_pred):
        return True

    gt_answer = _additional_normalization(gt_answer)
    predicted_answer = _additional_normalization(predicted_answer)

    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    if normalized_gt.replace(" ", "") == normalized_pred.replace(" ", ""):
        return True

    text_literal_pattern = r"[a-zA-Z ,]+"
    if re.fullmatch(text_literal_pattern, normalized_gt) and re.fullmatch(text_literal_pattern, normalized_pred):
        return False

    current_gt = gt_answer
    current_pred = predicted_answer
    latex_env_pattern = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    if not re.search(latex_env_pattern, current_gt, re.DOTALL):
        current_gt = f"${current_gt}$"
    if not re.search(latex_env_pattern, current_pred, re.DOTALL):
        current_pred = f"${current_pred}$"

    parsed_gt = parse(current_gt, [LatexExtractionConfig()])
    parsed_pred = parse(current_pred, [LatexExtractionConfig()])
    return verify(parsed_gt, parsed_pred)


def _search_regex(string: str, regex: str) -> Optional[str]:
    matches = re.findall(regex, string)
    return matches[-1] if matches else None


def _search_boxed(string: str) -> Optional[str]:
    if "\\boxed" not in string:
        return None
    idx = string.rfind("\\boxed")
    if idx < 0:  # pragma: no cover  -- unreachable: guarded by line above
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right = None
    open_count = 0
    while i < len(string):
        if string[i] == "{":
            open_count += 1
        if string[i] == "}":
            open_count -= 1
            if open_count == 0:
                right = i
                break
        i += 1
    if right is None:
        return None
    retval = string[idx : right + 1]
    left = "\\boxed{"
    if retval.startswith(left) and retval.endswith("}"):
        return retval[len(left) : -1]
    return None


def extract_dsbench_answer(generation: str, extract_regex: str) -> Optional[str]:
    """Skills' extract_answer with relaxed=True and DSBench's extract_regex.

    Tries the regex first (returns the LAST match), falls back to ``\\boxed{...}``.
    """
    return _search_regex(generation, extract_regex) or _search_boxed(generation)


def relaxed_equal(gt_answer: Any, predicted_answer: Any) -> bool:
    """DSBench fallback: JSON-parse, recurse, MCQ, math_equal.

    Vendored verbatim from ``nemo_skills/evaluation/evaluator/dsbench.py``.
    """
    if predicted_answer is None:
        return gt_answer is None

    try:
        predicted_answer = json.loads(predicted_answer)
    except Exception:
        pass
    try:
        gt_answer = json.loads(gt_answer)
    except Exception:
        pass

    if isinstance(predicted_answer, dict):
        if not isinstance(gt_answer, dict):
            return any(relaxed_equal(gt_answer, p) for p in predicted_answer.values())
        return all(
            k in predicted_answer and relaxed_equal(gt_answer[k], predicted_answer[k]) for k in gt_answer.keys()
        )

    if isinstance(predicted_answer, list):
        if not isinstance(gt_answer, list):
            return any(relaxed_equal(gt_answer, p) for p in predicted_answer)
        return len(gt_answer) == len(predicted_answer) and all(
            relaxed_equal(e, p) for e, p in zip(gt_answer, predicted_answer)
        )

    mcq_options = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    norm_gt = str(gt_answer).strip().upper()
    norm_pred = str(predicted_answer).strip().upper()
    if re.fullmatch("|".join(mcq_options), norm_gt):
        parsed_gt = parse(norm_gt, [StringExtractionConfig(strings=tuple(mcq_options))])
        parsed_pred = parse(norm_pred, [StringExtractionConfig(strings=tuple(mcq_options))])
        if verify(parsed_gt, parsed_pred):
            return True

    return _math_equal(str(gt_answer), str(predicted_answer))


class DSBenchDAResourcesServerConfig(BaseResourcesServerConfig):
    # DSBench-specific extractor pattern. Matches Skills' eval_config.extract_regex
    # set in DSBenchEvaluator.__init__.
    extract_regex: str = r"(?:The final answer is |\\boxed=)(.+)$"


class DSBenchDARunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    question: Optional[str] = None
    expected_answer: Optional[str] = None


class DSBenchDAVerifyRequest(DSBenchDARunRequest, BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class DSBenchDAVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    expected_answer: Optional[str] = None
    extracted_answer: Optional[str] = None
    symbolic_correct: bool = False
    relaxed_correct: bool = False


def _combine_assistant_text(response) -> str:
    parts: List[str] = []
    for item in response.output:
        if item.type != "message":
            continue
        for content in item.content:
            if content.type == "output_text":
                parts.append(content.text)
    return "".join(parts)


class DSBenchDAResourcesServer(SimpleResourcesServer):
    config: DSBenchDAResourcesServerConfig

    async def verify(self, body: DSBenchDAVerifyRequest) -> DSBenchDAVerifyResponse:
        generation = _combine_assistant_text(body.response)
        predicted = extract_dsbench_answer(generation, self.config.extract_regex)

        try:
            symbolic = _math_equal(body.expected_answer, predicted)
        except Exception as exc:
            logger.warning("math_equal raised %r — treating as False", exc)
            symbolic = False

        relaxed = False
        if not symbolic:
            try:
                relaxed = relaxed_equal(body.expected_answer, predicted)
            except Exception as exc:
                logger.warning("relaxed_equal raised %r — treating as False", exc)
                relaxed = False

        is_correct = symbolic or relaxed
        return DSBenchDAVerifyResponse(
            **body.model_dump(),
            reward=1.0 if is_correct else 0.0,
            extracted_answer=predicted,
            symbolic_correct=symbolic,
            relaxed_correct=relaxed,
        )

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        return {"accuracy": float(r.get("reward", 0.0))}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_answer",
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))
        return key


if __name__ == "__main__":
    DSBenchDAResourcesServer.run_webserver()
