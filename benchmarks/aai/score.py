"""AAI composite score from Gym's per-agent aggregate metrics.

Reads `<rollouts>_aggregate_metrics.json` (the list produced by ng_collect_rollouts)
and computes `overall_score`, `math_score`, `code_score` per the AAI methodology:
https://artificialanalysis.ai/methodology/intelligence-benchmarking

Skills bug parity: aai_score.py reads metrics["aime24"]["pass@1[avg-of-10]"] even though
the aai BENCHMARKS dict declares "aime25". The Gym port uses aime25 (the actually-rolled-out
benchmark) and flags the inconsistency. To exactly reproduce Skills' buggy lookup, set
--reproduce-skills-aime-key-bug.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# Map AAI sub-benchmark short names → (Gym agent_name, score-fn key)
AAI_SUBS = {
    "mmlu_pro": ("mmlu_pro_mcqa_simple_agent", "accuracy"),
    "hle": ("hle_equivalence_llm_judge_simple_agent", "accuracy"),
    "gpqa": ("gpqa_mcqa_simple_agent", "accuracy"),
    "aime25": ("aime25_math_with_judge_simple_agent", "judge_accuracy"),
    "livecodebench": ("livecodebench_v5_2407_2412_code_gen_simple_agent", "accuracy"),
    "ifbench": ("ifbench_benchmark_simple_agent", "accuracy"),
    # scicode skipped per migration scope.
    # aalcr DROPPED — see benchmarks/aai/config.yaml comment.
}


def _find_pass_at_1(agent_metrics: Dict[str, float], score_key: str) -> Optional[float]:
    """Find the pass@1[avg-of-N]/<score_key> entry, picking the highest N if multiple."""
    matching = []
    for k, v in agent_metrics.items():
        if k.startswith("pass@1[avg-of-") and k.endswith(f"]/{score_key}"):
            try:
                n = int(k[len("pass@1[avg-of-") : k.rindex("]/")])
                matching.append((n, v))
            except ValueError:
                continue
    if not matching:
        # Fall back to plain pass@1/<score_key> (deterministic / num_repeats=1)
        plain = agent_metrics.get(f"pass@1/{score_key}")
        return plain
    matching.sort(reverse=True)
    return matching[0][1]


def _agent_by_name(aggregate_metrics: List[Dict[str, Any]], agent_name: str) -> Optional[Dict[str, Any]]:
    for entry in aggregate_metrics:
        if entry.get("agent_ref", {}).get("name") == agent_name:
            return entry
    return None


def compute_score(aggregate_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """Reproduce nemo_skills.dataset.aai.aai_score.compute_score against Gym output."""
    sub_scores: Dict[str, Optional[float]] = {}
    for short, (agent_name, score_key) in AAI_SUBS.items():
        entry = _agent_by_name(aggregate_metrics, agent_name)
        if entry is None:
            sub_scores[short] = None
            continue
        sub_scores[short] = _find_pass_at_1(entry.get("agent_metrics", {}), score_key)

    present = {k: v for k, v in sub_scores.items() if v is not None}
    overall = sum(present.values()) / len(present) if present else 0.0
    math_score = sub_scores["aime25"] if sub_scores.get("aime25") is not None else 0.0
    # Skills: code_score = (scicode + livecodebench) / 2. scicode is skipped here,
    # so report code_score as livecodebench only and annotate.
    code_score = sub_scores["livecodebench"] if sub_scores.get("livecodebench") is not None else 0.0

    return {
        "overall_score": overall,
        "math_score": math_score,
        "code_score_livecodebench_only": code_score,
        **{k: (v if v is not None else float("nan")) for k, v in sub_scores.items()},
        "n_subs_present": len(present),
        "n_subs_expected_full_aai": 8,
        "n_subs_expected_this_migration": 6,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-metrics", required=True, help="Path to <rollouts>_aggregate_metrics.json")
    ap.add_argument("--output", default=None, help="Where to write the composite score JSON (default: stdout)")
    args = ap.parse_args()

    with open(args.aggregate_metrics) as f:
        aggregate = json.load(f)

    result = compute_score(aggregate)
    out_text = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(out_text + "\n")
        print(f"Wrote AAI composite score to {args.output}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
