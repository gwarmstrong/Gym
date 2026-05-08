#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end wmt24pp on SLURM, all-Gym (no NeMo-Skills dependency).
#
# Allocates SERVING_NODES + EXTRA_GPU_NODES nodes:
#   * Ranks [0, SERVING_NODES)            host vLLM workers (full GPU visibility)
#   * Ranks [SERVING_NODES, total_nodes)  host xCOMET-XXL verifier actors
#                                         (--num-gpus=0, advertise extra_gpu=N)
#
# The extra_gpu masking is what lets the verifier coexist with vLLM in
# the same Ray cluster without confusing vLLM's DP placement: vLLM only
# sees the serving nodes' GPUs, while wmt_translation's CometActor pool
# schedules onto the masked nodes via @ray.remote(resources={"extra_gpu": 1}).
#
# Rank 0 is both Ray head and workload runner (ng_e2e_collect_rollouts);
# all other ranks join Ray with the appropriate flags and block until the
# head signals teardown via a coordination file.
#
# This script is a self-contained alternative to Skills'
# `ns nemo_gym_rollouts --server_type vllm_dp_ray`, intended for users
# running benchmarks that need GPU-side verifiers entirely through Gym.
#
# --- Usage ---
#
#   # 1. Prepare wmt24pp data + xCOMET-XXL HF cache on the cluster
#   #    (one-time; see benchmarks/wmt24pp/README.md "Prepare benchmark data").
#
#   # 2. Submit:
#   sbatch \
#       --account=$ACCOUNT --partition=$PARTITION \
#       --nodes=2 --gres=gpu:8 --ntasks-per-node=1 \
#       --time=2:00:00 --job-name=wmt24pp_gym \
#       --export=ALL,GYM_DIR=$PWD,CONTAINER=$IMG_SQSH,HF_HOME_HOST=$HF_DIR,\
# WORKSPACE_HOST=$WS_DIR,MODEL_CONFIG=responses_api_models/local_vllm_model/configs/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml \
#       benchmarks/wmt24pp/scripts/submit_slurm.sh
#
# Required env vars:
#   GYM_DIR         host path to the Gym repo (mounted to /gym in container)
#   CONTAINER       enroot squashfs image with Gym + unbabel-comet baked in,
#                   or a docker:// URL if your site allows pyxis on-the-fly
#   HF_HOME_HOST    host HF cache dir (mounted to /hf-cache); must already
#                   contain Unbabel/XCOMET-XXL from `ng_prepare_benchmark`
#   WORKSPACE_HOST  host writable dir (mounted to /workspace) — output_dir
#   MODEL_CONFIG    path-to-yaml for the policy model (e.g. nvidia/NVIDIA-...)
#
# Optional:
#   SERVING_NODES        (default = SLURM_NNODES - 1)
#   EXTRA_GPU_NODES      (default = 1)
#   NUM_GPUS_PER_NODE    (default 8) must match --gres=gpu:N
#   COMET_NUM_SHARDS     (default = NUM_GPUS_PER_NODE * EXTRA_GPU_NODES)
#   NUM_REPEATS          (default 1)
#   NUM_SAMPLES_IN_PARALLEL (default 64)
#   LIMIT                (unset; set for smoke tests, e.g. 20)
#   PACK_STRATEGY        (default strict; use span if a single replica
#                         spans >1 node, e.g. TP=16 across 2 nodes)
#   READY_TIMEOUT_S      (default 180) seconds to wait for cluster to assemble

#SBATCH --ntasks-per-node=1

set -euo pipefail

# --- required ---
: "${GYM_DIR:?GYM_DIR must be set (host path to Gym repo)}"
: "${CONTAINER:?CONTAINER must be set (squashfs path or docker://...)}"
: "${HF_HOME_HOST:?HF_HOME_HOST must be set}"
: "${WORKSPACE_HOST:?WORKSPACE_HOST must be set}"
: "${MODEL_CONFIG:?MODEL_CONFIG must be set (yaml under responses_api_models/...)}"

# --- defaults ---
: "${SLURM_NNODES:?This script must run under sbatch}"
EXTRA_GPU_NODES="${EXTRA_GPU_NODES:-1}"
SERVING_NODES="${SERVING_NODES:-$((SLURM_NNODES - EXTRA_GPU_NODES))}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
COMET_NUM_SHARDS="${COMET_NUM_SHARDS:-$((NUM_GPUS_PER_NODE * EXTRA_GPU_NODES))}"
NUM_REPEATS="${NUM_REPEATS:-1}"
NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-64}"
PACK_STRATEGY="${PACK_STRATEGY:-strict}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-180}"

EXPECTED_NODES=$((SERVING_NODES + EXTRA_GPU_NODES))
if [ "$SLURM_NNODES" -ne "$EXPECTED_NODES" ]; then
    echo "ERROR: SLURM_NNODES=$SLURM_NNODES but expected $EXPECTED_NODES (SERVING_NODES=$SERVING_NODES + EXTRA_GPU_NODES=$EXTRA_GPU_NODES)" >&2
    exit 1
fi

HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
HEAD_IP="$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -i | awk '{print $1}')"
RAY_PORT=6379

# Coordination file lives on the shared filesystem so all ranks see it.
# Lives outside the container; mounted into /coord on each rank.
COORD_DIR="${COORD_DIR:-$WORKSPACE_HOST/.gym-slurm-$SLURM_JOB_ID}"
mkdir -p "$COORD_DIR"

echo "[submit] head=$HEAD_NODE ($HEAD_IP)  serving=$SERVING_NODES  extra_gpu=$EXTRA_GPU_NODES  total=$EXPECTED_NODES"

# Limit param expanded only when set (avoids `++limit=` empty override)
LIMIT_OVERRIDE=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_OVERRIDE="++limit=$LIMIT"
fi

srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --container-image="$CONTAINER" \
    --container-mounts="$GYM_DIR:/gym,$HF_HOME_HOST:/hf-cache,$WORKSPACE_HOST:/workspace,$COORD_DIR:/coord" \
    --container-workdir=/gym \
    bash -c "
        set -euo pipefail
        export HF_HOME=/hf-cache
        export RAY_DEDUP_LOGS=0
        export VLLM_RAY_DP_PACK_STRATEGY=$PACK_STRATEGY

        RANK=\${SLURM_PROCID}
        echo \"[rank \$RANK on \$(hostname)] starting\"

        if [ \"\$RANK\" -eq 0 ]; then
            ray start --head \\
                --node-ip-address=$HEAD_IP --port=$RAY_PORT \\
                --num-gpus=$NUM_GPUS_PER_NODE \\
                --disable-usage-stats
        elif [ \"\$RANK\" -lt $SERVING_NODES ]; then
            # Other serving nodes — give vLLM full visibility of their GPUs.
            sleep 15
            ray start --address=$HEAD_IP:$RAY_PORT \\
                --num-gpus=$NUM_GPUS_PER_NODE \\
                --disable-usage-stats
        else
            # Extra-GPU node — hide GPUs from Ray accounting and re-advertise
            # them under a custom 'extra_gpu' resource. wmt_translation's
            # CometActor pool schedules onto these nodes via
            # @ray.remote(resources={'extra_gpu': 1}); GPUs stay reachable
            # because we set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES on
            # the actor (see resources_servers/wmt_translation/app.py).
            sleep 15
            ray start --address=$HEAD_IP:$RAY_PORT \\
                --num-gpus=0 \\
                --resources='{\"extra_gpu\": $NUM_GPUS_PER_NODE}' \\
                --disable-usage-stats
        fi

        if [ \"\$RANK\" -eq 0 ]; then
            # Wait for all ranks to join. Poll via Ray's runtime context;
            # the GCS API is more stable across versions than parsing
            # 'ray status' text.
            echo '[head] waiting for cluster to assemble...'
            python -c \"
import os, sys, time, ray
ray.init(address='$HEAD_IP:$RAY_PORT', ignore_reinit_error=True)
deadline = time.time() + $READY_TIMEOUT_S
while True:
    n = sum(1 for _ in ray.nodes())
    alive = sum(1 for n in ray.nodes() if n['Alive'])
    if alive >= $EXPECTED_NODES:
        print(f'[head] cluster ready: {alive}/{$EXPECTED_NODES} alive')
        break
    if time.time() > deadline:
        print(f'[head] cluster failed to assemble: {alive}/{$EXPECTED_NODES} alive after $READY_TIMEOUT_S s', file=sys.stderr)
        sys.exit(1)
    time.sleep(2)
\"

            # Activate Gym venv (assumes container baked it; otherwise sync).
            if [ -d .venv ]; then
                source .venv/bin/activate
            else
                uv sync --extra dev
                source .venv/bin/activate
            fi

            # Run wmt24pp end-to-end against the existing Ray cluster.
            #   - +ray_head_node_address makes initialize_ray() join (not start)
            #   - compute_comet=true gates the xCOMET-XXL actor pool
            #   - reuse_existing_data_preparation skips re-running prepare.py
            ng_e2e_collect_rollouts \\
                \"+config_paths=[$MODEL_CONFIG,benchmarks/wmt24pp/config.yaml]\" \\
                ++output_jsonl_fpath=/workspace/wmt24pp_rollouts.jsonl \\
                ++split=benchmark \\
                ++num_repeats=$NUM_REPEATS \\
                ++num_samples_in_parallel=$NUM_SAMPLES_IN_PARALLEL \\
                ++ray_head_node_address=$HEAD_IP:$RAY_PORT \\
                ++reuse_existing_data_preparation=true \\
                $LIMIT_OVERRIDE \\
                ++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=true \\
                ++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.comet_num_shards=$COMET_NUM_SHARDS

            # Signal workers to tear down, then stop ray on the head.
            touch /coord/done
            ray stop
        else
            # Block until head signals completion, then stop ray on this rank.
            until [ -f /coord/done ]; do sleep 5; done
            ray stop
        fi

        echo \"[rank \$RANK] exiting\"
    "

# Tidy up the coordination dir on success. On failure we leave it in
# place to aid debugging.
rm -rf "$COORD_DIR"
echo "[submit] done — rollouts at $WORKSPACE_HOST/wmt24pp_rollouts.jsonl"
