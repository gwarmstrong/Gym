# WMT24++ Translation Benchmark

English to {de_DE, es_MX, fr_FR, it_IT, ja_JP} segment-level translation
from [`google/wmt24pp`](https://huggingface.co/datasets/google/wmt24pp).

Verification is deterministic corpus-level BLEU (sacrebleu) per language
pair, with cross-pair aggregations `en->xx`, `xx->xx`, and `xx->{tgt}`.
Optionally augments with xCOMET-XXL neural QE scores when
`compute_comet: true` is set on the wmt_translation server.

See `resources_servers/wmt_translation/README.md` for the verifier
details and the Ray GPU-scheduled COMET path.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/wmt24pp/config.yaml]"
```

In addition to writing `data/wmt24pp_benchmark.jsonl`, the prepare step
pre-fetches the xCOMET-XXL checkpoint and its xlm-roberta-xxl tokenizer
into `HF_HOME` (when `unbabel-comet` is installed in the active env).
That keeps the resource server's Ray actors fully offline at runtime —
no HF Hub calls during `verify()`, no rate-limit retries.

## Running servers

The xCOMET-XXL actor pool requires the `extra_gpu` Ray resource, which
must be advertised by extra nodes that joined Ray with `--num-gpus=0
--resources='{"extra_gpu": N}'`. Two paths set this up: the all-Gym
NeMo-RL `ray.sub` path described below (a small `EXTRA_GPU_NODES` diff
to `ray.sub`, mirroring aviary's `SETUP_COMMAND` pattern), or
NeMo-Skills' `get_ray_server_cmd` via `ns nemo_gym_rollouts
--server_type vllm_dp_ray`. Local / single-node runs disable COMET via
Hydra override and rely on corpus-BLEU only; xCOMET scoring still works
end-to-end on the cluster path:

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/wmt24pp/config.yaml"
ng_run "+config_paths=[$config_paths]" \
    "++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=false"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=wmt24pp_wmt_translation_simple_agent \
    +prompt_config=benchmarks/wmt24pp/prompts/default.yaml \
    +input_jsonl_fpath=benchmarks/wmt24pp/data/wmt24pp_benchmark.jsonl \
    +output_jsonl_fpath=results/wmt24pp_rollouts.jsonl \
    +num_repeats=4
```

## End-to-end reproduction on a SLURM cluster (all-Gym)

This path runs the full benchmark — vLLM DP serving plus the xCOMET-XXL
actor pool — on a single SLURM allocation against NeMo-RL's `ray.sub`,
without going through NeMo-Skills. The verifier node(s) join Ray with
`--num-gpus=0 --resources='{"extra_gpu": N}'`: vLLM's DP placement
ignores their GPUs (so model bundles can't accidentally land on a
verifier node) while wmt_translation's CometActor pool schedules onto
them via the matching `resources={"extra_gpu": 1}` request.

`ray.sub` ships uniform-cluster bring-up only — every worker advertises
the full `GPUS_PER_NODE` count — so a small patch is needed to expose
the verifier-masking topology. Apply this diff to your local
NeMo-RL `ray.sub` (the same place aviary documents its `SETUP_COMMAND`
extension):

```diff
diff --git a/ray.sub b/ray.sub
--- a/ray.sub
+++ b/ray.sub
@@ -50,6 +50,11 @@ maybe_gres_arg() {
 CONTAINER=$CONTAINER
 MOUNTS=$MOUNTS
 COMMAND=${COMMAND:-}  # This is a script relative to the SLURM_SUBMIT_DIR. If left empty, it will leave the cluster idle after it's brought up.
+EXTRA_GPU_NODES=${EXTRA_GPU_NODES:-0}  # If > 0, the last N workers in the
+# allocation join Ray with --num-gpus=0 and advertise the custom 'extra_gpu'
+# resource instead. Lets a GPU-side verifier pool coexist with a DP-on-Ray
+# vLLM on the same allocation without competing for its placement-group
+# bundles. Used by Gym's wmt_translation resource server (xCOMET-XXL).
 ########################################################
@@ -301,6 +306,16 @@ NUM_ACTORS=$((GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
 # Start from node 1 since node 0 is running the head
 for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
   node_i=${nodes_array[$i]}

+  # Decide this worker's Ray resource string. extra_gpu workers hide their
+  # GPUs from Ray's GPU accounting so DP placement-group bundles can't land
+  # on them; they re-advertise the same GPUs under the custom 'extra_gpu'
+  # resource that @ray.remote(resources={"extra_gpu": 1}) actors can request.
+  if (( i >= SLURM_JOB_NUM_NODES - EXTRA_GPU_NODES )); then
+    worker_resources_arg="--num-gpus=0 --resources=\"{\\\"extra_gpu\\\": $GPUS_PER_NODE, \\\"slurm_managed_ray_cluster\\\": 1}\""
+  else
+    worker_resources_arg="--resources=\"{\\\"worker_units\\\": $GPUS_PER_NODE, \\\"slurm_managed_ray_cluster\\\": 1}\""
+  fi
+
   worker_cmd=$(cat <<EOF
@@ -365,7 +380,7 @@ log-sync-sidecar &
 cat <<EOFINNER | tee /launch-worker.sh
 ray start --address "$ip_head" \
           --disable-usage-stats \
-          --resources="{\"worker_units\": $GPUS_PER_NODE, \"slurm_managed_ray_cluster\": 1}" \
+          $worker_resources_arg \
           --min-worker-port=${MIN_WORKER_PORT} \
           --max-worker-port=${MAX_WORKER_PORT} \
           \
```

### One-time preparation

```bash
# Writes the benchmark JSONL and pre-fetches Unbabel/XCOMET-XXL +
# facebook/xlm-roberta-xxl into HF_HOME so the CometActors stay offline.
ng_prepare_benchmark "+config_paths=[benchmarks/wmt24pp/config.yaml]"
```

### Container requirements

`ray.sub` runs a single container on every node. For this path the image
must contain, at the system level:

- `vllm` and `ray` compatible with `LocalVLLMModel` (vLLM 0.18.x +
  Ray 2.52.x is the validated combination)
- `nemo-gym` (so `ng_e2e_collect_rollouts` is on `PATH`)
- `unbabel-comet>=2.2` and `sacrebleu[ja,ko]>=2.4` for the wmt_translation
  resource server

The set of pins above is roughly `pip install vllm==0.18.1
"ray[default]==2.52.1" -e <gym-repo> "unbabel-comet>=2.2"
"sacrebleu[ja,ko]>=2.4"` applied on top of `vllm/vllm-openai:v0.18.1`.

### Launch

```bash
# 5 nodes = 4 vLLM serving + 1 xCOMET-XXL verifier.
# Bumps to bigger topologies just scale --nodes and EXTRA_GPU_NODES.

read -r -d '' COMMAND <<'EOF'
ng_e2e_collect_rollouts \
    "+config_paths=[responses_api_models/local_vllm_model/configs/deepseek-ai/DeepSeek-V2-Lite.yaml,benchmarks/wmt24pp/config.yaml]" \
    ++output_jsonl_fpath=results/wmt24pp_rollouts.jsonl \
    ++split=benchmark \
    ++num_repeats=1 \
    ++num_samples_in_parallel=32 \
    ++reuse_existing_data_preparation=true \
    ++limit=20 \
    ++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=true \
    ++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.comet_num_shards=8
EOF

COMMAND="$COMMAND" \
CONTAINER=<image-with-vllm+gym+comet> \
MOUNTS="$PWD:$PWD" \
EXTRA_GPU_NODES=1 \
sbatch \
    --nodes=5 --gres=gpu:8 --time=01:00:00 \
    --account=<your-account> --partition=<your-partition> \
    --job-name=wmt24pp_4plus1 \
    ray.sub
```

The model yaml's `vllm_serve_kwargs` controls `data_parallel_size` /
`tensor_parallel_size`; the canned `DeepSeek-V2-Lite.yaml` ships TP=2
DP=2. If a single replica spans more than one node (e.g. `TP=16` across
two model nodes), set `VLLM_RAY_DP_PACK_STRATEGY=span` in the model
yaml's `vllm_serve_env_vars`.

`ng_e2e_collect_rollouts` runs on the Ray head (rank 0) inside the
container; `initialize_ray()` discovers the cluster automatically via
`ray.init(address="auto")` so no extra address plumbing is needed.

Output artifacts land under `results/`: `wmt24pp_rollouts.jsonl` carries
per-row `sentence_bleu` and `comet_score`; the corresponding
`*_aggregate_metrics.json` carries per-pair (`en->de_DE/comet`) and
cross-pair (`en->xx/comet`, `xx->xx/comet`) aggregations.

## End-to-end reproduction on a SLURM cluster (via NeMo-Skills)

The commands above assume the Gym head and a Ray cluster have been
brought up manually. For a fully reproducible run on SLURM, use
NeMo-Skills' `ns nemo_gym_rollouts` CLI — it handles vLLM bring-up
with the right Ray topology (the `vllm_dp_ray` server type reserves a
hidden `extra_gpu` node for the streaming xCOMET-XXL actor pool),
placement-group setup, and Gym launch in one shot.

### One-time setup

```bash
# 1. Install NeMo-Skills (provides the `ns` CLI). Pinned to the SHA where
#    server_type=vllm_dp_ray landed on main; bump as new Skills releases tag.
pip install git+https://github.com/NVIDIA-NeMo/Skills.git@f57b1735b
```

#### Cluster config

Define a cluster config at `cluster_configs/<your-cluster>.yaml`. See
[the NeMo-Skills cluster-configs docs](https://nvidia-nemo.github.io/Skills/basics/cluster-configs/)
for the full schema; this recipe needs:

| Field | Why | Example |
|---|---|---|
| `executor: slurm` | recipe is SLURM-only | `slurm` |
| `ssh_tunnel:` | so `ns` submits jobs over SSH | `host:`, `user:`, `job_dir:`, `identity:` |
| `account` / `partition` / `cpu_partition` | SLURM accounting + queue | per-cluster values |
| `containers.vllm_dp_ray` | the policy server image | based on `vllm/vllm-openai:v0.18.1` (or any 0.18.x with `ray>=2.48`) |
| `containers.nemo-gym` | the Gym + xCOMET-XXL image | any image with `nemo-gym[dev]` + `unbabel-comet` installed |
| `containers.sandbox` | required by `ns nemo_gym_rollouts` even though wmt24pp doesn't sandbox | any NeMo-Skills sandbox image |
| `mounts:` | a directory for the HF cache (paired with the `HF_HOME` env var below — Skills requires both) and a writable workspace dir for `--output_dir` | `<host-hf-dir>:/models`, `<host-workspace>:/workspace` |
| `env_vars:` | `HF_HOME=<in-container-path>` is **required** by Skills and must point inside one of your mounts (so the COMET prefetch survives across jobs). Optionally bump `VLLM_ENGINE_READY_TIMEOUT_S` above vLLM's 600s default if your model's cold-load (weights + KV-cache init + warmup) exceeds it — typical triggers are very large models or cross-node TP. | `HF_HOME=/models/hf-cache` (matching the `:/models` mount above) |

The two container fields that aren't trivial:

- **`vllm_dp_ray`**: any vLLM 0.18.x image. The Skills repo ships
  `dockerfiles/Dockerfile.vllm` which builds on `vllm/vllm-openai:v0.18.1`
  with `ray[cgraph]` + audio/Qwen-VL extras. **The bundled `ray` version
  in this image MUST match the `ray` resolved by `nemo-gym`'s `uv.lock`**
  — cross-container Ray-cluster joins fail with `ConnectionError: Could
  not read 'temp_dir' from GCS` on protocol mismatch.
- **`nemo-gym`**: any image where `pip install -e <gym>[dev]` resolves
  cleanly AND has `unbabel-comet`, `torch>=2.5`, `sacrebleu` baked in.
  The lazy-install path in `resources_servers/wmt_translation/.venv`
  works as a fallback but adds 2–3 min to first-job startup.

#### Prepare benchmark data on the cluster

The local `ng_prepare_benchmark` from [above](#prepare-benchmark-data)
writes the JSONL to your dev workstation. For a SLURM run, the JSONL
plus the `Unbabel/XCOMET-XXL` cache need to live on the cluster's
filesystem. Dispatch the prepare via `ns run_cmd` with the `nemo-gym`
container (which has `unbabel-comet` so the prefetch step actually
runs):

```bash
ns run_cmd \
    --cluster <your-cluster> \
    --container nemo-gym \
    --expname wmt24pp_prepare \
    --command 'ng_prepare_benchmark "+config_paths=[benchmarks/wmt24pp/config.yaml]"'
```

This populates `benchmarks/wmt24pp/data/wmt24pp_benchmark.jsonl` and
prefetches `Unbabel/XCOMET-XXL` + its `xlm-roberta-xxl` tokenizer into
the cluster's `HF_HOME`. Subsequent rollout jobs read both from the
shared filesystem.

### 2-node smoke topology (1 model node + 1 extra_gpu COMET node)

Sized to fit on an `interactive` partition for fast iteration. Bump
`server_nodes` for DP>1 (`server_nodes = dp_size + num_extra_gpu_nodes`)
and switch to a batch partition for larger evaluations.

```bash
# Pick a translation-capable policy model accessible from your cluster.
# The PR's parity numbers come from nvidia/Nemotron-3-Nano-30B-A3B-BF16.
MODEL="nvidia/Nemotron-3-Nano-30B-A3B-BF16"

ns nemo_gym_rollouts \
    --cluster <your-cluster> \
    --partition interactive \
    --server_type vllm_dp_ray \
    --server_gpus 8 \
    --server_nodes 2 \
    --server_args "--tensor-parallel-size 8 --data-parallel-size 1 --data-parallel-size-local 1 --data-parallel-backend ray --distributed-executor-backend ray --api-server-count 1 --reasoning-parser deepseek_r1 --trust-remote-code --dtype auto --enforce-eager" \
    --model "$MODEL" \
    --config_paths "benchmarks/wmt24pp/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml" \
    --input_file benchmarks/wmt24pp/data/wmt24pp_benchmark.jsonl \
    --output_dir /workspace/wmt24pp_smoke \
    --expname wmt24pp_smoke \
    -- \
    +agent_name=wmt24pp_wmt_translation_simple_agent \
    +prompt_config=benchmarks/wmt24pp/prompts/default.yaml \
    +num_repeats=1 \
    +limit=20 \
    +num_samples_in_parallel=64 \
    ++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=true
```

`--reasoning-parser deepseek_r1` is required for the Nemotron-3-Nano
family above; drop it for non-reasoning models.

The job allocates 2 nodes (1 model node hosting `server_gpus` vLLM
workers + 1 extra_gpu node hosting the xCOMET-XXL actor pool), starts
vLLM in DP-on-Ray mode on the model node, schedules the actor pool onto
the extra node via the custom `extra_gpu` Ray resource, and writes
`rollouts.jsonl` (with per-row `comet_score`) plus
`rollouts_aggregate_metrics.json` to `--output_dir`.
