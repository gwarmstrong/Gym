# local_sglang_model

Spins up an [SGLang](https://github.com/sgl-project/sglang) OpenAI-compatible
server **inside the Gym head process** via a Ray actor and routes the
`SGLangModel` adapter (`responses_api_models/sglang_model`) at it.

Compared to `local_vllm_model`, the launcher is intentionally simpler:

* **Subprocess launch** (`python -m sglang.launch_server`) — SGLang already runs
  its scheduler / tokenizer workers as subprocesses, so isolating the launcher
  itself avoids the signal-handler / uvicorn-logger / Ray-DP-placement-group
  monkeypatches that `local_vllm_model` carries.
* **Standard Ray GPU accounting:** the actor requests
  `num_gpus = tp_size * pp_size`, Ray sets `CUDA_VISIBLE_DEVICES` and the
  subprocess inherits it. No `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`
  required.

## Scope

* Single-instance, single-node (`dp_size = 1`, `tp_size * pp_size <= node GPUs`).
* Multi-instance / multi-node SGLang DP is **deferred** — needs a placement-group
  strategy similar to `local_vllm_model`.
* Inherits `SGLangModel`'s `tool_choice` injection and the
  `return_token_id_information` / `is_responses_native` `NotImplementedError`
  guards.

## Config knobs

`sglang_serve_kwargs` is a kwargs dict translated to CLI flags for
`python -m sglang.launch_server` (`tp_size` → `--tp-size`, etc.):

```yaml
sglang_serve_kwargs:
  tp_size: 1
  pp_size: 1
  dp_size: 1
  mem_fraction_static: 0.7
  tool_call_parser: qwen
```

`sglang_serve_env_vars` is a flat dict of env vars to set in the actor's
runtime env before launch.

## Example run

Single-GPU smoke against Qwen2.5-1.5B-Instruct (config provided):

```bash
config_paths="\
resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/local_sglang_model/configs/Qwen/Qwen2.5-1.5B-Instruct.yaml"
ng_run "+config_paths=[${config_paths}]" &> temp.log &
```

Watch logs:
```bash
tail -f temp.log
```

Hit the agent client (expect a tool call):
```bash
python responses_api_agents/simple_agent/client.py
```
