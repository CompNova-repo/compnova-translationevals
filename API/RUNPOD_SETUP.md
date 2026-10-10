# Running the S2ST evaluation service on RunPod

How to bring the evaluation API and dashboard up on a RunPod GPU pod. One script does everything, on a brand-new pod or an existing one, for anyone on the team.

## Quick start

1. In RunPod, make sure the pod exposes HTTP ports **8000** and **8501** (set this when creating the pod; changing it later restarts the pod).
2. Open a terminal on the pod (Connect, then Web Terminal).
3. If the repo isn't in `/workspace` yet, clone it (use a GitHub token as the password):
   ```bash
   cd /workspace
   unset GIT_CONFIG_PARAMETERS
   git clone -b SahilPatil266-API https://github.com/CompNova-repo/compnova-translationevals.git
   ```
4. Run:
   ```bash
   bash /workspace/compnova-translationevals/API/runpod_start.sh
   ```

On a new pod this takes about 10 minutes the first time: it installs the Python packages, downloads about 10 GB of models, then starts both services. On a pod that's already set up it takes seconds. It asks for a Hugging Face token once per pod if it can't find a saved one (input is hidden). The token's account must have accepted the terms of `Unbabel/wmt22-cometkiwi-da` on huggingface.co.

When it finishes it prints the two URLs:

| What | URL |
|---|---|
| Dashboard | `https://<pod-id>-8501.proxy.runpod.net` |
| API docs (run evaluations here) | `https://<pod-id>-8000.proxy.runpod.net/docs` |

Always run it with `bash`. The `/workspace` volume doesn't allow making files executable.

## Other commands

```bash
bash runpod_start.sh --check     # is everything running? changes nothing
bash runpod_start.sh --restart   # restart API and dashboard, e.g. after pulling new code
```

To update the code on the pod (plain `git pull` fails there, see below):

```bash
cd /workspace/compnova-translationevals/API
unset GIT_CONFIG_PARAMETERS
git pull origin SahilPatil266-API
bash runpod_start.sh --restart
```

## What the script does

1. **System checks**: installs `git`/`curl` if missing, warns if Python isn't 3.10 to 3.12 or no GPU is visible, and stops early if there isn't about 15 GB of disk for models.
2. **Shell defaults**: writes the environment settings into `~/.bashrc` so new terminals have them.
3. **Python packages**: removes packages that break imports (jax, TensorFlow, wandb, peft), installs the evaluation stack if it's missing, and re-applies the version pins COMET needs (numpy 1.26.4, transformers 4.39.3, tokenizers 0.15.2, protobuf 4.25.9, accelerate 0.27.2, huggingface_hub 0.36.2, torchmetrics 0.10.3). `unbabel-comet` 2.2.7 and `sentence-transformers` 5.7.0 are installed without dependencies because their declared requirements conflict with the pins even though they work with them.
4. **Library fixes**: patches faster-whisper for PyAV 14+ (otherwise every evaluation fails with "unexpected keyword argument 'metadata_errors'"), and puts torch's CUDA 12 libraries where Whisper can find them.
5. **MetricX**: clones Google's MetricX at the pinned commit `fc4978e` into `/opt/metricx` and links `API/metricx` to it. It can't live on `/workspace` because git needs to change file permissions there.
6. **Models**: downloads all seven models into `/root/hf-cache` on the pod's own disk and checks no file is empty, re-downloading any that are.
7. **API** on port 8000, started detached so closing the terminal doesn't stop it. If the process dies during startup, the script shows the log immediately instead of waiting.
8. **Dashboard** on port 8501, with the settings it needs to work through RunPod's proxy.

Only one copy of the script can run at a time.

## Why models are not kept on /workspace

The shared `/workspace` volume served model files as **empty** to a newly created pod, then showed their real contents minutes later. The API crashed on startup reading an "empty" Whisper model. Local disk avoids that and also loads faster. The cost is a re-download (a few minutes) whenever a pod is recreated. To use a different location, set `S2ST_MODEL_CACHE=/some/path` before running the script.

## What survives what

| Event | What's lost | Fix |
|---|---|---|
| Closing the browser or terminal | nothing | |
| Pod restart, or a new pod on the same volume | packages, models, MetricX, running services | run the script |
| New pod on a new volume | also the repo and stored results | clone, then run the script |

## Known quirks of these pods

- **git**: the pod sets `GIT_CONFIG_PARAMETERS` in a format git rejects, so run `unset GIT_CONFIG_PARAMETERS` first in any new terminal (the script's `~/.bashrc` block does this for you). On `/workspace`, git also can't save branch tracking settings, so pull with `git pull origin SahilPatil266-API`.
- **PYTHONPATH**: the pod setting contains a literal `${PYTHONPATH:+...}` that RunPod doesn't expand. The script overrides it.
- **pip cache warning** about `/workspace/.cache/pip` is harmless.

## Known limits

- **No authentication.** Anyone with the URLs can run evaluations and read stored results. Share links only with people who need them.
- **One evaluation at a time.** MetricX uses fixed file names, so simultaneous requests can overwrite each other.
- **Proxy timeout about 100 seconds.** A `524` error may still mean the evaluation finished; check the dashboard or the API log.

## Troubleshooting

Logs are in `/root/logs/api.log` and `/root/logs/ui.log` (the pod's own disk; the shared volume shows new log lines late).

If an evaluation fails with a 500 error mentioning `metricx24.predict`, run MetricX by hand to see its real error:

```bash
cd /workspace/compnova-translationevals/API/metricx
HF_HOME=/root/hf-cache python -m metricx24.predict --tokenizer google/mt5-xl \
  --model_name_or_path google/metricx-24-hybrid-large-v2p6-bfloat16 \
  --max_input_length 1536 --batch_size 1 \
  --input_file results/metricx_api_input.jsonl \
  --output_file results/metricx_api_output.jsonl --qe 2>&1 | tail -25
```

If a model fails to download with "access denied", the token's Hugging Face account hasn't accepted that model's terms.

## To do

Add an API key. Give MetricX's files a unique name per request. Add `datasets` to the GPU Dockerfile. Trim the dashboard's `requirements.txt` to `streamlit`, `altair`, `requests`, `pandas`.
