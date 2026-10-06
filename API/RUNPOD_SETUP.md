# Running the S2ST evaluation service on RunPod

This is how the evaluation API and dashboard run on the team's RunPod pod (one NVIDIA A40, 48 GB). It records what the pod needs and why, so the setup never has to be worked out again.

## Quick start

On the pod, open a terminal (Connect, then Web Terminal) and run:

```bash
bash /workspace/compnova-translationevals/API/runpod_start.sh
```

That's all. The script checks every part of the setup, fixes whatever is missing, starts the API and the dashboard if they aren't already running, and prints both URLs. It is safe to run at any time, including on a pod that is already working, where it just confirms everything is up.

Run it with `bash` rather than `./runpod_start.sh`. The `/workspace` volume does not allow changing file permissions, so the script can't be marked executable.

Once it finishes:

| What | URL |
|---|---|
| Dashboard | `https://<pod-id>-8501.proxy.runpod.net` |
| API docs (run evaluations here) | `https://<pod-id>-8000.proxy.runpod.net/docs` |

The current pod ID is `uzmdo0dc1kreia`. The script prints the URLs with the right ID filled in.

## What survives a pod restart

Only `/workspace` is persistent. It holds the repo, the Hugging Face model cache (`/workspace/models/huggingface`), Pulkit's Python patch (`/workspace/python_patches`), the logs and `evaluations.db`.

Everything else is wiped when the pod restarts: installed Python packages and their versions, the faster-whisper patch, the MetricX clone in `/opt/metricx`, `~/.bashrc`, and any running processes. The start script rebuilds all of these, which is why it exists.

**Do not change the pod's exposed ports or other settings unless necessary.** Editing the pod restarts it. Nothing is lost permanently, but the services stay down until someone runs the script again.

## Why each step is needed

**Environment variables.** `HF_HOME` points at the shared model cache so nothing is downloaded again. `PYTHONPATH` loads Pulkit's `sitecustomize.py`, which stops Python from failing on `/workspace`'s permission restrictions. The pod's own `PYTHONPATH` setting is broken: it contains the literal text `${PYTHONPATH:+:$PYTHONPATH}`, which RunPod doesn't expand, so the script overrides it. Fixing that value in the pod settings to plain `/workspace/python_patches` would be cleaner, but it restarts the pod. `USE_TF=0` and `USE_TORCH=1` stop `transformers` from importing TensorFlow if it ever reappears. The script also adds these lines to `~/.bashrc`, so new terminals pick them up.

**`GIT_CONFIG_PARAMETERS`.** The pod sets this variable in a format git can't parse, which makes every git command fail with "bogus format in GIT_CONFIG_PARAMETERS". Unsetting it fixes git for that terminal.

**Package pins.** COMET-Kiwi (`unbabel-comet==2.2.7`) needs older libraries than most installs pull in by default. The pinned versions are numpy 1.26.4, transformers 4.39.3, tokenizers 0.15.2, protobuf 4.25.9, accelerate 0.27.2, huggingface_hub 0.36.2 and torchmetrics 0.10.3. Installing almost anything else tends to upgrade some of these silently, so the script re-applies any that drifted. It also writes `/workspace/constraints.txt`. Install new packages with `pip install <package> -c /workspace/constraints.txt`, which lets pip add dependencies but forbids it from changing the pins.

**Packages removed.** jax, TensorFlow, wandb, peft and their relatives crash on import with these pins (TensorFlow ships protobuf 5 files, jax needs numpy 2). `transformers` imports them automatically if they're installed, which breaks COMET and sentence-transformers. None of them are needed.

**`datasets`.** MetricX's `predict.py` imports it. It isn't in the Dockerfile either, so the Docker build will need it too.

**faster-whisper patch.** PyAV 14 and later removed the `metadata_errors` argument that faster-whisper 1.2.1 still passes when opening audio. Without the patch, every evaluation fails instantly with "open() got an unexpected keyword argument 'metadata_errors'".

**MetricX location.** MetricX is Google research code, not a pip package. `eval_core.py` expects it in an `API/metricx` folder and will try to clone it there itself. That clone fails on `/workspace`, because git needs to change file permissions. So the script clones it into `/opt/metricx` on the container disk, at the pinned commit `fc4978e`, and `API/metricx` is a link to it. After a restart, `/opt` is empty and the script clones it again (a few seconds).

**Starting the services.** Both are started with `setsid nohup ... &`, which detaches them from the terminal, so closing the browser tab doesn't stop them. Streamlit needs `--server.address 0.0.0.0` and the three proxy flags (`enableCORS`, `enableXsrfProtection`, `enableWebsocketCompression` all off), otherwise the page loads through RunPod's proxy but hangs because its websocket is rejected.

## Running an evaluation

Open the API docs URL, expand `POST /evaluate`, click Try it out, choose the source and translated audio files, and click Execute. The response includes an `id`, and the result appears on the dashboard after a refresh. From a script:

```bash
curl -F "source_audio=@EN.wav" -F "target_audio=@ES.wav" https://<pod-id>-8000.proxy.runpod.net/evaluate
```

## Known limits

**No authentication.** Anyone with the URLs can run evaluations on the GPU and read every stored result, transcripts included. Share the links directly with the people who need them. An API key is the next thing to add.

**One evaluation at a time.** MetricX reads and writes fixed file names (`metricx/results/metricx_api_input.jsonl` and `metricx_api_output.jsonl`), so two simultaneous evaluations can overwrite each other's results.

**The proxy times out after about 100 seconds.** If a request returns `524`, the evaluation may still have finished. Check `tail /workspace/api.log` or the dashboard.

**SQLite on a network volume.** `evaluations.db` lives on `/workspace`. If the log ever shows "database is locked", restart the API with `EVAL_DB_PATH=/root/evaluations.db` set. That moves the database to the container disk, which means it no longer survives restarts.

## Troubleshooting

To see what's running:

```bash
pgrep -af "[u]vicorn|[s]treamlit"
curl -s localhost:8000/health; echo
tail -30 /workspace/api.log
tail -20 /workspace/ui.log
```

If the RunPod page says "Waiting for service to respond", the service on that port isn't running. Run the start script.

If an evaluation returns a 500 error mentioning `metricx24.predict`, MetricX failed but its error is hidden by the API. Run it by hand to see the real message:

```bash
cd /workspace/compnova-translationevals/API/metricx
python -m metricx24.predict --tokenizer google/mt5-xl \
  --model_name_or_path google/metricx-24-hybrid-large-v2p6-bfloat16 \
  --max_input_length 1536 --batch_size 1 \
  --input_file results/metricx_api_input.jsonl \
  --output_file results/metricx_api_output.jsonl --qe 2>&1 | tail -25
```

To restart everything from scratch, stop both services and run the script:

```bash
pkill -f "[u]vicorn api:app"; pkill -f "[s]treamlit run"
bash /workspace/compnova-translationevals/API/runpod_start.sh
```

## To do

Add `datasets` to the GPU Dockerfile. Remove `librosa`, `soundfile`, `gTTS` and `plotly` from the demo's `requirements.txt` (the dashboard now needs only `streamlit`, `altair`, `requests` and `pandas`). Add an API key. Give MetricX's input and output files a unique name per request. Fix the pod's `PYTHONPATH` setting at the next planned restart.
