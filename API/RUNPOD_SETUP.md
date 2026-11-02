# Running the S2ST evaluation service on RunPod

How to bring the evaluation API, the dashboard and the automation pipeline up on a
RunPod GPU pod. One script does everything, on a brand-new pod or an existing one, for
anyone on the team.

---

## Quick start

1. In RunPod, make sure the pod exposes HTTP ports **8000** and **8501** (set this when creating the pod; changing it later restarts the pod).
2. Open a terminal on the pod (Connect, then Web Terminal).
3. If the repo isn't in `/workspace` yet, clone it (use a GitHub token as the password):
   ```bash
   cd /workspace
   unset GIT_CONFIG_PARAMETERS
   git clone https://github.com/CompNova-repo/compnova-translationevals.git
   ```
4. Run:
   ```bash
   bash /workspace/compnova-translationevals/API/runpod_start.sh
   ```
5. Install the automation pipeline — a cron job that picks up audio from a folder and
   evaluates it:
   ```bash
   bash /workspace/compnova-translationevals/API/runpod_start.sh --install-cron
   ```
6. Check that the whole thing works:
   ```bash
   bash /workspace/compnova-translationevals/API/verify_setup.sh
   ```

On a new pod this takes about 10 minutes the first time: it installs the Python
packages, downloads about 10 GB of models, then starts both services. On a pod that's
already set up it takes seconds. It asks for a Hugging Face token once per pod if it
can't find a saved one (input is hidden). The token's account must have accepted the
terms of `Unbabel/wmt22-cometkiwi-da` on huggingface.co.

When it finishes it prints the two URLs:

| What | URL |
|---|---|
| Dashboard | `https://<pod-id>-8501.proxy.runpod.net` |
| API docs (run evaluations here) | `https://<pod-id>-8000.proxy.runpod.net/docs` |

Always run it with `bash`. The `/workspace` volume doesn't allow making files executable.

---

## The automation pipeline

`runpod_start.sh --install-cron` installs a real cron job on the pod:

```
/workspace/eval-pipeline/
├── inbox/        # drop audio here
├── processing/   # claimed by the runner while in flight
├── done/         # evaluated successfully
├── failed/       # gave up — see failed/<key>.error.txt
└── results/      # <eval_id>.json — the local ledger
```

```bash
# put a pair in the inbox
cp EN_001.wav ES_001.wav /workspace/eval-pipeline/inbox/

# watch it get processed
tail -f /var/log/compnova-eval-batch.log
```

The runner pairs files by their naming convention (`EN_*` with `ES_*`), submits them
to the API's **job** endpoint, polls for the result, writes a local ledger and moves
the files to `done/` or `failed/`.

* **Pairs** (`EN_001.wav` + `ES_001.wav`) → evaluated as-is.
* **Source only** (`EN_002.wav` with no `ES_002.wav`) → the API translates it into the
  target language first, then evaluates. Set the target with `--target-lang spa`.

To use a different folder or interval:

```bash
bash runpod_start.sh --install-cron   # defaults
# or drive it directly:
bash cron/install_cron.sh --inbox /data/inbox --api-url http://localhost:8000 --interval-min 5
```

The job is gated on `GET /health/ready`, so a pod that is still loading its models
simply skips that round instead of producing a wall of failures.

See [`cron/README.md`](cron/README.md) for the details and
[`API_GUIDE.md`](API_GUIDE.md) §10 for the full walkthrough.

---

## Other commands

```bash
bash runpod_start.sh --check          # is everything running? changes nothing
bash runpod_start.sh --restart        # restart API and dashboard, e.g. after pulling new code
bash runpod_start.sh --install-cron   # install the folder pipeline
```

`--check` reports whether the API is up, whether the models are loaded, where the
database is, and whether the cron job is installed.

To update the code on the pod (plain `git pull` fails there, see below):

```bash
cd /workspace/compnova-translationevals
unset GIT_CONFIG_PARAMETERS
git pull origin main
bash API/runpod_start.sh --restart
```

---

## Configuration

Everything is read from the environment. Export these before running the start script,
or add them to `~/.bashrc` (the script writes its own block there).

| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token. Needed to download the gated COMET-Kiwi model. |
| `EVAL_API_KEYS` | *(empty)* | Comma-separated accepted `X-API-Key` values. **Empty = the service is open — anyone with the URL can spend GPU time and read every transcript.** Set this before sharing the pod URL. |
| `EVAL_DB_PATH` | `/var/lib/compnova/evaluations.db` | Where evaluations are stored. Point it at `/workspace/...` if the results must survive the pod being recreated. |
| `EVAL_SPOOL_DIR` | `/var/lib/compnova/spool` | Where uploads are staged. |
| `EVAL_JOB_WORKERS` | `1` | How many evaluations may run at once. Try `2` on a 48 GB pod. |
| `EVAL_JOB_BATCH_SIZE` | `8` | How many queued jobs share one MetricX model load. |
| `EVAL_MAX_UPLOAD_MB` | `200` | Largest accepted upload. |
| `EVAL_MAX_AUDIO_SECONDS` | `1800` | Longest accepted clip. |
| `EVAL_TRANSLATE_ENABLED` | `1` | Set to `0` to disable the translation models. |
| `S2ST_PIPELINE_DIR` | `/workspace/eval-pipeline` | The pipeline folder the cron job watches. |
| `S2ST_CRON_INTERVAL_MIN` | `10` | How often the cron job runs. |
| `S2ST_MODEL_CACHE` | `/root/hf-cache` | Where models are downloaded. |
| `S2ST_LOG_DIR` | `/root/logs` | Where logs go. |

Example:

```bash
export EVAL_API_KEYS="$(openssl rand -hex 24)"
export EVAL_DB_PATH=/workspace/compnova/evaluations.db
export S2ST_PIPELINE_DIR=/workspace/eval-pipeline
bash API/runpod_start.sh
```

---

## What the script does

1. **System checks**: installs `git`/`curl` if missing, warns if Python isn't 3.10 to 3.12 or no GPU is visible, and stops early if there isn't about 15 GB of disk for models.
2. **Shell defaults**: writes the environment settings into `~/.bashrc` so new terminals have them.
3. **Python packages**: removes packages that break imports (jax, TensorFlow, wandb, peft), installs the evaluation stack if it's missing, and re-applies the version pins COMET needs (numpy 1.26.4, transformers 4.39.3, tokenizers 0.15.2, protobuf 4.25.9, accelerate 0.27.2, huggingface_hub 0.36.2, torchmetrics 0.10.3). `unbabel-comet` 2.2.7 and `sentence-transformers` 5.7.0 are installed without dependencies because their declared requirements conflict with the pins even though they work with them. The pins live in `constraints.txt` — one source of truth, shared with the Dockerfile.
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

**Evaluations** live in SQLite at `EVAL_DB_PATH`. The default is on the pod's own disk
(`/var/lib/compnova`), so a recreated pod starts with an empty store. Point
`EVAL_DB_PATH` at `/workspace/...` to keep them across pod recreations, and export
them regularly with `GET /evaluations/export`.

---

## Known quirks of these pods

- **git**: the pod sets `GIT_CONFIG_PARAMETERS` in a format git rejects, so run `unset GIT_CONFIG_PARAMETERS` first in any new terminal (the script's `~/.bashrc` block does this for you). On `/workspace`, git also can't save branch tracking settings, so pull with `git pull origin main`.
- **PYTHONPATH**: the pod setting contains a literal `${PYTHONPATH:+...}` that RunPod doesn't expand. The script overrides it.
- **pip cache warning** about `/workspace/.cache/pip` is harmless.

## Known limits

- **One evaluation at a time by default.** `EVAL_JOB_WORKERS=2` will run two at once if the GPU has the headroom. The pipeline is GPU-bound and the models are large.
- **Proxy timeout about 100 seconds.** A `524` error may still mean the evaluation finished. Use the job API (`POST /jobs`) from any remote client — it returns immediately and is never affected by the proxy timeout.
- **No authentication unless you set `EVAL_API_KEYS`.** Share the URL only with people who need it — and set the variable.
- **MetricX loads its model once per batch.** The job worker batches up to `EVAL_JOB_BATCH_SIZE` queued jobs so the model load is shared, but a batch of one still pays it.

---

## Troubleshooting

Logs are in `/root/logs/api.log` and `/root/logs/ui.log` (the pod's own disk; the shared volume shows new log lines late). The pipeline log is `/var/log/compnova-eval-batch.log`.

**Is the API up? Is the model loaded?**

```bash
curl -s localhost:8000/health
curl -s localhost:8000/health/ready | python3 -m json.tool
```

`/health` answers as soon as the process is up. `/health/ready` returns `503` until
every model is loaded, and its `missing` array names the one that isn't. The models
load in the background, so a `503` for the first ten minutes on a new pod is normal.

**Run the whole checklist:**

```bash
bash API/verify_setup.sh
```

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

**The cron job isn't running.**

```bash
cat /etc/cron.d/compnova-eval
tail -50 /var/log/compnova-eval-batch.log
service cron status
```

If cron isn't installed on the image the installer adds it; if the pod has no
`systemd`, it starts `cron` directly. As a manual fallback, run the worker yourself:

```bash
python3 API/cron/run_batch.py --inbox /workspace/eval-pipeline/inbox --once
```
