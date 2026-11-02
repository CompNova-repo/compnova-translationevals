# `API/cron` — the automation pipeline

The "audio lands in a folder, a cron job evaluates it" pipeline.

| File | What it is |
|---|---|
| **`run_batch.py`** | The worker. Standard library + `requests` only. |
| **`install_cron.sh`** | Installs the worker as a real cron job on this machine. |
| **`mock_api.py`** | A stand-in for the service, so you can exercise the worker without a GPU. |

Full walkthrough, request/response contract, scheduling and the 524 caveat:
[`../API_GUIDE.md`](../API_GUIDE.md) §10.
What is still missing in the service: [`../ROADMAP.md`](../ROADMAP.md).

---

## Install

```bash
pip install requests
```

## Directory layout it expects

```
/workspace/eval-pipeline/
├── inbox/        # the watched folder — the producer drops files here
├── processing/   # claimed by the runner while in flight
├── done/         # evaluated successfully
├── failed/       # gave up — see <key>.error.txt for why
└── results/      # <eval_id>.json — the durable local ledger
```

`processing/`, `done/`, `failed/` and `results/` are created automatically next to the
inbox if they don't exist.

## Naming convention

Two modes, selected with `--pair-mode`:

| Mode | Example | Rule |
|---|---|---|
| `prefix` (default) | `EN_001_#1.wav` + `ES_001_#1.wav` | leading `<LANG>_` is stripped; the remainder is the pair key |
| `dotted` | `clip_001.EN.wav` + `clip_001.ES.wav` | the last dot-separated component is the language |

The language codes come from `--src-lang` / `--tgt-lang` (default `EN` / `ES`).

This matches the convention already used in the repo — `download_dral_subset.py`
writes `EN_<id>` / `ES_<id>`, and `run_ser_eval.py` pairs on the same stripped-prefix
key.

**Source audio only?** Use `--items sources --target-lang spa`. Every file in the inbox
is sent to the API, which translates it and then evaluates it.

## Usage

```bash
# see what would be sent, send nothing
python3 run_batch.py --inbox /workspace/eval-pipeline/inbox --dry-run

# process the current inbox and exit  (this is what cron runs)
python3 run_batch.py --inbox /workspace/eval-pipeline/inbox --once

# long-running watcher
python3 run_batch.py --inbox /workspace/eval-pipeline/inbox --poll 60

# only run when the API is ready (recommended for a schedule)
python3 run_batch.py --inbox /workspace/eval-pipeline/inbox --require-healthy

# remote API, with a key
python3 run_batch.py \
    --inbox /workspace/eval-pipeline/inbox \
    --api-url https://<pod-id>-8000.proxy.runpod.net \
    --api-key "$EVAL_API_KEY" \
    --read-timeout 600
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--mode job\|sync` | `job` | `job` = submit + poll (safe behind a proxy). `sync` = block on one request. |
| `--items pairs\|sources` | `pairs` | Whether the folder holds pairs or source audio only. |
| `--target-lang` | `spa` | Target language when `--items sources`. |
| `--min-similarity` / `--max-merge` | `0.5` / `2` | Passed through to the API. |
| `--max-attempts` | `3` | Retries with exponential backoff. |
| `--job-timeout` | `3600` | How long to wait for a submitted job. |
| `--poll-interval` | `5` | How often to poll a running job. |
| `--processing` / `--done` / `--failed` / `--results` | derived | Override the default directories. |
| `--log FILE` | — | Also write logs to a file. |

## Installing it as a cron job

```bash
# on the API host (the supported path)
bash ../runpod_start.sh --install-cron

# or directly
bash install_cron.sh \
    --inbox /workspace/eval-pipeline/inbox \
    --api-url http://localhost:8000 \
    --interval-min 10 \
    --log /var/log/compnova-eval-batch.log
```

This writes `/etc/cron.d/compnova-eval`, installs the cron daemon if the image does
not have it, and starts it. Safe to re-run — the file is rewritten and cron reloaded.

The generated entry looks like:

```cron
*/10 * * * * root /usr/bin/flock -n /run/compnova-eval.lock \
  /usr/bin/python3 .../API/cron/run_batch.py \
    --inbox /workspace/eval-pipeline/inbox \
    --api-url http://localhost:8000 \
    --mode job --items pairs --target-lang spa \
    --require-healthy --log /var/log/compnova-eval-batch.log
```

`flock -n` guarantees no two runs overlap, on top of the runner's own atomic
claiming. `--require-healthy` makes the job exit cleanly when the API is not ready,
so a restarting pod does not produce a wall of failures.

### systemd alternative

Better if you want `journalctl` and restart-on-failure:

```ini
# /etc/systemd/system/compnova-eval.service
[Unit]
Description=CompNova evaluation batch runner
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/path/to/repo/API/cron
ExecStart=/usr/bin/python3 run_batch.py --inbox /workspace/eval-pipeline/inbox --api-url http://localhost:8000 --require-healthy
User=root
```

```ini
# /etc/systemd/system/compnova-eval.timer
[Unit]
Description=Run the CompNova evaluation batch every 10 minutes

[Timer]
OnCalendar=*:0/10
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl daemon-reload && systemctl enable --now compnova-eval.timer
```

## What it guarantees

| Guarantee | How |
|---|---|
| **Two runs never grab the same file** | Files are claimed with `os.replace()` into `processing/` — atomic on POSIX. |
| **A completed-but-timed-out evaluation is not repeated** | `X-Request-Id` (derived from the file names, sizes and mtimes) makes the request idempotent, and the runner reconciles against the API's stored results before retrying. |
| **Transient failures don't lose work** | Exponential backoff (5 s → 15 s → 45 s) over `--max-attempts`; permanent failures (HTTP 4xx) are not retried. |
| **An auth failure doesn't lose work** | `401`/`403` puts the files back in the inbox rather than in `failed/` — it is a configuration problem, not a bad file. |
| **Every outcome is recorded** | `results/<eval_id>.json` on success; `failed/<key>.error.txt` with the reason on failure. |
| **A crashed run doesn't re-do completed work** | A `results/<key>.done` marker short-circuits anything already evaluated. |
| **The API is not hammered** | Serial by default — the service processes one evaluation at a time. |

## Testing without a live service

`mock_api.py` implements the endpoints the runner uses. The first `/jobs` submission
is deliberately slow, so the client's polling path can be exercised.

```bash
python3 mock_api.py 8765 6 &          # port, seconds to sleep on the first request

mkdir -p /tmp/p/inbox
echo x > /tmp/p/inbox/EN_001.wav
echo x > /tmp/p/inbox/ES_001.wav
echo x > /tmp/p/inbox/EN_002.wav        # unpaired — the API will translate it

python3 run_batch.py --inbox /tmp/p/inbox --api-url http://127.0.0.1:8765 \
    --poll-interval 1
```

Expected: `EN_001` and `EN_002` are both evaluated, `ES_002` is used as the target of
`EN_002`… actually `EN_002` is unpaired so it is submitted for translation, and
`results/` holds one JSON per item.

Set `MOCK_API_KEYS=secret` before starting the mock to exercise the auth path:

```bash
MOCK_API_KEYS=secret python3 mock_api.py 8765 1 &
python3 run_batch.py --inbox /tmp/p/inbox --api-url http://127.0.0.1:8765 --api-key secret
```

## Running against a remote API

Read `API_GUIDE.md` §10.5 first. The short version:

* The RunPod proxy times out at ~100 s and returns **524** while the evaluation
  usually succeeds server-side.
* The runner defaults to `--mode job`, which submits and polls — nothing blocks on
  the proxy, so a timeout can never cause a duplicate evaluation.
* Prefer running the cron job **on the API host** and pushing results elsewhere.
