# Alignment v2: verification and debugging

Implemented on `fix/comet-alignment-trace`, based on origin commit `66aab5d`.

## What changed

Whisper now requests word timestamps, preserves word probabilities and original segments,
and accepts explicit source/target languages. Clause reconstruction crosses ASR segment
boundaries and splits inside segments at punctuation, pauses, or a configurable word bound.
Every word has an original segment ID and a stable word ID. No word is silently dropped.

The monotonic aligner minimizes a cost weighted by consumed word mass, normalized per
language. Match cost is `mass * (1 - similarity + merge_cost + timing_cost)`;
skips cost `mass * gap_cost`. Consuming the same text in two pairs no longer gives twice
the similarity reward. Normalized audio position supplies a soft timing constraint.
The default merge limit is now 3 reconstructed units, with a maximum of 6.

COMET is averaged using source ASR word durations, excluding intervening silence.
Source and target coverage use the same speech-duration basis. These are coverage of
recognized speech, not guaranteed coverage of the original audio: ASR omissions remain
possible. Weak alignments or coverage below the configured threshold produce
`status=review_required` and a null headline `comet_kiwi`. The conditional
`matched_comet_kiwi` remains available for investigation. Similarity is not calibrated confidence.

API results and SQLite records retain parameters, raw ASR, reconstructed units, chosen
matches/gaps, alignment costs, audio SHA-256 hashes, model names, package versions,
and stage timing. `debug=true` adds the candidate similarity/cost table for exact
offline alignment replay. Stage logs include a run ID and traceback on failure.
MetricX uses a private temporary directory for each batch and a 600-second timeout.
Model output length and finiteness are validated. Inputs exceeding model token budgets
are rejected instead of silently truncated. Importing the evaluation module does not
clone repositories or authenticate to external services.

## Local regression suite (no model downloads)

```bash
python -m pip install -r API/requirements-test.txt
python -m pytest tests/test_alignment_pipeline.py tests/test_evaluation_api.py -q
```

Tests cover ASR boundary reconstruction, splitting, 0.60+0.60 versus 0.95,
0.55+0.55 versus 0.99, one-to-many translation, omissions, unrelated and empty
inputs, weighted aggregation, review gates, invalid parameters, timestamp requirements,
ASR language forwarding, model truncation, malformed model outputs, and trace replay.
Model calls are mocked; these tests do not establish real COMET quality improvement.

## Run the real gold pair on the configured RunPod environment

From the repository root, with the same Python environment used by `runpod_start.sh`:

```bash
python -m API.debug_evaluation --source EN_147l.wav --target ES_147l.wav \
  --source-language en --target-language es --output reports/gold-v2.json
python -m API.debug_evaluation --replay reports/gold-v2.json
```

For the local two-minute files use their paths under `codebase_eval_pulkit/` instead.
They are a different audio pair. Compare only runs with identical audio hashes.
API equivalent: POST `/evaluate?source_language=en&target_language=es&debug=true`.
The response is persisted by the existing SQLite store without a schema migration.
Download `result` from GET `/evaluations/{id}` to replay a stored run.

## How to investigate a suspicious score

1. Check `summary.status`, both coverage values, and `review_reasons` before the score.
2. Use `trace.asr` and the audio to check recognition, timestamps and detected languages.
3. Inspect `trace.units`: boundary reasons identify punctuation, pauses and length caps.
4. Inspect `trace.alignment.path` for gaps, weak similarities and questionable merges.
   Candidate rows explain competing path costs; replay verifies the stored decision.
5. Read each chunk's exact COMET source/target text and score. Compare with manually
   aligned, reviewed text using `cometeval_batch` to isolate alignment from ASR errors.
6. Compare old and new paths on fixed gold and deliberately corrupted pairs. Track
   coverage, manual alignment correctness, review rate and score distribution. Do not
   tune the system merely to force gold scores to 1.0.

## Limits and release gate

Clause reconstruction is a heuristic. Missing punctuation and ambiguous pauses can
still cause bad boundaries; word-limit boundaries may cut a clause. This implementation
does not promise zero boundary errors or prove that all of the reported 0.65 was due
to alignment. Word timestamps are estimates, not phoneme annotations. Alignment is
monotonic and cannot correctly model arbitrary reordered translations.

Defaults are initial operating settings, not thresholds calibrated on human judgments.
Before production rollout, review actual gold traces, include translation omissions and
unrelated/reordered audio as negative controls, and approve coverage/review thresholds.
The model packages are absent from the local development environment, so real model
inference must be validated on the configured GPU environment. A reference-based
candidate-versus-gold mode is a separate evaluation task, not implemented here.

Restart the new branch's API after updating the pod; normal startup retains an existing
healthy process. The dashboard already reads the API and now exposes review warnings
and the diagnostic trace. No automatic deployment or model-score recalibration is performed.

## Verification recorded for this change

24 regression and HTTP integration tests passed locally. This includes a real SQLite round-trip and a subprocess replay of a saved alignment trace; neural inference is mocked. Python compilation passed. FastAPI emits its existing startup-event deprecation warning. Full gold audio inference and visual dashboard verification remain required in the configured deployment environment.
