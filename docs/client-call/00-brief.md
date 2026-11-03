# Client call brief — two pieces of feedback we had not thought of

One page. Everything here is backed by a document in this folder.

---

## Feedback 1 — "the file has background noise and other languages being spoken in the background; how do we get only the right speaker?"

### Our answer, in order

1. **We do not separate the voices.** Target-speaker extraction is research-grade:
   no mainstream pretrained model, no dependable cross-language accuracy, and it
   changes the audio being scored. Denoising (hiss, fan, reverb) is mature and we
   can do that; *unmixing two people talking in two languages* is not something we
   will put a client-facing number on.
2. **We measure and flag.** Every file gets: SNR, speech ratio, speaker count,
   overlap ratio, and the languages present — including the language of the
   *background* speech. A contaminated file is reported as contaminated instead of
   quietly getting a wrong score.
3. **We score the intended speaker when asked.** With diarization we keep the
   dominant speaker's **words** — word-level, so a background sentence inside the
   target speaker's own turn is dropped too — and compute the score on that audio only.

### What to show them

```bash
curl -s -X POST "$BASE/audio/quality" -F "audio=@EN_001.wav"
```

```json
{"source": {"snr_db": 18.2, "speaker_count": 2, "dominant_speaker_ratio": 0.83,
            "overlap_ratio": 0.06, "languages": ["eng"],
            "background_languages": ["spa"], "background_speech_detected": true,
            "flags": ["multi_speaker", "background_speech"],
            "usable": true, "contaminated": true},
 "flags": ["multi_speaker", "background_speech"], "usable": true}
```

### Ammunition

* The closest comparable vendor says the same thing: Translated's Lara 3 —
  *"Audio translation … currently built for single-speaker content; multi-speaker
  support with voice cloning is planned for later in 2026."*
* Every API vendor (Google, Azure, AWS, Deepgram, AssemblyAI) **labels** speakers
  and documents that overlap and noise hurt accuracy. None of them remove a voice.
* NIST SRE24: top-ranked systems *"avoided incorporating speaker diarization
  mainly because it tended to degrade recognition performance."*
* Cross-language is genuinely the weak case: the multi-language benchmark reports
  DER 0.081 (NeMo) vs 0.181 (pyannote), and 0.142 with Japanese in the mix vs 0.050
  without. We will not pretend otherwise.

### The line to use

> "We detect it, we flag it, and we score the speaker you meant. We won't claim we
> removed the background voice — if you want separation as an *extra* signal next to
> the clean score, we can run that too."

---

## Feedback 2 — "ElevenLabs is very good and expensive; at an ERP level the cost doesn't make sense"

### Our answer

She is right about the *shape* of the cost, and the cost is not where she thinks.

| | |
|---|---|
| ElevenLabs TTS v4 | **$0.08 / 1k chars** ≈ $0.072 per minute of audio ($4.80/hour) |
| ElevenLabs Flash/Turbo | $0.04 / 1k chars ≈ $0.036 per minute |
| ElevenLabs **Voice Isolator** (noise removal) | **$0.12 / minute** = $7.20 per hour of audio |
| ElevenLabs plans | Free → Starter $6 → Creator $22 → Pro $99 → Scale $299 → **Business $990**; credits do not roll over |
| Our GPU | RTX A6000 48 GB **$0.59/hour** on RunPod (A100 80 GB $1.79) |

**Per 5-minute evaluation job (63 s of GPU time):**

| Setup | TTS | scoring | total |
|---|---|---|---|
| ElevenLabs v4 | $0.360 | $0.010 | **$0.370** ← TTS is 97% of the bill |
| Self-hosted TTS on the pod we already rent | $0.001 | $0.010 | **$0.011** |

**10,000 jobs/month:** $3,703 with ElevenLabs v4 vs **$113** self-hosted.

### The three points that land

1. **TTS is not on our evaluation path.** We score the audio the *client* sends.
   TTS only appears in our cascaded translate path and in demo material.
2. **Scoring our own synthesised speech would be circular** — COMET would be
   grading our TTS, not their translation. We should not do it anyway.
3. **Per-minute metered APIs are where her argument really bites.** If we ever clean
   every submitted file, $0.12/min is $7.20 per audio hour; the same job on the GPU
   we already rent is ~150× cheaper.

### Where ElevenLabs *is* worth it

Demos and onboarding, where the perceived quality is what the client is reacting to
and the volume is tens of minutes, not thousands. That is $5–20/month. Use it where
it buys perception; self-host where it buys unit economics.

### Questions to ask her

1. Audio minutes and jobs per month?
2. Where does TTS sit in *their* product — or do they already pay ElevenLabs and
   should we just score *their* output?
3. Do they need voice cloning, and do they have the rights to the cloned voice?
4. Is bulk audio cleaning in scope (that is the per-minute cost decision)?
5. What is the quality bar in Spanish — emotional delivery, or intelligibility?

### The line to use

> "You're right that ElevenLabs' metering doesn't fit a per-minute product — at
> 10,000 five-minute jobs a month their API is about $3,700 of our bill and running
> the same model on the GPU we already rent is about $113. But the bigger point:
> the evaluation service doesn't need to generate speech at all. You send us the
> target audio your engine produced and we score it."

---

## ⚠️ One thing to raise internally before the call

The cost audit turned up something bigger than the API price. Verified against the
Hugging Face model API:

| Model | Where we use it | Licence | Commercial? |
|---|---|---|---|
| `Unbabel/wmt22-cometkiwi-da` | **our headline COMET score** | **CC-BY-NC-SA-4.0** | **no** |
| `facebook/nllb-200-distilled-600M` | cascaded translation | **CC-BY-NC-4.0** | **no** |
| `facebook/seamless-m4t-v2-large` | `seamless` translate mode | **CC-BY-NC-4.0** | **no** |
| `google/metricx-24-hybrid-large-v2p6-bfloat16` | MetricX-24 QE | Apache-2.0 | yes |
| `openai/whisper-large-v3-turbo`, SER, pyannote 3.1 | ASR / SER / diarization | MIT / Apache-2.0 / MIT | yes |

Running a non-commercial model to produce a commercial score is, on the plain
reading of CC-BY-NC, commercial use. That is a procurement conversation, not a
call conversation — but it is better to know before a contract is signed than
after. Options: buy the Unbabel/Meta commercial licences (predictable per-unit
cost) or move the headline metric to MetricX-24 QE / `wmt22-comet-da`.

---

## Where everything lives

| Document | What it is |
|---|---|
| `01-noisy-and-multispeaker-audio.md` | the full answer to feedback 1, with vendor evidence and limits |
| `02-speech-costs.md` | the full answer to feedback 2, with the price list and the model options |
| `tts_unit_economics.py` | regenerates every number in 02 |
| `API/audio_quality.py` | the implementation behind feedback 1 |
| `API/API_GUIDE.md` §6.16 | the `POST /audio/quality` endpoint |
| PR #12 | the branch with all of it |
