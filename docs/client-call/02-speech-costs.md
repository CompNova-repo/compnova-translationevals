# Client call: "ElevenLabs is very good and expensive — at an ERP level the cost doesn't make sense"

The feedback, as given:

> She mentioned ElevenLabs a lot, saying they're very good and expensive in their use
> cases, but the cost on an ERP level doesn't make sense.

## The short answer

She is right about the *shape* of the cost and slightly wrong about *where* it lands.

- ElevenLabs' API is metered per **character** for TTS and per **minute** for
  audio processing. That is exactly the wrong shape for a product that is itself
  metered per audio minute, and it is why it looks absurd on a spreadsheet.
- But in **our** product, TTS is not on the critical path of the thing we sell.
  We *score* audio; we only *generate* audio for demos, for the cascaded
  translate path, and for pipeline self-tests. The evaluation path itself has
  **no TTS cost at all**.
- The place where her argument really bites is the opposite direction: if we
  ever need to clean audio in bulk (background noise, see
  `01-noisy-and-multispeaker-audio.md`), a per-minute API
  (ElevenLabs Voice Isolator is **$0.12/min = $7.20 per hour of audio**) is
  untenable, while the same job on the GPU we already rent is ~100× cheaper.

## What ElevenLabs actually charges today

Verified on <https://elevenlabs.io/pricing/api> and
<https://elevenlabs.io/pricing>, November 2026.

### Pay-as-you-go API

| Product | Price | Note |
|---|---|---|
| TTS v4 | **$0.08 / 1k characters** | 90+ languages, voice tags |
| TTS v4 Turbo | $0.04 / 1k chars | ~100 ms latency |
| TTS v3 | $0.08 / 1k chars | 70+ languages |
| TTS v3 Conversational | $0.04 / 1k chars | |
| TTS v2 Multilingual | $0.08 / 1k chars | 29 languages |
| TTS Flash / Turbo | $0.04 / 1k chars | 32 languages |
| Scribe v2 (STT) | $0.22 / hour | |
| Scribe v2 Realtime (STT) | $0.39 / hour | word-level timestamps |
| **Voice Isolator** (noise removal) | **$0.12 / minute** | removes ambient noise, reverb, interference |
| Voice Changer | $0.12 / minute | |
| Dubbing v1 | $0.33 / minute | |
| Dubbing v2 | $2.20 / minute | |

ElevenLabs quote the TTS price as "~$0.08/minute" of generated audio, so
**$4.80 per hour of speech** at v4, **$2.40/hour** on Flash/Turbo.

### Subscriptions (the commitment problem)

| Plan | $/month | Credits/month | Effective $/1k chars | Audio included |
|---|---|---|---|---|
| Free | $0 | 10,000 | — | 0.2 h |
| Starter | $6 | 30,000 | $0.20 | 0.6 h |
| Creator | $22 | 121,000 | $0.18 | 2.2 h |
| Pro | $99 | 600,000 | $0.17 | 11 h |
| Scale | $299 | 1,800,000 | $0.17 | 33 h |
| Business | $990 | 6,000,000 | $0.17 | 111 h |
| Enterprise | custom | custom | "significant discounts at scale" | |

The ladder is the ERP trap she is describing: **credits reset monthly and do not
roll over**, the plan you need is set by your *worst* month, and the volume
price ("as low as 5c/minute") only starts at $990/month.

> **The API tiers are prepaid list price, not a discount.** Every tier works out
> to the same $0.08/1k characters — only an Enterprise contract ("significant
> discounts at scale") beats it. If anyone is spending more than ~$5k/month on
> ElevenLabs, get a quote before switching vendors.

### What the competitors charge (same day, same units)

| Provider | Product | Price | $/1M chars |
|---|---|---|---|
| **Azure** | Neural | $15 | $15 |
| **Deepgram** | Aura-1 | $15 | $15 |
| **AWS** | Polly Neural | $16 | $16 |
| **Google** | Neural2 | $16 | $16 |
| **Nari Labs** | hosted Qwen3-TTS | $5 | $5 |
| **Speechify** | Simba 3.5 | $6–20 (launch ends 21 Nov 2026) | $6–20 |
| **Deepgram** | Aura-2 | $30 | $30 |
| **Google** | Chirp 3 HD | $30 | $30 |
| **Cartesia** | Sonic-3.6 | $37–50 | $37–50 |
| **ElevenLabs** | v4 / v3 / v2-mtl | $80 | $80 |
| **ElevenLabs** | v4 Turbo / Flash | $40 | $40 |

So ElevenLabs is **5.3×** Azure Neural, and a hosted open model (Nari, Qwen3-TTS)
is **16×** cheaper.

### And the ASR we would replace, for reference

| Provider | Model | Price |
|---|---|---|
| OpenAI | gpt-4o-mini-transcribe | $0.003 / min |
| Deepgram | Nova-3 | $0.0077 / min (+$0.002 for diarization) |
| Google | STT v2 standard | $0.016 / min |
| Azure | batch / fast / real-time | $0.225 / $0.36 / $1.20 per hour |
| ElevenLabs | Scribe v2 | **$0.22 / hour** |

Scribe v2 at $0.22/hour is ~5× cheaper than Azure real-time and ~2× cheaper than
Google STT — the inversion that surprises people. For our volume, self-hosted
Whisper is still ~20× cheaper than all of them.

### Quality: where the money actually buys something

Independent leaderboard (Artificial Analysis, provider-voice board, retrieved
2026-11-03):

| Model | Elo | Note |
|---|---|---|
| **Eleven v4** | **1334 ±15** | best on the board |
| **Eleven v4 Turbo** | **1331 ±16** | **the same quality as v4, at half the price** |
| Qwen-Audio TTS (hosted open model) | 1301 ±15 | best non-ElevenLabs |
| Cartesia Sonic 3.6 | 1293 ±13 | |
| Google Gemini Flash TTS | 1282 ±13 | |
| Speechify Simba 3.2 | 1244 ±12 | |
| Azure HD 2.5 | 1143 ±11 | |
| Chatterbox HD | 1108 ±12 | |
| Kokoro-82M | 1061 ±11 | what self-hosting actually buys you |

Two things to take from this: **v4 Turbo is v4's quality at half the price**, and
the gap between the best hosted model and the open weights you would self-host is
**~250 Elo** — decisive. So "swap ElevenLabs for a self-hosted model" is *not* a
free 150× saving; it is a real quality drop that you have to decide to accept.

## Where TTS sits in our product — and where it does not

```
client audio  ──▶ [ our service ]  ──▶ quality score
                    │
                    ├── ASR (Whisper)          ── the scored artifact
                    ├── alignment (LaBSE)
                    ├── COMET-Kiwi / MetricX-24
                    └── SER

POST /translate (optional)  ──▶ Whisper ──▶ NLLB-200 ──▶ TTS ──▶ target audio
                                    └── our own target, used to *demo* the
                                        translation or to self-test the pipeline
```

- **Scoring a client's file: no TTS cost.** The target audio is the client's
  (their engine's output, or a human translation). We never pay to produce it.
- **Synthesising our own target:** this is where the meter runs — and it is the
  `cascaded` translate path plus demo material.

There is a second, more important reason not to synthesise the target ourselves:
**scoring your own TTS output with COMET is circular.** The score then measures
our synthesizer's fluency, not the client's translation quality. We should score
audio the client produced, and use TTS only where a synthesized voice is the
product (demos) rather than the measurement.

## The unit economics, computed

Assumptions: 5-minute source, 5-minute generated target, ~900 characters per
minute of speech, one evaluation ≈ 63 s of GPU time on a rented GPU.
Regenerate with `python3 docs/client-call/tts_unit_economics.py`.

### Cost of one minute of generated speech

| Option | per minute | list price |
|---|---|---|
| ElevenLabs v4 | **$0.0720** | $0.08 / 1k chars |
| ElevenLabs Flash/Turbo | $0.0360 | $0.04 / 1k chars |
| ElevenLabs Voice Isolator (denoise) | **$0.1200** | $0.12 / min |
| Azure Neural (hosted API) | $0.0150 | $15 / 1M chars |
| AWS Polly Neural / Google Neural2 | $0.0160 | $16 / 1M chars |
| Nari (hosted Qwen3-TTS) | $0.0050 | $5 / 1M chars |
| Self-hosted TTS on an RTX A6000 48 GB pod ($0.33/hr) at 20× realtime | $0.00028 | — |
| Self-hosted TTS on the same pod at 50× realtime | $0.00011 | — |

> The self-hosted rows are **estimates**: they assume an open TTS model
> (Chatterbox, CosyVoice2, Qwen3-TTS) generating audio 20–50× faster than real
> time on a 48 GB GPU. Measure it on the pod before quoting it. The order of
> magnitude is what matters: ~700× cheaper per minute, but self-hosting
> carries a fixed monthly cost - see the break-even below.

### Cost of one evaluation job (5-minute files)

| Setup | TTS | scoring | total | TTS share |
|---|---|---|---|---|
| ElevenLabs v4 | $0.4000 | $0.0103 | **$0.4103** | 97% |
| ElevenLabs v4 Turbo / Flash | $0.2000 | $0.0103 | $0.2103 | 95% |
| Azure Neural (hosted) | $0.0750 | $0.0103 | $0.0853 | 88% |
| Self-hosted TTS on the pod we already rent | $0.0010 | $0.0103 | **$0.0113** | 9% |

### Monthly bill

| jobs / month | GPU hours | ElevenLabs v4 | ElevenLabs Flash/Turbo | Self-hosted TTS (incl. fixed) |
|---|---|---|---|---|
| 1,000 | 18 | $406 | $206 | $3,033 |
| 10,000 | 175 | $4,058 | $2,058 | $3,268 |
| 100,000 | 1,750 | $40,578 | $20,578 | $4,648 |

### Where self-hosting actually wins

Per minute, self-hosting is ~700× cheaper than ElevenLabs. It is not free,
though: one always-on GPU (~$1,015/month) plus ops labour (~$2,000/month)
is a **$3,015/month fixed cost**. So:

| Comparison | Break-even |
|---|---|
| self-hosted vs ElevenLabs v4 | ~7,500 jobs/month |
| self-hosted vs ElevenLabs v4 Turbo | ~15,000 jobs/month |
| self-hosted vs Azure Neural (the cheap-API hybrid) | ~40,000 jobs/month |

**Below ~7,500 jobs/month, do not self-host.** Switch the bulk synthesis to a
cheap hosted API (Azure Neural / Google Neural2 / AWS Polly Neural at $15–16 per
1M characters) — it costs nothing to operate and is still 5× cheaper than
ElevenLabs. Self-hosting's real advantages are privacy, latency and voice-cloning
control, not unit price.

## The trap that costs more than the price

**ElevenLabs charges credits per *generation request*, not per download** (their
own pricing FAQ). So re-running the pipeline re-bills the whole corpus: at 100k
jobs/month, **one** full re-run costs $40,000 (v4) or $20,000 (Turbo). If the
team re-scores on every metric-version bump, or generates N candidate
syntheses per file, the bill is N × $20–40k/month.

The fix is not a different vendor, it is a **content-addressed cache** keyed on
(target_text, voice_id, model, params, seed). That typically removes 50–90% of
TTS spend and is worth doing first.

**We have already built it.** The cascaded translate path now caches synthesised
speech by (language, text) in `EVAL_TTS_CACHE_DIR`, so a repeated translation is
free, deterministic, and never re-bills the speech API (`API/eval_core.synthesis`,
tests in `API/test_synthesis_cache.py`).

The other half of that story: `gTTS` — what we use today — is free but hits
Google's *unofficial* endpoint, is low quality, gets rate-limited, and is not
something to build a paid product on. Replacing it is on the roadmap, with the
price table above deciding which engine.

RunPod pod prices used above (page dated 28 Oct 2026):
RTX A6000 48 GB **$0.59/hr**, A40 48 GB $0.59, L40S 48 GB $1.09,
A100 80 GB $1.79, H100 80 GB $3.99. Serverless is dearer (A100 80 GB $2.72/hr,
48 GB $1.22/hr) but bills per second.

**The headline:** self-hosted TTS is ~150–350× cheaper per minute than ElevenLabs
v4, so it wins at the *first* job — there is no volume break-even, only a
quality/ops trade-off. And the per-minute metered *audio processing* APIs
(Voice Isolator at $0.12/min) are the ones that would really hurt at scale:
10,000 hours of audio through that API is ~$72,000/month.

## The bigger lever than price: model licences

While costing this out, one thing turned up that matters more than the API price.
Several models in the current pipeline are **non-commercial licensed**. Verified
against the Hugging Face model API, November 2026:

| Model | Where we use it | Licence | Commercial? |
|---|---|---|---|
| `Unbabel/wmt22-cometkiwi-da` | **the headline COMET score** (`get_comet()`) | **CC-BY-NC-SA-4.0** | **no** — needs a commercial licence from Unbabel |
| `Unbabel/wmt22-comet-da` | the reference-based alternative | Apache-2.0 | yes (needs a reference translation) |
| `facebook/nllb-200-distilled-600M` | cascaded translation | **CC-BY-NC-4.0** | **no** — Meta sells a commercial licence |
| `facebook/seamless-m4t-v2-large` | the `seamless` translate mode | **CC-BY-NC-4.0** | **no** — same |
| `google/metricx-24-hybrid-large-v2p6-bfloat16` | MetricX-24 QE | Apache-2.0 | yes |
| `ehcalabres/wav2vec2-lg-xlsr-en-…-recognition` | SER | Apache-2.0 | yes |
| `openai/whisper-large-v3-turbo` | ASR | MIT | yes |
| `pyannote/speaker-diarization-3.1` | the new audio-quality layer | MIT | yes |

If we are selling this as a service, using a non-commercial model to produce a
commercial score is, on the plain reading of CC-BY-NC, commercial use. This is a
bigger deal than the difference between ElevenLabs v4 and Flash, and it is the kind
of thing that surfaces in a client's procurement review rather than on a call.

Options, in order of preference:

1. **Buy the commercial licences** (Unbabel and Meta both sell them) and price them
   into the cost model — that is a predictable per-unit cost, unlike a metered API.
2. **Keep MetricX-24 QE (Apache-2.0) as the headline** and treat COMET-Kiwi as
   optional/ licensed, or switch to `wmt22-comet-da` where a reference exists.
3. Keep the translation models for demos only and put a commercially licensed
   backend behind the paid translate endpoint.

> Worth checking with whoever owns the company's legal/commercial side before the
> next call, not after a contract is signed.

## The recommendation

| Use | Do this | Why |
|---|---|---|
| Scoring client audio | no TTS at all | the target audio is the client's; also avoids circularity |
| Demos, onboarding, "listen to our Spanish" | **ElevenLabs v4** — but cap the volume | the quality is what the client is reacting to; a demo month is a few dollars, and this is where perception is bought |
| The cascaded translate path at volume | a cheap hosted API (Azure Neural $15/1M) until ~7,500 jobs/month, then **self-hosted** open TTS | 5× cheaper than ElevenLabs with zero ops; self-hosting is ~700× cheaper per minute but carries $3,015/month of fixed cost |
| Bulk noise/voice cleanup | **self-hosted** (Demucs v4 / Resemble Enhance) | $0.12/min APIs do not survive an ERP review |
| If the client already pays ElevenLabs | ask them to send us *their* target audio | then we never generate anything, and we score the voice they already chose |

### Open models that are actually usable commercially

| Model | Code / weights licence | Commercial in a paid product? | Note |
|---|---|---|---|
| **Chatterbox** (Resemble AI) | MIT / MIT | **yes** | 23+ languages; Turbo 350 M (low VRAM), Nano 110 M runs 3× realtime on 8 CPU cores |
| **CosyVoice2** | Apache-2.0 / Apache-2.0 | **yes** | 9 languages incl. Spanish; bi-streaming, latency as low as ~150 ms; zero-shot cloning |
| **Qwen3-TTS** | Apache-2.0 / Apache-2.0 | **yes** | 10 languages incl. Spanish; ~97 ms synthesis latency, 10 RPS on one H100 |
| **Kokoro-82M** | Apache-2.0 / Apache-2.0 | **yes** | 82 M params, CPU-friendly, Spanish voice included; lowest quality (Elo ~1061) |
| **Piper** | MIT / MIT | **yes** | CPU, very fast, low fidelity |
| **Sesame CSM** | Apache-2.0 | **yes** | English only |
| **Orpheus** | Apache-2.0 (English weights) | English only | multilingual set is a "research release" — check terms |
| F5-TTS | MIT code / **CC-BY-NC-4.0** weights | **no** | the pre-trained weights are NC because of the Emilia training data |
| Coqui XTTS-v2 | MPL-2.0 code / **CPML — non-commercial** | **no** | "allows only non-commercial use of the model and its outputs" |
| Fish Speech | Fish Audio Community License | **no, as a hosted service** | commercial use only below $500k revenue **and** "no license is granted to offer a Hosted Service" — that is exactly our product shape |

> Verify each licence before it goes into a paid product — the distinction between
> "the code is MIT" and "the weights you download are MIT" is where teams get burned.
> Fish Speech is the nastiest one: it is the model people reach for when they want
> multilingual cloning, and its licence forbids precisely what we would do with it.

## The five-day experiment that settles it

Nobody outside this repo can tell us which TTS is "good enough" for our product,
because the metric that matters is ours: **how much does the TTS layer move the
scores we sell?** Five working days, one gold set:

1. Build a **100-file Spanish gold set** — real client-style recordings plus human
   reference translations.
2. Generate the target side four ways: (a) ElevenLabs v4, (b) ElevenLabs v4
   Turbo, (c) Azure Neural, (d) Qwen3-TTS self-hosted.
3. Measure what we actually sell on: **ASR WER of the TTS output against the
   reference transcript**, the **COMET/MetricX deltas** the TTS layer introduces,
   and a **blind listening test** (n ≥ 30 pairs).
4. If (c) or (d) lands inside our tolerance, the CTO's argument is *proven*, we
   keep ~80% of the saving, and we show a chart instead of an assertion.

That experiment is also the answer to "is ElevenLabs worth it?" — it is the only
way to find out whether the quality gap we are paying for matters for our metric
rather than for a demo.

## Questions to ask her (this is what her sentence is really missing)

1. **Volumes.** How many audio minutes and how many jobs per month? The per-minute
   APIs are irrelevant at 50 hours/month and decisive at 5,000.
2. **Where does TTS sit in *their* product?** If they generate the target audio
   themselves, we never need to — we score it.
3. **Do they already pay ElevenLabs?** If yes, the conversation is not "should we
   use ElevenLabs" but "should we evaluate the output you are already paying for".
4. **Do they need voice cloning, and is the clone allowed in their output?**
   Cloning rights are a legal question, not a quality question.
5. **Is the audio cleaning in scope?** If they want every submitted file cleaned
   before scoring, that is a per-minute cost decision (see doc 01).
6. **What is the quality bar?** If they need emotional delivery in Spanish, an
   open model may not be enough for the demo, and the demo is worth $20/month.

## How to answer on the call

> "You're right that ElevenLabs is metered in a way that doesn't fit a per-minute
> product — and we've costed it: at 10,000 five-minute jobs a month their API is
> about $3,700 of our bill, and running the same model on the GPU we already rent
> is about $113. But there's a bigger point: the evaluation service doesn't need
> to generate speech at all. You send us the target audio your engine produced and
> we score it. We use ElevenLabs where it's worth it — demos and onboarding, where
> the quality is what you see — and we self-host where it's not. We won't grade our
> own synthesiser's output anyway; that would be measuring the wrong thing."

That does three things: it takes her objection seriously, it shows the number
instead of arguing, and it moves the conversation to the thing we are actually
for — scoring *her* audio.

## Related

- `docs/client-call/tts_unit_economics.py` — regenerates every number above
- `docs/client-call/01-noisy-and-multispeaker-audio.md` — the audio-quality answer
- `API/API_GUIDE.md` §6.5 — the translate endpoint and its gTTS dependency
