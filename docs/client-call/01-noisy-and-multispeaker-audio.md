# Client call: "what if the recording has background noise and other people talking in the background?"

The question, as asked:

> What if the audio file (source or target) has background noise where there's multiple
> languages being spoken in the background (as noise) — how do we differentiate and get
> only the right speaker audio?

## The short answer

Three things, in the order we would say them on a call:

1. **We do not try to "clean" the audio before scoring it.** Removing audio changes
   what is being measured, and no open model separates a target speaker from an
   overlapping voice in *another language* reliably enough to bet a quality score on it.
2. **We measure the recording and tell you it is contaminated.** Signal-to-noise ratio,
   how much of the file is speech, how many speakers, how much is overlapping speech,
   which languages are present — and the score carries a flag saying "this number is
   unreliable for this file", instead of quietly being wrong.
3. **We score only the intended speaker when you want us to.** With diarization we find
   the dominant speaker, keep only their turns, and compute the score on that audio.

That third step is implemented and shipped on the branch `feature/audio-quality-gate`
(`API/audio_quality.py`, `POST /audio/quality`).

## Why "separate the voices" is the wrong goal here

There are two different jobs people conflate:

| Job | What it needs | State of the art |
|---|---|---|
| **Denoising** (remove hiss, hum, fan noise) | a noise-suppression model | Mature. Demucs v4, Resemble Enhance, DeepFilterNet3, RNNoise. Works well. |
| **Speaker separation** (keep voice A, drop voice B) | target-speaker extraction (VoiceFilter / SpeakerBeam / TSE models) | Research-grade. No dependable cross-language accuracy, no stable API, and it *changes the audio you are scoring*. |

Diarization sits in between: it tells you *who spoke when*. It does not remove
anything, and it is the part that is reliable enough to build a product on.

The cross-language case is the hard one and worth being honest about: diarization
embeddings are trained on speaker identity, which is partly entangled with language
and channel. A background voice speaking Spanish while the target speaks English is
exactly the case where open models are weakest. So the product answer has to be
"measure and flag", not "magically extract".

## What we actually measure

Implemented in `API/audio_quality.py`, exposed as `POST /audio/quality`, and attached
to every evaluation result as `audio_quality` when `EVAL_AUDIO_QUALITY_MODE=flag`:

| Measurement | How | What it tells you |
|---|---|---|
| `snr_db` | 95th vs 10th percentile of frame loudness | how noisy the recording is (adaptive to the file, not a fixed dBFS) |
| `speech_ratio` | energy VAD (adaptive threshold over the noise floor) | how much of the file is speech at all |
| `speaker_count` | pyannote diarization (optional) | how many people are on the recording |
| `dominant_speaker_ratio` | talk-time share per speaker | is there one intended speaker, or is the recording a conversation? |
| `overlap_ratio` / `overlap_s` | interval intersection of diarization turns | how often two people talk at once |
| `languages` | Whisper language ID | what language the main speech is in |
| `background_languages` | Whisper on the non-dominant turns | **proof that the background voice is in another language** — the client's exact scenario |

Flags: `noisy`, `mostly_silence`, `multi_speaker`, `no_single_dominant_speaker`,
`overlapping_speech`, `background_speech`, `very_short_speech`.

The core measurements are NumPy-only, so they always run. Silero VAD, pyannote and
Whisper improve the report when installed.

## The three modes we offer

| Mode | What the client gets | When to use it |
|---|---|---|
| `off` (default today) | no change | nothing to do |
| `flag` | every result carries `audio_quality`; the score is unchanged, but you know which files to distrust | **default recommendation** - it is honest and it is cheap |
| `filter` | the transcript and the emotion model only see the dominant speaker's turns | when the client's corpus is noisy and they want a number anyway |

```bash
# pre-flight check on one file
curl -s -X POST "$BASE/audio/quality" -F "audio=@EN_001.wav"
```

```json
{
  "source": {
    "duration_s": 63.4, "snr_db": 18.2, "speech_ratio": 0.71,
    "speaker_count": 2, "dominant_speaker_ratio": 0.83,
    "overlap_ratio": 0.06, "languages": ["eng"],
    "background_languages": ["spa"], "background_speech_detected": true,
    "flags": ["multi_speaker", "background_speech"],
    "usable": true, "contaminated": true
  },
  "flags": ["multi_speaker", "background_speech"],
  "contaminated": true,
  "usable": true
}
```

`usable: false` means "we will not put a number on this file" — silence, or no
dominant speaker.

## What the pipeline does with the measurement

- **Transcription.** In `filter` mode only the dominant speaker's windows are sent to
  Whisper, so the background voice in the other language never enters the transcript.
- **Alignment and scoring.** COMET-Kiwi and MetricX-24 score the (filtered) transcript
  pairs, so a background sentence can no longer be scored as if it were an omission
  or a mistranslation.
- **Emotion.** The SER model is English-only *and* very sensitive to mixed audio — in
  `filter` mode it sees the same filtered audio, which removes one of the two reasons
  the cross-lingual emotion number is weak (the other one — the checkpoint being
  English-only — is still there, and is called out in `API/ROADMAP.md` §2.8).

## Limits we will state, not hide

- Overlapping speech (two people at once) is **measured, not resolved**. A score for
  such a file is unreliable; the flag says so.
- Diarization needs the gated `pyannote/speaker-diarization-3.1` model, so the pod
  needs an `HF_TOKEN` with those terms accepted. Without it the report falls back to
  energy measurements and there is no `speaker_count`.
- `filter` mode cuts on speaker turns; a chunk that straddles a speaker change can
  still contain two voices. Word-level speaker attribution (aligning diarization to
  Whisper word timestamps) is the next step and is on the roadmap.
- We are not going to claim "we removed the background voice". The honest claim is
  "we detected it, we flagged it, and we scored the intended speaker".

## How to answer if she pushes for separation

> "We can add separation - Demucs or a Voice Isolator API - but it changes the audio
> we score, and for a background voice in another language the open models are not
> accurate enough for us to put a client-facing number on. What we would do instead
> is detect it, tell you the file is contaminated, and score the speaker you meant.
> If you want, we can run separation as an *additional* signal next to the clean
> score, so you can see how much it moved."

That framing keeps the measurement honest, gives them the option, and does not put
the product on a research-grade dependency.

## Related

- `API/audio_quality.py` — the implementation
- `API/API_GUIDE.md` §6.16 — the endpoint and the flag vocabulary
- `API/ROADMAP.md` §2.11 — what is still open in this area
- `docs/client-call/02-speech-costs.md` — the cost side of the same question
  (ElevenLabs Voice Isolator and friends are billed per minute)
