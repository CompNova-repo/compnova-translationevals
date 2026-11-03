"""
Tests for the speech-synthesis cache.

Speech APIs bill per *generation request*, not per download, so a pipeline
that re-synthesises on every run pays for the same audio again and again.
The cache is keyed on (language, text).

No GPU, no network, no model weights: `torch` is stubbed and `gtts` is
replaced with a fake that counts how often it was called.

    python3 API/test_synthesis_cache.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import types

API_DIR = os.path.dirname(os.path.abspath(__file__))
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

import testkit  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  ok   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name} {detail}")


class FakeGTTTS:
    """Stands in for `gtts.gTTS`; counts calls and writes a small file."""

    calls = 0

    def __init__(self, text=None, lang=None):
        self.text = text
        self.lang = lang

    def save(self, path):
        FakeGTTTS.calls += 1
        with open(path, "wb") as fh:
            fh.write(b"ID3FAKEMP3" + (self.text or "").encode("utf-8"))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="compnova_tts_cache_test_")
    os.environ["EVAL_TTS_CACHE_DIR"] = os.path.join(tmp, "cache")

    testkit.install_torch_stub()
    sys.modules["gtts"] = types.ModuleType("gtts")
    sys.modules["gtts"].gTTS = FakeGTTTS

    import config
    import eval_core

    print("\n== configuration ==")
    check("the cache directory is configurable",
          config.TTS_CACHE_DIR == os.environ["EVAL_TTS_CACHE_DIR"],
          config.TTS_CACHE_DIR)

    print("\n== caching ==")
    FakeGTTTS.calls = 0
    first = eval_core.synthesize_speech("Hola a todos", "spa")
    check("the first call synthesises", FakeGTTTS.calls == 1, str(FakeGTTTS.calls))
    check("the file exists", os.path.exists(first), first)
    check("it lands in the cache directory",
          first.startswith(config.TTS_CACHE_DIR), first)

    second = eval_core.synthesize_speech("Hola a todos", "spa")
    check("the second call does not re-synthesise", FakeGTTTS.calls == 1,
          str(FakeGTTTS.calls))
    check("it returns the same path", first == second, f"{first} vs {second}")

    other_text = eval_core.synthesize_speech("Buenos dias equipo", "spa")
    check("different text is synthesised separately", FakeGTTTS.calls == 2,
          str(FakeGTTTS.calls))
    check("and gets its own cache entry", other_text != first, other_text)

    other_lang = eval_core.synthesize_speech("Hola a todos", "eng")
    check("a different language is a different entry", FakeGTTTS.calls == 3,
          str(FakeGTTTS.calls))
    check("and gets its own file", other_lang != first, other_lang)

    print("\n== resilience ==")
    # A corrupt cache entry must not be served; it must be re-synthesised.
    with open(first, "wb") as fh:
        fh.write(b"")
    FakeGTTTS.calls = 0
    eval_core.synthesize_speech("Hola a todos", "spa")
    check("an empty cache entry is re-synthesised", FakeGTTTS.calls == 1,
          str(FakeGTTTS.calls))

    print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
