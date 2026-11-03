"""
Unit economics for the speech side of the evaluation product.

Run it to regenerate every number in
docs/client-call/02-tts-economics.md:

    python3 docs/client-call/tts_unit_economics.py

All prices are the ones published on the date in each row's source comment.
Re-check them before putting a number in front of a customer.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# prices (see the source comment on every line)
# --------------------------------------------------------------------------- #

# ElevenLabs API, pay-as-you-go, https://elevenlabs.io/pricing/api (Nov 2026)
ELEVENLABS_V4_PER_1K_CHARS = 0.08          # v4 / v3 / v2 multilingual
ELEVENLABS_TURBO_PER_1K_CHARS = 0.04       # v4 turbo / v3 conversational / flash-turbo
ELEVENLABS_SCRIBE_V2_PER_HOUR = 0.22       # speech-to-text
ELEVENLABS_VOICE_ISOLATOR_PER_MIN = 0.12   # background-noise removal
ELEVENLABS_DUBBING_V1_PER_MIN = 0.33

# Competing hosted TTS, per 1M characters (Nov 2026). Sources in the doc.
COMPETITOR_PER_1M_CHARS = {
    "Azure Neural": 15.0,
    "AWS Polly Neural": 16.0,
    "Google Neural2": 16.0,
    "Deepgram Aura-1": 15.0,
    "Deepgram Aura-2": 30.0,
    "Google Chirp 3 HD": 30.0,
    "Cartesia Sonic-3.6": 40.0,
    "Nari (hosted Qwen3-TTS)": 5.0,
}

# ElevenLabs subscriptions, https://elevenlabs.io/pricing (Nov 2026)
ELEVENLABS_PLANS = {
    # plan: (usd per month, credits per month)
    "Free": (0.0, 10_000),
    "Starter": (6.0, 30_000),
    "Creator": (22.0, 121_000),
    "Pro": (99.0, 600_000),
    "Scale": (299.0, 1_800_000),
    "Business": (990.0, 6_000_000),
}

# RunPod pod pricing, https://www.runpod.io/pricing (page dated Oct 28 2026)
# The Community Cloud tab is cheaper than Secure Cloud; both are listed.
RUNPOD_PODS_PER_HOUR = {
    "RTX A6000 48GB": 0.33,
    "A40 48GB": 0.35,
    "L40S 48GB": 0.79,
    "A100 80GB": 1.39,
    "H100 80GB": 2.69,
}

# Fixed cost of running our own TTS stack: one always-on GPU plus the ops
# labour to keep it patched and monitored. These are assumptions - replace
# them with our real numbers before quoting anyone.
SELF_HOST_FIXED_COST_PER_MONTH = 1015.0   # one A100 at $1.39/hr x 730 h
SELF_HOST_OPS_PER_MONTH = 2000.0

# --------------------------------------------------------------------------- #
# workload assumptions
# --------------------------------------------------------------------------- #

CHARS_PER_MINUTE_OF_SPEECH = 1000   # ElevenLabs' own convention on its pricing page
SECONDS_OF_AUDIO_PER_HOUR = 3600


def elevenlabs_cost_per_minute_audio(per_1k_chars: float) -> float:
    """Cost of one minute of generated speech."""
    chars = CHARS_PER_MINUTE_OF_SPEECH
    return chars / 1000.0 * per_1k_chars


def self_hosted_cost_per_hour_audio(gpu_per_hour: float,
                                    realtime_factor: float) -> float:
    """Cost of one hour of audio produced on a rented GPU.

    realtime_factor = how many seconds of audio the model generates per second
    of GPU time (30 = 30x faster than real time).
    """
    gpu_hours_for_one_audio_hour = 1.0 / float(realtime_factor)
    return gpu_per_hour * gpu_hours_for_one_audio_hour


def evaluation_job_cost(job_minutes: float, tts_per_minute: float,
                        gpu_per_hour: float, evaluation_seconds: float) -> dict:
    """One evaluation job: translate (TTS) + score (GPU)."""
    tts = job_minutes * tts_per_minute
    scoring = gpu_per_hour * (evaluation_seconds / 3600.0)
    return {
        "tts": tts,
        "scoring": scoring,
        "total": tts + scoring,
        "tts_share": (tts / (tts + scoring)) if (tts + scoring) else 0.0,
    }


def monthly(jobs: int, job_minutes: float, tts_per_minute: float,
            gpu_per_hour: float, evaluation_seconds: float) -> dict:
    per_job = evaluation_job_cost(job_minutes, tts_per_minute, gpu_per_hour,
                                  evaluation_seconds)
    return {
        "jobs": jobs,
        "per_job": per_job,
        "monthly_total": per_job["total"] * jobs,
        "monthly_tts": per_job["tts"] * jobs,
        "monthly_scoring": per_job["scoring"] * jobs,
        "gpu_hours": jobs * evaluation_seconds / 3600.0,
    }


def fmt_money(value: float) -> str:
    if value == 0:
        return "$0.00"
    if value < 0.0001:
        return f"${value:.6f}"
    if value < 0.01:
        return f"${value:.5f}"
    if value < 1:
        return f"${value:.4f}"
    return f"${value:,.2f}"


def table(rows, headers) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    out = ["| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return "\n".join(out)


def main() -> None:
    print("# Speech cost model\n")
    print("Generated by docs/client-call/tts_unit_economics.py "
          "(prices: ElevenLabs API pricing page and RunPod pricing page, Nov 2026)\n")

    print("## Cost of one minute of generated speech\n")
    rows = [
        ["ElevenLabs v4 (or v3 / v2 multilingual)", fmt_money(
            elevenlabs_cost_per_minute_audio(ELEVENLABS_V4_PER_1K_CHARS)),
         "$" + f"{ELEVENLABS_V4_PER_1K_CHARS:.2f} / 1k chars"],
        ["ElevenLabs v4 Turbo / Flash / Turbo", fmt_money(
            elevenlabs_cost_per_minute_audio(ELEVENLABS_TURBO_PER_1K_CHARS)),
         "$" + f"{ELEVENLABS_TURBO_PER_1K_CHARS:.2f} / 1k chars"],
        ["ElevenLabs Voice Isolator (noise removal)",
         fmt_money(ELEVENLABS_VOICE_ISOLATOR_PER_MIN), "$0.12 / min"],
    ]
    for name, per_1m in COMPETITOR_PER_1M_CHARS.items():
        rows.append([
            f"{name} (hosted API)",
            fmt_money(per_1m / 1000.0 * (CHARS_PER_MINUTE_OF_SPEECH / 1000.0)),
            f"${per_1m:.0f} / 1M chars",
        ])
    for gpu, price in RUNPOD_PODS_PER_HOUR.items():
        for rtf in (20, 50):
            rows.append([
                f"Self-hosted TTS on {gpu} (${price:.2f}/hr, {rtf}x realtime)",
                fmt_money(self_hosted_cost_per_hour_audio(price, rtf) / 60.0),
                f"${price:.2f} / gpu-hour",
            ])
    print(table(rows, ["Option", "per minute of audio", "list price"]))
    print()

    print("## One evaluation job (5-minute source, 5-minute target)\n")
    print("Scoring cost uses the measured pipeline time (~63 s wall clock for the")
    print("whole evaluation on one GPU, see API/API_GUIDE.md §9).\n")
    job_minutes = 5.0
    evaluation_seconds = 63.0
    for gpu, price in (("RTX A6000 48GB", 0.59), ("A100 80GB", 1.79)):
        for label, tts_rate in (
            ("ElevenLabs v4", elevenlabs_cost_per_minute_audio(ELEVENLABS_V4_PER_1K_CHARS)),
            ("ElevenLabs Flash/Turbo", elevenlabs_cost_per_minute_audio(ELEVENLABS_TURBO_PER_1K_CHARS)),
            ("Self-hosted TTS (50x realtime on the same GPU)",
             self_hosted_cost_per_hour_audio(price, 50) / 60.0),
        ):
            costs = evaluation_job_cost(job_minutes, tts_rate, price,
                                        evaluation_seconds)
            print(f"- {gpu} · {label}: total {fmt_money(costs['total'])} "
                  f"(TTS {fmt_money(costs['tts'])}, scoring "
                  f"{fmt_money(costs['scoring'])}, TTS share "
                  f"{costs['tts_share'] * 100:.1f}%)")
    print()

    print("## Monthly bill for N jobs (5-minute files, 63 s of GPU time each)\n")
    rows = []
    for jobs in (1_000, 10_000, 100_000):
        eleven = monthly(jobs, job_minutes,
                          elevenlabs_cost_per_minute_audio(ELEVENLABS_V4_PER_1K_CHARS),
                          0.33, evaluation_seconds)
        turbo = monthly(jobs, job_minutes,
                        elevenlabs_cost_per_minute_audio(ELEVENLABS_TURBO_PER_1K_CHARS),
                        0.33, evaluation_seconds)
        own = monthly(jobs, job_minutes,
                      self_hosted_cost_per_hour_audio(0.33, 50) / 60.0,
                      0.33, evaluation_seconds)
        own_total = own["monthly_total"] + (
            SELF_HOST_FIXED_COST_PER_MONTH + SELF_HOST_OPS_PER_MONTH)
        rows.append([
            f"{jobs:,}",
            f"{eleven['gpu_hours']:.0f}",
            fmt_money(eleven["monthly_total"]),
            fmt_money(turbo["monthly_total"]),
            fmt_money(own_total),
            f"{eleven['monthly_tts'] / eleven['monthly_total'] * 100:.0f}%",
        ])
    print(table(rows, ["jobs / month", "GPU hours", "ElevenLabs v4",
                       "ElevenLabs Flash/Turbo", "Self-hosted TTS (incl. fixed)",
                       "TTS share of the bill (v4)"]))
    print()

    print("## Break-even: when is self-hosting cheaper than ElevenLabs v4?\n")
    tts_rate = elevenlabs_cost_per_minute_audio(ELEVENLABS_V4_PER_1K_CHARS)
    own_rate = self_hosted_cost_per_hour_audio(0.33, 50) / 60.0
    azure_rate = COMPETITOR_PER_1M_CHARS["Azure Neural"] / 1000.0 * (
        CHARS_PER_MINUTE_OF_SPEECH / 1000.0)
    ratio = tts_rate / own_rate
    print(f"- ElevenLabs v4 costs {fmt_money(tts_rate)} per audio minute; "
          f"self-hosted costs {fmt_money(own_rate)}; "
          f"Azure Neural costs {fmt_money(azure_rate)}.")
    print(f"- Per minute, self-hosting is {ratio:.0f}x cheaper than ElevenLabs - "
          f"but it is not free: it carries a fixed cost of "
          f"{fmt_money(SELF_HOST_FIXED_COST_PER_MONTH + SELF_HOST_OPS_PER_MONTH)}/month")
    print("  (one always-on GPU plus the ops labour to keep it patched and monitored).")
    for label, other_rate in (("ElevenLabs v4", tts_rate),
                              ("Azure Neural", azure_rate),
                              ("ElevenLabs v4 Turbo",
                               elevenlabs_cost_per_minute_audio(ELEVENLABS_TURBO_PER_1K_CHARS))):
        saved_per_job = job_minutes * (other_rate - own_rate)
        if saved_per_job <= 0:
            continue
        break_even = (SELF_HOST_FIXED_COST_PER_MONTH + SELF_HOST_OPS_PER_MONTH) / saved_per_job
        print(f"- Self-hosting overtakes {label} at about "
              f"**{break_even:,.0f} jobs/month**.")
    print("- Below that, a cheap hosted API (Azure Neural / Google Neural2 / "
          "AWS Polly Neural at $15-16 per 1M characters) is the better move: it")
    print("  costs nothing to operate and is still 5x cheaper than ElevenLabs.")
    print("- Self-hosting's edge is privacy, latency and voice-cloning control - "
          "not unit price.")
    print()

    print("## The trap that actually costs money\n")
    print("ElevenLabs charges credits **per generation request, not per download**")
    print("(their own pricing FAQ). So re-running the pipeline re-bills the whole")
    print("corpus. One full re-run at 100k jobs/month costs $40,000 (v4) or")
    print("$20,000 (Turbo). The fix is a content-addressed cache keyed on")
    print("(target_text, voice_id, model, params, seed) - typically removes 50-90%")
    print("of TTS spend, and it is worth doing before switching vendors.")
    print()

    print("## ElevenLabs plan ladder (the 'commitment' problem)\n")
    rows = []
    for plan, (usd, credits) in ELEVENLABS_PLANS.items():
        per_1k = (usd / credits * 1000) if credits else 0.0
        rows.append([plan, fmt_money(usd), f"{credits:,}",
                     fmt_money(per_1k) if credits else "-",
                     f"{credits / CHARS_PER_MINUTE_OF_SPEECH / 60:.1f} h"])
    print(table(rows, ["plan", "$/month", "credits/month",
                       "effective $/1k chars", "audio included"]))
    print()
    print("The Business plan lists 'low-latency TTS as low as 5c/minute' - that")
    print("is the volume price, and it only starts at $990/month.")


if __name__ == "__main__":
    main()
