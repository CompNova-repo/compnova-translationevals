"""S2ST evaluation monitor: a read-only view of results stored by the API.

This app loads no models and evaluates nothing itself. Audio pairs are
evaluated by POSTing them to the API's /evaluate endpoint; this app lists
and displays the stored results via GET /evaluations and GET /evaluations/{id}.

Requirements: streamlit, altair, requests, pandas.
"""
import html
import os
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_API_URL = os.environ.get("EVAL_API_URL", "http://localhost:8000")
COMET_FLAG_THRESHOLD = 0.4  # same threshold as run_eval_v3.py
METRICX_MAX = 25.0          # MetricX shown as 1 - score/25, to match COMET's 0-1 scale

st.set_page_config(page_title="S2ST evaluation monitor", layout="wide")


# ---------- helpers ----------

def auth_headers(api_key):
    """The API accepts an X-API-Key header when it is configured with keys."""
    return {"X-API-Key": api_key} if api_key else {}


def api_get(base_url, path, api_key="", **params):
    resp = requests.get(
        f"{base_url}{path}", params=params, timeout=15, headers=auth_headers(api_key)
    )
    resp.raise_for_status()
    return resp.json()


def api_post(base_url, path, files=None, data=None, params=None, api_key="",
             timeout=1800):
    """POST with the API's error envelope turned into a readable exception."""
    resp = requests.post(
        f"{base_url}{path}",
        files=files,
        data=data,
        params=params,
        timeout=(15, timeout),
        headers=auth_headers(api_key),
    )
    if not resp.ok:
        try:
            detail = resp.json().get("error", {})
            raise RuntimeError(
                f"{detail.get('code', 'HTTP ' + str(resp.status_code))}: "
                f"{detail.get('message', resp.text[:300])}"
            )
        except ValueError:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


@st.cache_data(ttl=10, show_spinner=False)
def cached_health(base_url, api_key=""):
    return api_get(base_url, "/health", api_key)


@st.cache_data(ttl=10, show_spinner=False)
def cached_list(base_url, limit, api_key=""):
    return api_get(base_url, "/evaluations", api_key, limit=limit)


@st.cache_data(max_entries=50, show_spinner=False)
def cached_record(base_url, eval_id, api_key=""):
    """A stored evaluation never changes, so it is cached until the app restarts."""
    return api_get(base_url, f"/evaluations/{eval_id}", api_key)


def fmt_time(ts):
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return "unknown time"


def eval_label(meta):
    src = meta.get("source_name") or "source"
    tgt = meta.get("target_name") or "target"
    return f"{fmt_time(meta.get('created_at'))}  |  {src} to {tgt}  ({meta['id']})"


def find_col(df, *needles):
    """First numeric column whose name contains every needle (case-insensitive)."""
    for col in df.columns:
        name = str(col).lower()
        if all(n in name for n in needles) and pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


# ---------- sections ----------

def render_summary(summary):
    if not summary:
        return
    st.subheader("Summary")
    if not isinstance(summary, dict):
        st.json(summary)
        return
    scalars = {k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))}
    if scalars:
        cols = st.columns(min(len(scalars), 4))
        for i, (key, val) in enumerate(scalars.items()):
            shown = f"{val:.3f}" if isinstance(val, float) else str(val)
            cols[i % len(cols)].metric(str(key).replace("_", " ").capitalize(), shown)
    nested = {k: v for k, v in summary.items() if k not in scalars}
    if nested:
        with st.expander("Other summary fields"):
            st.json(nested)


def build_score_chart(scores, flagged):
    """Altair chart spec for the per-chunk scores (rendered in the browser, not by Streamlit)."""
    long = (
        scores.reset_index()
        .melt(id_vars="chunk", var_name="metric", value_name="score")
        .dropna()
    )
    base = alt.Chart(long).encode(
        x=alt.X("chunk:Q", title="Chunk", axis=alt.Axis(tickMinStep=1, format="d")),
        y=alt.Y("score:Q", title="Score (higher is better)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("metric:N", title=None, legend=alt.Legend(orient="top")),
    )
    tooltip = [
        alt.Tooltip("chunk:Q", title="Chunk", format="d"),
        alt.Tooltip("metric:N", title="Metric"),
        alt.Tooltip("score:Q", title="Score", format=".3f"),
    ]
    layers = [
        base.mark_line(),
        base.mark_point(filled=True, size=60).encode(tooltip=tooltip),
        # Large invisible targets so a click doesn't have to land exactly on a point.
        base.mark_point(size=500, opacity=0.001, cursor="pointer").encode(tooltip=tooltip),
    ]
    flagged_points = long[(long["metric"] == "COMET-Kiwi") & long["chunk"].isin(flagged)]
    if not flagged_points.empty:
        layers.append(
            alt.Chart(flagged_points)
            .mark_point(shape="diamond", size=220, color="#d62728", strokeWidth=2)
            .encode(x="chunk:Q", y="score:Q")
        )
    return alt.layer(*layers).properties(height=300, width="container")


def render_interactive_chunks(chart, df, comet_col, flagged):
    """Chart plus chunk list in one browser-side component.

    Clicking a point opens and scrolls to that chunk entirely in the browser,
    so there is no Streamlit rerun and no round trip to the server.
    """
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    items = []
    for i, row in df.iterrows():
        label = f"Chunk {i}"
        if comet_col:
            label += f", COMET {row[comet_col]:.3f}"
        if i in flagged:
            label += ", below threshold"
        # Transcript text comes from audio we don't control, so it is always escaped.
        fields = "".join(
            f"<div class='field'><div class='name'>{html.escape(str(c))}</div>"
            f"<div>{html.escape(str(row[c]))}</div></div>"
            for c in text_cols
        )
        nums = ", ".join(
            f"{c}: {row[c]:.3f}" if isinstance(row[c], float) else f"{c}: {row[c]}"
            for c in num_cols
        )
        css_class = "flag" if i in flagged else ""
        items.append(
            f"<details id='chunk-{i}' class='{css_class}'>"
            f"<summary>{html.escape(label)}</summary>{fields}"
            f"<div class='nums'>{html.escape(nums)}</div></details>"
        )

    page = f"""
<style>
  :root {{ color-scheme: light dark; --fg:#1f2328; --muted:#6b7280; --line:#d0d7de; --flag:rgba(255,75,75,.14); --sel:rgba(60,130,255,.18); }}
  @media (prefers-color-scheme: dark) {{ :root {{ --fg:#e6edf3; --muted:#9aa4af; --line:#30363d; }} }}
  body {{ margin:0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--fg); background:transparent; }}
  #chart {{ width:100%; }}
  .hint {{ color:var(--muted); font-size:13px; margin:4px 0 12px; }}
  h3 {{ font-size:17px; margin:8px 0; }}
  details {{ border:1px solid var(--line); border-radius:6px; margin:6px 0; padding:6px 10px; }}
  details.flag {{ background:var(--flag); }}
  details.sel {{ background:var(--sel); outline:2px solid #3c82ff; }}
  summary {{ cursor:pointer; font-weight:600; }}
  .field {{ margin:8px 0; }}
  .name {{ font-size:12px; color:var(--muted); }}
  .nums {{ font-size:12px; color:var(--muted); margin-top:6px; }}
</style>
<div id="chart"></div>
<p class="hint">Click a point to open that chunk below.</p>
<h3>Chunk details</h3>
<div id="list">{''.join(items)}</div>
<script src="https://cdn.jsdelivr.net/npm/vega@6"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@6"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@7"></script>
<script>
  const spec = {chart.to_json()};
  vegaEmbed("#chart", spec, {{actions: false}}).then(result => {{
    result.view.addEventListener("click", (event, item) => {{
      if (!item || !item.datum || item.datum.chunk === undefined) return;
      const el = document.getElementById("chunk-" + Math.round(item.datum.chunk));
      if (!el) return;
      document.querySelectorAll("details.sel").forEach(d => d.classList.remove("sel"));
      el.open = true;
      el.classList.add("sel");
      el.scrollIntoView({{behavior: "smooth", block: "start"}});
    }});
  }}).catch(err => {{
    document.getElementById("chart").textContent = "Chart failed to load: " + err;
  }});
</script>
"""
    if hasattr(st, "iframe"):  # newer Streamlit; components.html is being retired
        st.iframe(page, height=900)
    else:
        components.html(page, height=900, scrolling=True)


def render_chunks(chunks):
    st.subheader("Per-chunk scores")
    if not chunks:
        st.info("This evaluation has no aligned chunks.")
        return

    df = pd.json_normalize(chunks)
    df.index = range(1, len(df) + 1)
    df.index.name = "chunk"

    comet_col = find_col(df, "comet")
    mx_norm_col = find_col(df, "metricx", "norm")
    mx_raw_col = find_col(df, "metricx")

    scores = pd.DataFrame(index=df.index)
    if comet_col:
        scores["COMET-Kiwi"] = df[comet_col]
    if mx_norm_col:
        scores["MetricX (normalised)"] = df[mx_norm_col]
    elif mx_raw_col:
        scores["MetricX (normalised)"] = (1 - df[mx_raw_col] / METRICX_MAX).clip(0, 1)

    flagged = []
    if comet_col:
        flagged = df.index[df[comet_col] < COMET_FLAG_THRESHOLD].tolist()
        if flagged:
            st.warning(
                f"{len(flagged)} chunk(s) scored below COMET {COMET_FLAG_THRESHOLD}: "
                + ", ".join(map(str, flagged))
            )
        else:
            st.success(f"No chunks scored below COMET {COMET_FLAG_THRESHOLD}.")

    if len(scores.columns):
        render_interactive_chunks(build_score_chart(scores, flagged), df, comet_col, flagged)
    else:
        st.caption("No COMET or MetricX columns found in the chunk data, so there is no chart.")

    with st.expander("All chunks as a table"):
        if comet_col:
            def highlight(row):
                bad = row[comet_col] < COMET_FLAG_THRESHOLD
                return ["background-color: rgba(255, 75, 75, 0.18)" if bad else ""] * len(row)
            st.dataframe(df.style.apply(highlight, axis=1).format(precision=3))
        else:
            st.dataframe(df)


def render_transcripts(transcripts, eval_id):
    if not transcripts:
        return
    st.subheader("Transcripts")
    if not isinstance(transcripts, dict):
        st.json(transcripts)
        return
    cols = st.columns(len(transcripts))
    for col, (side, text) in zip(cols, transcripts.items()):
        if isinstance(text, list):
            text = "\n".join(
                t.get("text", str(t)) if isinstance(t, dict) else str(t) for t in text
            )
        with col:
            st.text_area(
                str(side).capitalize(),
                value=str(text),
                height=220,
                disabled=True,
                key=f"transcript_{eval_id}_{side}",  # per-evaluation key, so switching refreshes it
            )


def render_unmatched(unmatched):
    if not unmatched:
        return
    if isinstance(unmatched, dict):
        count = sum(len(v) for v in unmatched.values() if isinstance(v, list))
    else:
        count = len(unmatched)
    with st.expander(f"Unmatched segments ({count})"):
        st.json(unmatched)


# ---------- page ----------

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("Evaluation API URL", DEFAULT_API_URL).rstrip("/")
    api_key = st.text_input(
        "API key (X-API-Key)",
        os.environ.get("EVAL_API_KEY", ""),
        type="password",
        help="Only needed when the service was started with EVAL_API_KEYS set.",
    )
    follow_latest = st.toggle("Always show the latest evaluation", value=True)
    limit = st.slider("Evaluations to list", 5, 50, 20)
    if st.button("Refresh"):
        cached_health.clear()
        cached_list.clear()

st.title("S2ST evaluation monitor")
st.caption(
    "Results from the evaluation API. Upload a pair in the **New evaluation** tab "
    "to add more."
)

try:
    cached_health(api_url, api_key)
except requests.RequestException as e:
    st.error(
        f"Can't reach the API at {api_url}. Check that it is running and the URL "
        f"is right. ({e})"
    )
    st.stop()

tab_results, tab_new = st.tabs(["Results", "New evaluation"])


# ---------- new evaluation ----------

def _render_result(result, eval_id):
    summary = result.get("summary", {})
    render_summary(summary)
    cols = st.columns(4)
    cols[0].metric("Chunks", summary.get("chunks", 0))
    cols[1].metric("COMET-Kiwi", summary.get("comet_kiwi"))
    cols[2].metric("MetricX-24", summary.get("metricx_24"))
    cols[3].metric("Sentiment match %", summary.get("sentiment_agreement_pct"))
    render_chunks(result.get("chunks"))
    render_transcripts(result.get("transcripts"), eval_id)
    render_unmatched(result.get("unmatched"))


with tab_new:
    st.subheader("Evaluate an audio pair")
    st.caption(
        "Upload the source audio and its translation. Leave the translation "
        "empty and pick a language to have the API translate it for you."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        source_file = st.file_uploader(
            "Source audio", type=["wav", "mp3", "flac", "m4a", "ogg"]
        )
    with col_b:
        target_file = st.file_uploader(
            "Translated audio (optional)", type=["wav", "mp3", "flac", "m4a", "ogg"]
        )

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        target_lang = st.selectbox(
            "Translate into",
            ["spa", "fra", "deu", "cmn", "ita", "por", "jpn", "eng"],
            help="Used when no translated audio is uploaded.",
        )
    with c2:
        min_similarity = st.slider("Min similarity", 0.0, 1.0, 0.5, 0.05)
    with c3:
        max_merge = st.slider("Max merge", 1, 3, 2)

    params = {"min_similarity": min_similarity, "max_merge": max_merge}

    b1, b2 = st.columns(2)
    with b1:
        evaluate_clicked = st.button(
            "Evaluate the pair",
            type="primary",
            disabled=source_file is None or target_file is None,
        )
    with b2:
        translate_clicked = st.button(
            "Translate, then evaluate",
            disabled=source_file is None or target_file is not None,
        )

    if evaluate_clicked:
        files = {
            "source_audio": (
                source_file.name, source_file.getvalue(), "application/octet-stream"
            ),
            "target_audio": (
                target_file.name, target_file.getvalue(), "application/octet-stream"
            ),
        }
        with st.spinner("Evaluating - this takes a few minutes per pair..."):
            try:
                result = api_post(
                    api_url, "/evaluate", files=files, params=params, api_key=api_key
                )
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
            else:
                st.success(f"Done - evaluation `{result.get('id')}`")
                _render_result(result, result.get("id", "new"))
                cached_list.clear()
                cached_record.clear()

    if translate_clicked:
        files = {
            "source_audio": (
                source_file.name, source_file.getvalue(), "application/octet-stream"
            ),
        }
        data = {"target_lang": target_lang}
        with st.spinner("Submitting - the API translates and then evaluates..."):
            try:
                job = api_post(
                    api_url, "/jobs", files=files, data=data, params=params,
                    api_key=api_key,
                )
            except Exception as e:
                st.error(f"Could not submit the job: {e}")
            else:
                job_id = job.get("job_id")
                st.info(f"Job `{job_id}` queued.")
                with st.spinner("Waiting for the result..."):
                    import time

                    final = None
                    for _ in range(720):  # up to an hour
                        time.sleep(5)
                        try:
                            record = api_get(
                                api_url, f"/jobs/{job_id}", api_key
                            )
                        except requests.RequestException:
                            continue
                        status = record.get("status")
                        if status == "done":
                            final = record.get("result") or {}
                            break
                        if status in ("failed", "cancelled"):
                            st.error(
                                f"The job {status}: "
                                f"{(record.get('error') or {}).get('message', '')}"
                            )
                            break
                if final:
                    st.success(f"Done - evaluation `{final.get('id')}`")
                    _render_result(final, final.get("id", job_id))
                    cached_list.clear()
                    cached_record.clear()


# ---------- results ----------

with tab_results:
    try:
        listed = cached_list(api_url, limit, api_key)
    except requests.RequestException as e:
        st.error(
            f"The API is up but listing evaluations failed. Does it have the "
            f"/evaluations endpoint? ({e})"
        )
        st.stop()

    items = listed.get("items", listed) if isinstance(listed, dict) else listed
    if not items:
        st.info(
            "No evaluations yet. Upload a pair in the 'New evaluation' tab, "
            "then press Refresh."
        )
    else:
        if follow_latest:
            selected = items[0]["id"]
        else:
            labels = {m["id"]: eval_label(m) for m in items}
            selected = st.selectbox(
                "Evaluation (newest first)", list(labels), format_func=labels.get
            )

        try:
            record = cached_record(api_url, selected, api_key)
        except requests.RequestException as e:
            st.error(f"Couldn't load evaluation {selected}. ({e})")
            st.stop()

        result = record.get("result", {})
        st.markdown(
            f"**{record.get('source_name') or 'source'}** to "
            f"**{record.get('target_name') or 'target'}**, "
            f"evaluated {fmt_time(record.get('created_at'))} (id `{record['id']}`)"
        )

        render_summary(result.get("summary"))
        render_chunks(result.get("chunks"))
        render_transcripts(result.get("transcripts"), record["id"])
        render_unmatched(result.get("unmatched"))

        with st.expander("Raw JSON"):
            st.json(record)
