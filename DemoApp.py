"""S2ST evaluation monitor: a read-only view of results stored by the API.

This app loads no models and evaluates nothing itself. Audio pairs are
evaluated by POSTing them to the API's /evaluate endpoint; this app lists
and displays the stored results via GET /evaluations and GET /evaluations/{id}.

Requirements: streamlit (1.35+), altair, requests, pandas.
"""
import os
import time
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

def api_get(base_url, path, **params):
    resp = requests.get(f"{base_url}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


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


def scroll_to(anchor_id):
    """Scroll the page to an element id. The timestamp forces the script to re-run."""
    components.html(
        f"""<script>
        const el = window.parent.document.getElementById("{anchor_id}");
        if (el) el.scrollIntoView({{behavior: "smooth", block: "start"}});
        // {time.time()}
        </script>""",
        height=0,
    )


def selected_chunk_from(event):
    """Pull the clicked chunk number out of a Streamlit chart selection event."""
    try:
        picked = event.selection.get("pick")
    except AttributeError:
        return None
    if isinstance(picked, list) and picked:
        value = picked[0].get("chunk")
    elif isinstance(picked, dict) and picked.get("chunk"):
        value = picked["chunk"]
        value = value[0] if isinstance(value, list) else value
    else:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def render_score_chart(scores, flagged, eval_id):
    """Clickable per-chunk chart. Returns the clicked chunk number, or None."""
    long = (
        scores.reset_index()
        .melt(id_vars="chunk", var_name="metric", value_name="score")
        .dropna()
    )
    pick = alt.selection_point(name="pick", fields=["chunk"], on="click",
                               nearest=True, empty=False)
    base = alt.Chart(long).encode(
        x=alt.X("chunk:Q", title="Chunk", axis=alt.Axis(tickMinStep=1, format="d")),
        y=alt.Y("score:Q", title="Score (higher is better)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("metric:N", title=None, legend=alt.Legend(orient="top")),
    )
    lines = base.mark_line()
    points = base.mark_point(filled=True, cursor="pointer").encode(
        size=alt.condition(pick, alt.value(180), alt.value(45)),
        tooltip=[
            alt.Tooltip("chunk:Q", title="Chunk", format="d"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("score:Q", title="Score", format=".3f"),
        ],
    ).add_params(pick)
    layers = [lines, points]

    flagged_points = long[(long["metric"] == "COMET-Kiwi") & long["chunk"].isin(flagged)]
    if not flagged_points.empty:
        layers.append(
            alt.Chart(flagged_points)
            .mark_point(shape="diamond", size=220, color="#d62728", strokeWidth=2)
            .encode(x="chunk:Q", y="score:Q")
        )

    chart = alt.layer(*layers).properties(height=320, width="container")
    event = st.altair_chart(chart, on_select="rerun", selection_mode="pick",
                            key=f"chart_{eval_id}")
    return selected_chunk_from(event)


def render_chunk_details(df, comet_col, flagged, selected):
    st.subheader("Chunk details")
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for i, row in df.iterrows():
        st.markdown(f"<div id='chunk-{i}'></div>", unsafe_allow_html=True)
        label = f"Chunk {i}"
        if comet_col:
            label += f", COMET {row[comet_col]:.3f}"
        if i in flagged:
            label += ", below threshold"
        if i == selected:
            label = f"Selected: {label}"  # new label = new element, so it opens expanded
        with st.expander(label, expanded=(i == selected)):
            for col in text_cols:
                st.markdown(f"**{col}**")
                st.write(str(row[col]))
            if num_cols:
                st.caption(", ".join(
                    f"{col}: {row[col]:.3f}" if isinstance(row[col], float) else f"{col}: {row[col]}"
                    for col in num_cols
                ))


def render_chunks(chunks, eval_id):
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

    selected = None
    if len(scores.columns):
        selected = render_score_chart(scores, flagged, eval_id)
        st.caption("Click a point to open that chunk below. Click empty space to clear.")
    else:
        st.caption("No COMET or MetricX columns found in the chunk data, so there is no chart.")

    def highlight(row):
        if row.name == selected:
            return ["background-color: rgba(60, 130, 255, 0.22)"] * len(row)
        if comet_col and row[comet_col] < COMET_FLAG_THRESHOLD:
            return ["background-color: rgba(255, 75, 75, 0.18)"] * len(row)
        return [""] * len(row)
    st.dataframe(df.style.apply(highlight, axis=1).format(precision=3))

    render_chunk_details(df, comet_col, flagged, selected)
    if selected is not None:
        scroll_to(f"chunk-{selected}")


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
    follow_latest = st.toggle("Always show the latest evaluation", value=True)
    limit = st.slider("Evaluations to list", 5, 50, 20)
    st.button("Refresh")  # any interaction reruns the script and re-fetches

st.title("S2ST evaluation monitor")
st.caption("Results from the evaluation API. Send audio pairs to POST /evaluate to add more.")

try:
    api_get(api_url, "/health")
except requests.RequestException as e:
    st.error(f"Can't reach the API at {api_url}. Check that it is running and the URL is right. ({e})")
    st.stop()

try:
    recent = api_get(api_url, "/evaluations", limit=limit)
except requests.RequestException as e:
    st.error(f"The API is up but listing evaluations failed. Does it have the /evaluations endpoint? ({e})")
    st.stop()

if not recent:
    st.info("No evaluations yet. Send an audio pair to POST /evaluate, then press Refresh.")
    st.stop()

if follow_latest:
    selected = recent[0]["id"]
else:
    labels = {m["id"]: eval_label(m) for m in recent}
    selected = st.selectbox("Evaluation (newest first)", list(labels), format_func=labels.get)

try:
    record = api_get(api_url, f"/evaluations/{selected}")
except requests.RequestException as e:
    st.error(f"Couldn't load evaluation {selected}. ({e})")
    st.stop()

result = record.get("result", {})
st.markdown(
    f"**{record.get('source_name') or 'source'}** to **{record.get('target_name') or 'target'}**, "
    f"evaluated {fmt_time(record.get('created_at'))} (id `{record['id']}`)"
)

render_summary(result.get("summary"))
render_chunks(result.get("chunks"), record["id"])
render_transcripts(result.get("transcripts"), record["id"])
render_unmatched(result.get("unmatched"))

with st.expander("Raw JSON"):
    st.json(record)
