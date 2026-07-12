"""Streamlit UI that wraps the existing CLI pipeline.

Usage:
    pip install streamlit
    streamlit run app.py

Does NOT modify any existing project files. Runs the same pipeline as:

    python main.py --input <uploaded.json> --output <results.json>
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import streamlit as st

try:
    import config as cfg
except Exception:
    cfg = None

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Video AMD Pipeline", layout="wide", page_icon="🎬")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.5rem; padding-bottom: 3rem; }
        h1 { font-size: 2.2rem; margin-bottom: 0.2rem; }
        .big-num { font-size: 1.6rem; font-weight: 700; }
        .card { background: #f8f9fb; border: 1px solid #e6e8ec;
                border-radius: 12px; padding: 1.1rem 1.3rem; }
        .section-gap { margin-top: 2.2rem; }
        .muted { color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Video AMD Pipeline UI")
st.markdown(
    "<div class='muted'>Upload a tasks JSON, run the pipeline, and explore "
    "per-video results & timing.</div>",
    unsafe_allow_html=True,
)
st.caption("Equivalent CLI: `python main.py --input tasks.json --output results.json`")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    output_name = st.text_input("Output filename", value="results.json")
    st.caption(f"Interpreter: `{sys.executable}`")
    st.divider()
    st.markdown("**Expected JSON format** (`tasks.json`):")
    st.code(
        '[\n'
        '  {\n'
        '    "task_id": "my_clip",\n'
        '    "video_url": "https://.../clip.mp4",\n'
        '    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]\n'
        '  }\n'
        ']',
        language="json",
    )

    if cfg is not None:
        with st.expander("🧩 Pipeline models & fine-tuning", expanded=True):
            st.markdown("**🤖 Models in use**")
            model_rows = [
                ("Vision model (Fireworks)", getattr(cfg, "FIREWORKS_VISION_MODEL", "")),
                ("Text model (Fireworks)", getattr(cfg, "FIREWORKS_TEXT_MODEL", "")),
                ("CLIP embedding", getattr(cfg, "CLIP_MODEL_NAME", "")),
            ]
            for name, val in model_rows:
                c1, c2 = st.columns([1.3, 2])
                c1.markdown(f"<span class='muted'>{name}</span>", unsafe_allow_html=True)
                c2.code(str(val), language=None)

            st.markdown("**⚙️ Fine-tuning parameters**")
            param_rows = [
                ("Max frames (cap)", getattr(cfg, "MAX_FRAMES", "")),
                ("Candidate FPS", getattr(cfg, "CANDIDATE_FPS", "")),
                ("Early-stop min dist", getattr(cfg, "EARLY_STOP_MIN_DIST", "")),
                ("Embedding batch size", getattr(cfg, "EMBEDDING_BATCH_SIZE", "")),
                ("Scene detector", getattr(cfg, "DETECTOR_CONFIG", {})),
                ("Report styles", ", ".join(getattr(cfg, "REPORT_STYLES", []))),
            ]
            for name, val in param_rows:
                c1, c2 = st.columns([1.3, 2])
                c1.markdown(f"<span class='muted'>{name}</span>", unsafe_allow_html=True)
                c2.code(str(val), language=None)


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------
def show_results(results: list, times: dict):
    """Render per-video results plus timing/JSON analysis."""
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.header("📊 Results")

    if not results:
        st.warning("No results produced.")
        return

    # --- Timing analysis ---
    task_times = {k: v for k, v in times.items() if not k.startswith("__")}
    done_times = [v["time"] for v in task_times.values() if v.get("state") == "done"]
    failed = [tid for tid, v in task_times.items() if v.get("state") == "failed"]
    overall = times.get("__total__", {}).get("time")
    total_time = overall if overall is not None else sum(done_times)
    avg_time = (total_time / len(task_times)) if task_times else 0.0

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎞 Tasks", len(results))
    m2.metric("⏱ Total time", f"{total_time:.1f}s")
    m3.metric("⏱ Avg / video", f"{avg_time:.1f}s")
    m4.metric("❌ Failed", len(failed))
    st.markdown("</div>", unsafe_allow_html=True)

    # --- JSON analysis ---
    n_captions = 0
    style_counts = {}
    placeholder = 0
    for r in results:
        for style, text in (r.get("captions") or {}).items():
            n_captions += 1
            style_counts[style] = style_counts.get(style, 0) + 1
            if isinstance(text, str) and "Processing failed" in text:
                placeholder += 1

    with st.expander("🔍 JSON analysis", expanded=True):
        st.markdown(
            f"- **Total tasks:** {len(results)}\n"
            f"- **Total captions generated:** {n_captions}\n"
            f"- **Placeholder/failed captions:** {placeholder}\n"
            f"- **Total time:** {total_time:.1f}s (overall) | "
            f"**Avg/video:** {avg_time:.1f}s\n"
            f"- **Per-video times:** "
            + ", ".join(f"{k}={v['time']:.1f}s" for k, v in task_times.items()
                        if v.get('state') == 'done')
        )
        if style_counts:
            st.markdown("**Captions per style:**")
            st.bar_chart(style_counts)

    # --- Per-video results ---
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.subheader("🎥 Per-video captions")
    task_ids = [r.get("task_id", f"task_{i}") for i, r in enumerate(results)]
    selected = st.selectbox("Select a video / task", task_ids, index=0)

    idx = task_ids.index(selected)
    result = results[idx]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### Task: `{selected}`")
    if selected in times:
        t = times[selected]
        if t.get("state") == "done":
            st.markdown(f"<span class='muted'>⏱ Processing time: {t['time']:.1f}s</span>",
                        unsafe_allow_html=True)
        elif t.get("state") == "failed":
            st.markdown("<span class='muted'>⏱ Processing: failed</span>",
                        unsafe_allow_html=True)

    captions = result.get("captions", {})
    if not captions:
        st.info("No captions were generated for this task.")
    else:
        style_names = list(captions.keys())
        tabs = st.tabs(style_names)
        for tab, style in zip(tabs, style_names):
            with tab:
                text = captions[style]
                if isinstance(text, str):
                    st.write(text)
                else:
                    st.json(text)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download results.json",
        data=json.dumps(results, indent=2),
        file_name=output_name,
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Upload + run
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("📤 Upload tasks JSON", type=["json"])

if uploaded is not None:
    try:
        tasks = json.loads(uploaded.getvalue().decode("utf-8"))
        if not isinstance(tasks, list):
            st.error("Top-level JSON must be a list of tasks.")
            st.stop()
        st.success(f"✅ Loaded {len(tasks)} task(s).")
        with st.expander("👀 Preview uploaded JSON"):
            st.json(tasks)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    run_col, _ = st.columns([1, 3])
    if run_col.button("🚀 Run pipeline", type="primary", use_container_width=True):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = os.path.join(tmp, "tasks.json")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(uploaded.getvalue().decode("utf-8"))

            output_path = os.path.join(tmp, output_name)
            cmd = [sys.executable, "main.py", "--input", input_path, "--output", output_path]

            status = {}
            status_box = st.empty()
            log_box = st.expander("📜 Full process log", expanded=False)
            log_text = []

            def render_status():
                rows = []
                if status:
                    for tid, info in status.items():
                        if info["state"] == "running":
                            dur = "⏳ processing..."
                        elif info["state"] == "done":
                            dur = f"{info['time']:.1f}s"
                        else:
                            dur = "❌ failed"
                        rows.append({"Task": tid, "Status": info["state"], "Time": dur})
                if not rows:
                    status_box.info("Waiting for tasks to start...")
                else:
                    status_box.table(rows)

            render_status()

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )

                re_start = re.compile(r"=== Processing task (.+?) ===")
                re_done = re.compile(r"Task (.+?) completed in ([\d.]+)s")
                re_fail = re.compile(r"Task (.+?) failed")
                re_all = re.compile(r"All tasks completed in ([\d.]+)s")

                for raw in proc.stdout:
                    line = raw.strip()
                    if not line:
                        continue
                    log_text.append(line)

                    m = re_start.search(line)
                    if m:
                        status[m.group(1)] = {"state": "running"}
                        render_status()
                        continue
                    m = re_done.search(line)
                    if m:
                        status[m.group(1)] = {"state": "done", "time": float(m.group(2))}
                        render_status()
                        continue
                    m = re_fail.search(line)
                    if m:
                        status[m.group(1)] = {"state": "failed"}
                        render_status()
                        continue
                    m = re_all.search(line)
                    if m:
                        status["__overall__"] = {"state": "done", "time": float(m.group(1))}
                        continue

                proc.wait()
                with log_box:
                    st.code("\n".join(log_text), language="text")

                if proc.returncode == 0:
                    st.success("✅ Pipeline finished successfully.")
                    with open(output_path, "r", encoding="utf-8") as f:
                        results = json.load(f)
                    overall = status.pop("__overall__", None)
                    if overall:
                        status["__total__"] = overall
                    st.session_state.results = results
                    st.session_state.times = status
                    show_results(results, status)
                else:
                    st.error(f"Pipeline exited with code {proc.returncode}.")
            except FileNotFoundError:
                st.error(f"Python interpreter not found: {sys.executable}")
else:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.info("👆 Upload a tasks JSON file to get started.")

if "results" in st.session_state and "times" in st.session_state:
    show_results(st.session_state.results, st.session_state.times)
