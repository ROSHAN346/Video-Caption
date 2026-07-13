"""Streamlit UI that wraps the existing CLI pipeline.

Usage:
    pip install streamlit opencv-python-headless
    streamlit run app.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

try:
    import config as cfg
except Exception:
    cfg = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _video_short_name(video_path: str) -> str:
    """Derive a short, filesystem-safe output folder name from the video file."""
    stem = Path(video_path).stem
    prefix = stem.split("-", 1)[0]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", prefix).strip("._-")
    return safe or stem


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
                border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 1rem; }
        .keyframe-card { background: #ffffff; border: 1px solid #e2e8f0;
                border-radius: 8px; padding: 0.6rem; text-align: center; }
        .section-gap { margin-top: 2.2rem; }
        .muted { color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Video AMD Pipeline UI")
st.markdown(
    "<div class='muted'>Upload a video directly, or provide a tasks JSON, "
    "run the single-pass pipeline, and explore per-video results & keyframes.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar & Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Allow real-time keys in the sidebar (perfect for cloud deployment!)
    worker_url_input = st.text_input(
        "WORKER_URL", 
        value=os.getenv("WORKER_URL", "https://patient-violet-5828.viratforedu175.workers.dev"),
        help="Cloudflare Worker proxy URL that handles API requests"
    )
    api_key_input = st.text_input(
        "Fireworks API Key (Optional)", 
        value=os.getenv("FIREWORKS_API_KEY", ""), 
        type="password",
        help="Direct Fireworks API key. Optional if WORKER_URL is set"
    )
    
    # Inject selected keys into current environment
    if worker_url_input:
        os.environ["WORKER_URL"] = worker_url_input
    if api_key_input:
        os.environ["FIREWORKS_API_KEY"] = api_key_input
        os.environ["FIREWORKS_TEXT_API_KEY"] = api_key_input
        
    st.divider()
    st.caption(f"Python: `{sys.executable}`")
    
    if cfg is not None:
        with st.expander("🧩 Model & Fine-Tuning Info", expanded=False):
            st.markdown("**🤖 Active Models**")
            model_rows = [
                ("Vision", getattr(cfg, "FIREWORKS_VISION_MODEL", "")),
                ("Text", getattr(cfg, "FIREWORKS_TEXT_MODEL", "")),
                ("CLIP", getattr(cfg, "CLIP_MODEL_NAME", "")),
            ]
            for name, val in model_rows:
                c1, c2 = st.columns([1, 1.8])
                c1.markdown(f"<span class='muted'>{name}</span>", unsafe_allow_html=True)
                c2.code(str(val), language=None)

            st.markdown("**⚙️ Parameters**")
            param_rows = [
                ("Max frames", getattr(cfg, "MAX_FRAMES", "")),
                ("Candidate FPS", getattr(cfg, "CANDIDATE_FPS", "")),
                ("Min dist", getattr(cfg, "EARLY_STOP_MIN_DIST", "")),
                ("Batch size", getattr(cfg, "EMBEDDING_BATCH_SIZE", "")),
            ]
            for name, val in param_rows:
                c1, c2 = st.columns([1, 1.8])
                c1.markdown(f"<span class='muted'>{name}</span>", unsafe_allow_html=True)
                c2.code(str(val), language=None)


# ---------------------------------------------------------------------------
# Results rendering (Batch mode)
# ---------------------------------------------------------------------------
def show_results(results: list, times: dict):
    """Render per-video results plus timing/JSON analysis for batch mode."""
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.header("📊 Results")

    if not results:
        st.warning("No results produced.")
        return

    # Timing analysis
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

    # JSON analysis
    n_captions = 0
    style_counts = {}
    placeholder = 0
    for r in results:
        for style, text in (r.get("captions") or {}).items():
            n_captions += 1
            style_counts[style] = style_counts.get(style, 0) + 1
            if isinstance(text, str) and "Processing failed" in text:
                placeholder += 1

    with st.expander("🔍 Detailed Analysis", expanded=True):
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

    # Per-video results
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
        file_name="results.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["🎥 Single Video Captioning", "📦 Batch JSON Processing"])

# ===========================================================================
# TAB 1: Single Video Captioning (Direct File Upload)
# ===========================================================================
with tab_single:
    st.markdown("### Process a single video file")
    st.markdown("Drag and drop any local video file. The pipeline will split it into scenes, "
                "select keyframes with CLIP, and query the vision-language model.")

    uploaded_video = st.file_uploader(
        "📤 Upload a video file", 
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Max file size ~200MB (Streamlit default limit)"
    )

    if uploaded_video is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.video(uploaded_video)
            
        with col2:
            st.markdown("#### Execution Parameters")
            style_choices = ["All Styles", "formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
            selected_style = st.selectbox(
                "Select Caption Tone", 
                style_choices, 
                index=0,
                help="'All Styles' generates captions in 4 different tones simultaneously!"
            )
            
            run_btn = st.button("🚀 Run Captioning Pipeline", type="primary", use_container_width=True)

        if run_btn:
            # 1. Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_video.read())
                temp_video_path = tmp.name

            short_name = _video_short_name(temp_video_path)
            
            # 2. Build run command
            cmd = [sys.executable, "main.py", temp_video_path, "--reports"]
            if selected_style == "All Styles":
                cmd.append("--all-styles")
            else:
                cmd.extend(["--style", selected_style])

            # 3. Spawn subprocess and show logs
            log_box = st.expander("📜 Process Live Console Logs", expanded=True)
            log_container = log_box.empty()
            log_lines = []

            status_placeholder = st.empty()
            status_placeholder.info("⏳ Initializing pipeline (Loading CLIP weights)...")

            try:
                # Add current worker_url and fireworks_key to subprocess environment
                sub_env = os.environ.copy()
                if worker_url_input:
                    sub_env["WORKER_URL"] = worker_url_input
                if api_key_input:
                    sub_env["FIREWORKS_API_KEY"] = api_key_input
                    sub_env["FIREWORKS_TEXT_API_KEY"] = api_key_input

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=sub_env,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )

                for raw in proc.stdout:
                    line = raw.strip()
                    if line:
                        log_lines.append(line)
                        log_container.code("\n".join(log_lines[-25:]), language="text")
                        
                        # Set user-friendly status updates from logs
                        if "Scene Detection" in line or "detect_scenes" in line:
                            status_placeholder.info("🎬 Detecting scenes & capturing frame candidates...")
                        elif "CLIP" in line or "select_keyframes" in line:
                            status_placeholder.info("⚡ Extracting CLIP embeddings & selecting farthest keyframes...")
                        elif "Analyzing" in line and "vision model" in line:
                            status_placeholder.info("👁️ Analyzing visual keyframes with Fireworks vision...")
                        elif "Generating" in line and "styles" in line:
                            status_placeholder.info("✍️ Generating multi-style captions with text model...")

                proc.wait()
                
                # Cleanup temporary video file
                try:
                    os.unlink(temp_video_path)
                except OSError:
                    pass

                if proc.returncode == 0:
                    status_placeholder.success("✅ Pipeline successfully completed!")
                    
                    # 4. Load outputs
                    output_dir_path = Path("output") / short_name
                    captions_file = output_dir_path / "captions.json"
                    
                    if captions_file.exists():
                        with open(captions_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            st.session_state["single_result"] = data
                            st.session_state["single_short_name"] = short_name
                    else:
                        st.error("Output captions.json was not created by the pipeline.")
                else:
                    status_placeholder.error(f"Pipeline crashed with return code {proc.returncode}.")
            except Exception as e:
                status_placeholder.error(f"Error during execution: {e}")

        # -------------------------------------------------------------------
        # Display Single Video Results (persistent in session state)
        # -------------------------------------------------------------------
        if "single_result" in st.session_state:
            data = st.session_state["single_result"]
            short_name = st.session_state["single_short_name"]
            
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.header("✨ Video Analysis Results")

            # A. Display Captions
            st.markdown("### ✍️ Generated Captions")
            captions = data.get("captions", {})
            if not captions:
                st.warning("No captions generated.")
            else:
                tabs = st.tabs(list(captions.keys()))
                for tab, (style, text) in zip(tabs, captions.items()):
                    with tab:
                        st.markdown(f"<div class='card'><b>{style.upper()}:</b><br><br>{text}</div>", unsafe_allow_html=True)

            # B. Display Extracted Keyframes
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.markdown("### 🖼️ CLIP Keyframe Selection")
            st.markdown("<div class='muted'>These frames were chosen as the most visually distinct and representative representations of the video content.</div>", unsafe_allow_html=True)
            
            keyframes = data.get("keyframes", [])
            if keyframes:
                cols = st.columns(min(len(keyframes), 5))
                for i, kf in enumerate(keyframes):
                    img_path = Path("output") / short_name / kf.get("image_path", "")
                    with cols[i % len(cols)]:
                        st.markdown(f"<div class='keyframe-card'>", unsafe_allow_html=True)
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.info("Image not found on disk")
                        st.markdown(
                            f"<span class='muted'>Scene: {kf.get('scene_id')}</span><br>"
                            f"⏱ <b>{kf.get('timestamp_sec')}s</b><br>"
                            f"<span class='muted' style='font-size:0.8rem;'>Novelty: {kf.get('novelty_score'):.3f}</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

            # C. Display Scene Metadata
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            with st.expander("🎬 Detected Scenes Timeline"):
                scenes = data.get("scenes", [])
                if scenes:
                    # Render nice table
                    st.table(scenes)
                else:
                    st.write("No scenes detected.")

            # D. Download captions.json
            st.download_button(
                "⬇️ Download full captions.json",
                data=json.dumps(data, indent=2),
                file_name="captions.json",
                mime="application/json"
            )


# ===========================================================================
# TAB 2: Batch JSON Processing (Original Competition Mode)
# ===========================================================================
with tab_batch:
    st.markdown("### Run pipeline on a batch tasks JSON")
    st.markdown("Provide a `tasks.json` containing URLs of videos, styles, and task IDs. "
                "The pipeline processes them in parallel and downloads the final `results.json`.")

    uploaded_batch = st.file_uploader("📤 Upload tasks JSON", type=["json"], key="batch_uploader")

    if uploaded_batch is not None:
        try:
            tasks = json.loads(uploaded_batch.getvalue().decode("utf-8"))
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
        if run_col.button("🚀 Run Batch Pipeline", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp:
                input_path = os.path.join(tmp, "tasks.json")
                with open(input_path, "w", encoding="utf-8") as f:
                    f.write(uploaded_batch.getvalue().decode("utf-8"))

                output_path = os.path.join(tmp, "results.json")
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
                    sub_env = os.environ.copy()
                    if worker_url_input:
                        sub_env["WORKER_URL"] = worker_url_input
                    if api_key_input:
                        sub_env["FIREWORKS_API_KEY"] = api_key_input
                        sub_env["FIREWORKS_TEXT_API_KEY"] = api_key_input

                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=sub_env,
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

    if "results" in st.session_state and "times" in st.session_state:
        show_results(st.session_state.results, st.session_state.times)
